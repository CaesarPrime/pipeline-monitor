"""
train.py
--------
Builds the simulated dataset, trains the proposed TCN and the baseline models,
and saves the models and the evaluation results used by the dashboard.

    python train.py            # full run (1,000 events per class)
    python train.py --quick    # fast check (200 events per class, few epochs)

Outputs
  models/   scaler.joblib, rf.joblib, svm.joblib, tcn.keras, tcn_int8.tflite, model_data.h
  results/  metrics.json, confusion_<model>.png, ablation.png, dataset_summary.json
"""

import argparse
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import GROUPS, N_FEATURES, extract_many
from simulator import CLASSES, generate_dataset

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:          # the dashboard still works with the scikit-learn models
    tf = None

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
SEED = 42
np.random.seed(SEED)


# ----------------------------------------------------------------- helpers
def time_per_sample_ms(predict, X, repeats=200):
    x = X[:1]
    predict(x)                                   # warm-up
    t = time.perf_counter()
    for _ in range(repeats):
        predict(x)
    return (time.perf_counter() - t) / repeats * 1000


def plot_confusion(cm, name):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm, cmap="Greys")
    short = ["Noise", "Hot-tap", "Refining", "Sabotage", "Vehicle"]
    ax.set_xticks(range(5), short, rotation=30)
    ax.set_yticks(range(5), short)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix: {name}")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(f"results/confusion_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)


def evaluate(name, y_true, y_pred, latency_ms, size_kb, results):
    cm = confusion_matrix(y_true, y_pred, labels=range(5))
    plot_confusion(cm, name)
    results["models"][name] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "inference_ms": round(float(latency_ms), 3),
        "size_kb": round(float(size_kb), 1),
        "per_class": classification_report(y_true, y_pred, labels=range(5),
                                           target_names=CLASSES, output_dict=True, zero_division=0),
        "confusion_matrix": cm.tolist(),
    }
    print(f"{name:15s} macro-F1={results['models'][name]['macro_f1']:.4f} "
          f"latency={latency_ms:.2f} ms size={size_kb:.0f} kB")


# ----------------------------------------------------------------- Keras models
def build_tcn(num_classes=5):
    """Same layers as Chapter Three: dilated causal Conv1D 64-128-256-512, dense 256-128."""
    inp = keras.Input(shape=(N_FEATURES, 1))
    x = inp
    for filters, d in [(64, 1), (128, 2), (256, 4), (512, 8)]:
        x = keras.layers.Conv1D(filters, 3, dilation_rate=d, padding="causal")(x)
        x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="tcn")


def build_rnn(kind, num_classes=5):
    layer = keras.layers.LSTM if kind == "LSTM" else keras.layers.GRU
    inp = keras.Input(shape=(N_FEATURES, 1))
    x = layer(64)(inp)
    x = keras.layers.Dense(64, activation="relu")(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name=kind.lower())


def fit_keras(model, Xtr, ytr, Xva, yva, epochs):
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(Xtr[..., None], ytr, validation_data=(Xva[..., None], yva),
                     epochs=epochs, batch_size=64, verbose=2,
                     callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    return {k: [float(v) for v in vals] for k, vals in hist.history.items()}


def export_tflite_int8(model, Xrep):
    def rep():
        for i in range(min(200, len(Xrep))):
            yield [Xrep[i:i + 1, :, None].astype(np.float32)]
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
    except Exception:
        model.export("models/tcn_savedmodel")
        conv = tf.lite.TFLiteConverter.from_saved_model("models/tcn_savedmodel")
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    data = conv.convert()
    with open("models/tcn_int8.tflite", "wb") as f:
        f.write(data)
    # C array for the ESP32 firmware (same as: xxd -i tcn_int8.tflite > model_data.h)
    with open("models/model_data.h", "w") as f:
        f.write("// Generated by train.py - quantized TCN for TensorFlow Lite Micro\n")
        f.write("alignas(16) const unsigned char g_model[] = {\n")
        for i in range(0, len(data), 12):
            f.write("  " + ", ".join(f"0x{b:02x}" for b in data[i:i + 12]) + ",\n")
        f.write("};\n")
        f.write(f"const unsigned int g_model_len = {len(data)};\n")
    return data


def tflite_accuracy(data, Xte):
    interp = tf.lite.Interpreter(model_content=data)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    s_in, z_in = inp["quantization"]
    preds = []
    for x in Xte:
        q = np.clip(np.round(x / s_in + z_in), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q.reshape(1, N_FEATURES, 1))
        interp.invoke()
        preds.append(int(np.argmax(interp.get_tensor(out["index"])[0])))
    return np.array(preds)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--per-class", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    per_class = args.per_class or (200 if args.quick else 1000)
    epochs = args.epochs or (5 if args.quick else 20)

    print(f"Simulating {per_class} events per class ...")
    events = generate_dataset(per_class, seed=SEED)
    X, y = extract_many(events)
    night = np.array([e["night"] for e in events])

    # 70 / 15 / 15 split
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, random_state=SEED, stratify=ytmp)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
    joblib.dump(scaler, "models/scaler.joblib")

    results = {"classes": CLASSES, "models": {}, "ablation": {}, "history": {},
               "dataset": {"events": int(len(y)), "per_class": per_class, "features": N_FEATURES,
                           "train": int(len(ytr)), "validation": int(len(yva)), "test": int(len(yte)),
                           "night_share": round(float(night.mean()), 3)}}

    # ---- simple threshold baseline (imitates a commercial vibration-threshold system)
    vib_energy = Xte[:, GROUPS["vibration"]].max(1)
    thr = np.percentile(Xtr[ytr == 0][:, GROUPS["vibration"]].max(1), 90)
    y_thr = np.where(vib_energy > thr, 1, 0)      # can only say "threat" (as hot-tapping) or "no threat"
    evaluate("Threshold", yte, y_thr, 0.01, 0, results)

    # ---- SVM and Random Forest
    svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED).fit(Xtr_s, ytr)
    joblib.dump(svm, "models/svm.joblib")
    evaluate("SVM (RBF)", yte, svm.predict(Xte_s), time_per_sample_ms(svm.predict, Xte_s),
             os.path.getsize("models/svm.joblib") / 1024, results)

    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Xtr_s, ytr)
    joblib.dump(rf, "models/rf.joblib")
    rf.set_params(n_jobs=1)
    evaluate("Random Forest", yte, rf.predict(Xte_s), time_per_sample_ms(rf.predict, Xte_s, 50),
             os.path.getsize("models/rf.joblib") / 1024, results)

    # ---- ablation study (Random Forest, so all seven configurations run quickly)
    configs = {
        "Vibration only": ["vibration"], "Acoustic only": ["acoustic"], "Light only": ["light"],
        "Vibration + Acoustic": ["vibration", "acoustic", "cross_modal"],
        "Vibration + Light": ["vibration", "light"], "Acoustic + Light": ["acoustic", "light"],
        "Tri-modal (all)": ["vibration", "acoustic", "light", "cross_modal"],
    }
    for name, groups in configs.items():
        cols = np.concatenate([np.arange(N_FEATURES)[GROUPS[g]] for g in groups])
        m = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Xtr_s[:, cols], ytr)
        results["ablation"][name] = round(float(f1_score(yte, m.predict(Xte_s[:, cols]), average="macro")), 4)
        print(f"  ablation {name:22s} {results['ablation'][name]:.4f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(results["ablation"]), list(results["ablation"].values()), color="0.35", edgecolor="black")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1-score")
    ax.set_title("Ablation study: contribution of each sensing modality")
    fig.tight_layout()
    fig.savefig("results/ablation.png", dpi=150)
    plt.close(fig)

    # ---- deep models
    if tf is not None:
        tf.random.set_seed(SEED)
        for kind in ["LSTM", "GRU"]:
            m = build_rnn(kind)
            results["history"][kind] = fit_keras(m, Xtr_s, ytr, Xva_s, yva, epochs)
            pred = np.argmax(m.predict(Xte_s[..., None], verbose=0), 1)
            lat = time_per_sample_ms(lambda x: m(x[..., None], training=False), Xte_s, 50)
            evaluate(kind, yte, pred, lat, m.count_params() * 4 / 1024, results)

        tcn = build_tcn()
        results["tcn_parameters"] = int(tcn.count_params())
        results["history"]["TCN"] = fit_keras(tcn, Xtr_s, ytr, Xva_s, yva, epochs)
        tcn.save("models/tcn.keras")
        pred = np.argmax(tcn.predict(Xte_s[..., None], verbose=0), 1)
        lat = time_per_sample_ms(lambda x: tcn(x[..., None], training=False), Xte_s, 50)
        evaluate("TCN (float32)", yte, pred, lat, tcn.count_params() * 4 / 1024, results)
        try:
            data = export_tflite_int8(tcn, Xtr_s)
            q_pred = tflite_accuracy(data, Xte_s)
            evaluate("TCN (int8, edge)", yte, q_pred, float("nan"), len(data) / 1024, results)
            results["models"]["TCN (int8, edge)"]["inference_ms"] = None   # measure on the ESP32
        except Exception as exc:                                         # conversion is optional
            print("TFLite export skipped:", exc)
        results["deployed_model"] = "TCN"
    else:
        print("TensorFlow not installed - deep models skipped; Random Forest is used by the dashboard.")
        results["deployed_model"] = "Random Forest"

    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved models/ and results/.")


if __name__ == "__main__":
    main()
