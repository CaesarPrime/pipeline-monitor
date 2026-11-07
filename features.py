"""
features.py
-----------
Extracts the 128-element feature vector described in Chapter Three:

  32 vibration spectral band energies (10 Hz - 2 kHz)
  40 acoustic Mel-frequency cepstral coefficients (50 Hz - 6 kHz)
  16 light statistics (mean, variance, minimum, maximum for each of 4 photodiodes)
  40 cross-modal features (vibration-acoustic envelope correlation at 40 time lags, 0-100 ms)
"""

import numpy as np
from scipy.fft import dct

from simulator import FS, N

N_FEATURES = 128
GROUPS = {
    "vibration": slice(0, 32),
    "acoustic": slice(32, 72),
    "light": slice(72, 88),
    "cross_modal": slice(88, 128),
}

_freqs = np.fft.rfftfreq(N, 1 / FS)
_vib_edges = np.geomspace(10, 2000, 33)


def _mel(f):
    return 2595 * np.log10(1 + f / 700)


def _mel_inv(m):
    return 700 * (10 ** (m / 2595) - 1)


def _mel_filterbank(n_filters=40, fmin=50, fmax=6000):
    pts = _mel_inv(np.linspace(_mel(fmin), _mel(fmax), n_filters + 2))
    fb = np.zeros((n_filters, len(_freqs)))
    for i in range(n_filters):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (_freqs >= lo) & (_freqs <= c)
        dn = (_freqs > c) & (_freqs <= hi)
        fb[i, up] = (_freqs[up] - lo) / (c - lo + 1e-9)
        fb[i, dn] = (hi - _freqs[dn]) / (hi - c + 1e-9)
    return fb


_MEL_FB = _mel_filterbank()


def vibration_features(vib):
    p = np.abs(np.fft.rfft(vib * np.hanning(N))) ** 2
    out = np.empty(32)
    for i in range(32):
        m = (_freqs >= _vib_edges[i]) & (_freqs < _vib_edges[i + 1])
        out[i] = np.log10(p[m].sum() + 1e-6)
    return out


def acoustic_features(aco):
    p = np.abs(np.fft.rfft(aco * np.hanning(N))) ** 2
    mel = np.log(_MEL_FB @ p + 1e-6)
    return dct(mel, type=2, norm="ortho")[:40]


def light_features(light):
    return np.concatenate([light.mean(1), light.var(1) * 100, light.min(1), light.max(1)])


def cross_modal_features(vib, aco):
    hop = 32                                   # 2.5 ms per envelope step
    ev = np.abs(vib[: N // hop * hop]).reshape(-1, hop).mean(1)
    ea = np.abs(aco[: N // hop * hop]).reshape(-1, hop).mean(1)
    ev = (ev - ev.mean()) / (ev.std() + 1e-9)
    ea = (ea - ea.mean()) / (ea.std() + 1e-9)
    out = np.empty(40)
    for lag in range(40):                      # 0 - 97.5 ms
        a, b = ev[: len(ev) - lag], ea[lag:]
        out[lag] = float(np.mean(a * b))
    return out


def extract(event):
    """128-element feature vector for one simulated event."""
    v, a, l = event["vibration"], event["acoustic"], event["light"]
    return np.concatenate([
        vibration_features(v), acoustic_features(a), light_features(l), cross_modal_features(v, a)
    ]).astype(np.float32)


def extract_many(events):
    return np.stack([extract(e) for e in events]), np.array([e["label"] for e in events])
