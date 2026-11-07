"""
simulator.py
------------
Generates simulated multi-modal sensor signals (vibration, acoustic and light)
for the five event classes used in the study.

The signals are synthetic. They imitate the physical signatures described in
Chapter Two and Chapter Three of the project (drilling vibration at 800-1,500 Hz,
engine rumble, fire crackle, torch light, rain and thunder, and so on), with
random variation in distance, loudness, background noise and overlapping
environmental activity so that the classification task is realistic rather
than trivial.
"""

import numpy as np

FS = 12_800            # sampling rate (Hz) for vibration and acoustic channels
DURATION = 0.5         # seconds per analysis window
N = int(FS * DURATION) # samples per window
N_LIGHT = 4            # four photodiodes

CLASSES = [
    "Environmental Noise",
    "Illegal Hot-Tapping",
    "Artisanal Refining",
    "Sabotage (Torch-Cutting)",
    "Vehicle/Boat Activity",
]
THREAT_CLASSES = {1, 2, 3, 4}   # class 0 (environmental noise) never raises an alert


def _t():
    return np.arange(N) / FS


def _bandlimited_noise(rng, low, high, n=N):
    """White noise filtered to [low, high] Hz in the frequency domain."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    spec[(f < low) | (f > high)] = 0
    x = np.fft.irfft(spec, n)
    return x / (np.std(x) + 1e-9)


def _crackle(rng, rate, n=N):
    """Random impulsive clicks (fire crackle, metal clanging, raindrops)."""
    x = np.zeros(n)
    k = rng.poisson(rate * DURATION)
    for pos in rng.integers(0, n - 64, size=k):
        L = rng.integers(8, 64)
        x[pos:pos + L] += rng.uniform(0.5, 1.5) * np.exp(-np.arange(L) / (L / 4)) * rng.choice([-1, 1])
    return x


def _envelope(rng, kind):
    t = _t()
    if kind == "steady":
        return 1 + 0.1 * np.sin(2 * np.pi * rng.uniform(0.5, 3) * t)
    if kind == "bursty":
        env = np.zeros(N)
        start = 0
        while start < N:
            on = rng.integers(N // 10, N // 3)
            off = rng.integers(N // 20, N // 6)
            env[start:start + on] = 1
            start += on + off
        return np.convolve(env, np.ones(200) / 200, mode="same")
    if kind == "ramp":   # approaching / passing vehicle or boat
        c = rng.uniform(0.2, 0.8)
        return np.exp(-((t / DURATION - c) ** 2) / rng.uniform(0.05, 0.2))
    return np.ones(N)


def background(rng, night):
    """Background vibration, sound and light that is always present."""
    vib = rng.uniform(0.05, 0.2) * _bandlimited_noise(rng, 10, 2000)
    aco = rng.uniform(0.05, 0.25) * _bandlimited_noise(rng, 50, 6000)
    base_light = rng.uniform(0.02, 0.08) if night else rng.uniform(0.5, 0.9)
    light = base_light + 0.01 * rng.standard_normal((N_LIGHT, N // 64))
    return vib, aco, light


def add_environment(rng, vib, aco, light, strength=1.0):
    """Rain, thunder, wind, birds or a distant boat - harmless activity."""
    kind = rng.choice(["rain", "thunder", "wind", "birds", "distant_engine"])
    s = strength * rng.uniform(0.4, 1.2)
    if kind == "rain":
        aco += s * 0.6 * _crackle(rng, 900) + s * 0.3 * _bandlimited_noise(rng, 1000, 6000)
        vib += s * 0.15 * _crackle(rng, 600)
    elif kind == "thunder":
        env = _envelope(rng, "ramp")
        aco += s * 1.2 * env * _bandlimited_noise(rng, 50, 300)
        vib += s * 0.5 * env * _bandlimited_noise(rng, 10, 120)
        if rng.random() < 0.5:   # lightning flash
            i = rng.integers(0, light.shape[1] - 3)
            light[:, i:i + 2] += rng.uniform(0.3, 0.8)
    elif kind == "wind":
        aco += s * 0.5 * _bandlimited_noise(rng, 50, 800) * _envelope(rng, "steady")
    elif kind == "birds":
        t = _t()
        f0 = rng.uniform(2500, 5000)
        aco += s * 0.4 * np.sin(2 * np.pi * (f0 + 600 * np.sin(2 * np.pi * 8 * t)) * t) * _envelope(rng, "bursty")
    else:  # distant engine: weak, looks a little like a vehicle/boat
        f0 = rng.uniform(60, 140)
        t = _t()
        aco += s * 0.25 * (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t))
        vib += s * 0.08 * np.sin(2 * np.pi * f0 * t)
    return kind


def generate_event(label, rng=None, night=None, distance=None):
    """
    Generate one 0.5 s multi-modal window for a given class label.

    Returns a dict with:
      vibration (N,), acoustic (N,), light (4, N/64), label, night, distance_m
    """
    rng = rng or np.random.default_rng()
    night = bool(rng.random() < 0.6) if night is None else night   # most theft happens at night
    distance = rng.uniform(1, 120) if distance is None else distance
    att = 1.0 / (1 + distance / 20)               # signal weakens with distance
    t = _t()
    vib, aco, light = background(rng, night)

    if label == 1:   # illegal hot-tapping: drilling into the pipe
        f0 = rng.uniform(800, 1500)
        env = _envelope(rng, rng.choice(["steady", "bursty"]))
        drill = np.sin(2 * np.pi * f0 * t + rng.uniform(0, 6)) + 0.4 * np.sin(2 * np.pi * 2 * f0 * t)
        vib += att * rng.uniform(0.4, 1.2) * env * (drill + 0.3 * _bandlimited_noise(rng, f0 - 150, f0 + 150))
        aco += att * rng.uniform(0.3, 0.9) * env * (drill + 0.4 * _bandlimited_noise(rng, 1500, 5000))
        if night and rng.random() < 0.4:          # head torch
            light += att * rng.uniform(0.05, 0.25)
    elif label == 2:  # artisanal refining: fire, boiling, metal clanging
        aco += att * rng.uniform(0.4, 1.0) * (_crackle(rng, 400) + 0.4 * _bandlimited_noise(rng, 100, 1200))
        vib += att * rng.uniform(0.05, 0.25) * _crackle(rng, 60)
        flicker = 1 + 0.3 * np.sin(2 * np.pi * rng.uniform(4, 12) * np.arange(light.shape[1]) / (FS / 64))
        light += att * rng.uniform(0.2, 0.6) * flicker * (1.0 if night else 0.3)
    elif label == 3:  # sabotage by torch / angle-grinder cutting
        f0 = rng.uniform(2500, 4500)
        env = _envelope(rng, "bursty")
        aco += att * rng.uniform(0.5, 1.3) * env * (0.6 * _bandlimited_noise(rng, f0 - 800, f0 + 800) + 0.4 * np.sin(2 * np.pi * f0 * t))
        vib += att * rng.uniform(0.3, 0.9) * env * _bandlimited_noise(rng, 1500, 2000)
        flash = rng.uniform(0.4, 1.0) * (1.0 if night else 0.35)
        light += att * flash * np.repeat(env[::64][None, :], N_LIGHT, axis=0)[:, :light.shape[1]]
    elif label == 4:  # vehicle or boat passing
        f0 = rng.uniform(40, 160)
        env = _envelope(rng, "ramp")
        engine = np.sin(2 * np.pi * f0 * t) + 0.6 * np.sin(2 * np.pi * 2 * f0 * t) + 0.3 * np.sin(2 * np.pi * 3 * f0 * t)
        aco += att * rng.uniform(0.5, 1.2) * env * (engine + 0.3 * _bandlimited_noise(rng, 50, 900))
        vib += att * rng.uniform(0.2, 0.6) * env * (np.sin(2 * np.pi * f0 * t) + 0.3 * _bandlimited_noise(rng, 10, 300))
        if night and rng.random() < 0.7:          # headlights
            light += att * rng.uniform(0.2, 0.6) * env[::64][None, :light.shape[1]]
    # label 0 (environmental noise) only receives environmental activity below

    # environmental activity: always for class 0, sometimes overlapping real threats
    if label == 0 or rng.random() < 0.5:
        add_environment(rng, vib, aco, light, strength=1.0 if label == 0 else 0.6)

    # sensor noise
    vib += 0.02 * rng.standard_normal(N)
    aco += 0.02 * rng.standard_normal(N)
    light = np.clip(light + 0.005 * rng.standard_normal(light.shape), 0, None)

    return {"vibration": vib.astype(np.float32), "acoustic": aco.astype(np.float32),
            "light": light.astype(np.float32), "label": int(label),
            "night": night, "distance_m": float(distance)}


def generate_dataset(n_per_class=1000, seed=42):
    """Balanced dataset of simulated events (list of dicts)."""
    rng = np.random.default_rng(seed)
    events = []
    for label in range(len(CLASSES)):
        for _ in range(n_per_class):
            events.append(generate_event(label, rng))
    order = rng.permutation(len(events))
    return [events[i] for i in order]
