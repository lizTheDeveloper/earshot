"""Synthetic test sounds. All mono float32 at SR."""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).parent))
SR = 16000
C = 343.0


def _smooth_noise(rng, n, sigma_samples):
    """Slowly varying unit noise: smoothed at 100 Hz and interpolated (fast)."""
    step = SR // 100
    m = n // step + 2
    y = gaussian_filter1d(rng.normal(size=m), max(sigma_samples / step, 0.5))
    x = np.interp(np.arange(n) / step, np.arange(m), y)
    return x / (np.std(x) + 1e-12)


def _comb(f_inst, amps, sr=SR):
    """Harmonic comb following instantaneous fundamental f_inst (array, Hz)."""
    phase = 2 * np.pi * np.cumsum(f_inst) / sr
    out = np.zeros_like(f_inst)
    for k, a in enumerate(amps, start=1):
        ok = k * f_inst < sr / 2 - 200
        out += a * np.sin(k * phase) * ok
    return out


def noise(seconds, rng, level=0.02):
    """Pinkish background noise."""
    n = int(seconds * SR)
    w = rng.normal(size=n)
    f = np.fft.rfft(w)
    f /= np.sqrt(np.maximum(np.fft.rfftfreq(n, 1 / SR), 20.0))
    x = np.fft.irfft(f, n)
    return (level * x / np.std(x)).astype(np.float32)


def drone(seconds, rng, f0=180.0, rotors=4, harmonics=18, level=0.02, wobble=0.012, doppler_speed=0.0,
          closest_m=30.0):
    """A multirotor: several close combs (one per rotor), each wobbling slowly (attitude control).
    With doppler_speed, it flies past in a straight line, closest at the clip's midpoint."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    if doppler_speed:
        x = doppler_speed * (t - seconds / 2)
        r = np.hypot(x, closest_m)
        v_radial = doppler_speed * x / r
        dop = C / (C + v_radial)
        gain = closest_m / r
    else:
        dop = np.ones(n)
        gain = np.ones(n)
    out = np.zeros(n)
    amps = [1.0 / k ** 0.6 for k in range(1, harmonics + 1)]
    for i in range(rotors):
        fi = f0 * (1 + rng.uniform(-0.03, 0.03))
        wob = 1 + wobble * _smooth_noise(rng, n, SR * 0.3)
        out += _comb(fi * wob * dop, amps) * (1 + 0.2 * _smooth_noise(rng, n, SR * 0.5))
    out *= gain
    return (level * out / np.std(out)).astype(np.float32)


def engine(seconds, rng, f0=120.0, level=0.02):
    """A small four-stroke engine: comb at the firing frequency f0 plus strong half-orders (f0/2 multiples)."""
    n = int(seconds * SR)
    f = f0 * (1 + 0.004 * _smooth_noise(rng, n, SR * 1.0))
    half = _comb(f / 2, [0.8 / k ** 0.5 for k in range(1, 40)])
    full = _comb(f, [1.0 / k ** 0.5 for k in range(1, 20)])
    out = half + full
    return (level * out / np.std(out)).astype(np.float32)


def hum(seconds, mains=60.0, level=0.02):
    n = int(seconds * SR)
    out = _comb(np.full(n, mains), [1.0 / k for k in range(1, 30)])
    return (level * out / np.std(out)).astype(np.float32)


def fan(seconds, f0=97.3, level=0.02):
    """A motor running at a constant speed: a perfectly steady comb."""
    n = int(seconds * SR)
    out = _comb(np.full(n, f0), [1.0 / k ** 0.7 for k in range(1, 25)])
    return (level * out / np.std(out)).astype(np.float32)


def speech_like(seconds, rng, level=0.03):
    """Voiced syllables: 0.15–0.5 s, pitch 90–240 Hz moving fast, formant-shaped harmonics, with gaps."""
    n = int(seconds * SR)
    out = np.zeros(n)
    i = 0
    while i < n:
        dur = int(rng.uniform(0.15, 0.5) * SR)
        gap = int(rng.uniform(0.05, 0.4) * SR)
        if i + dur >= n:
            break
        tt = np.arange(dur) / SR
        p0 = rng.uniform(90, 240)
        f = p0 * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(2, 6) * tt + rng.uniform(0, 6)))
        fm = rng.uniform(400, 900)
        amps = [np.exp(-((k * p0 - fm) / 600) ** 2) + 0.1 for k in range(1, 30)]
        seg = _comb(f, amps) * np.hanning(dur)
        out[i:i + dur] += seg
        i += dur + gap
    return (level * out / (np.std(out) + 1e-12)).astype(np.float32)
