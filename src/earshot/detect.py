"""Streaming multirotor-drone detector: harmonic comb tracking on whitened spectra.

Audio comes in as blocks of mono float samples and is kept only as long as one analysis frame
needs it (1 s). What leaves this module is an Event: a time span, a pitch, a harmonic count, a
level and a confidence. No audio and no spectra are stored or returned.

Per frame (1.0 s Hann, 0.25 s hop, 1 Hz bins):
  1. log spectrum 40–4200 Hz, minus a running median across ~41 Hz (the local noise floor);
  2. minus a slow per-bin background (sounds that never stop fade out);
  3. harmonic score for f0 = 80–600 Hz: sum over k·f0 ≤ 4 kHz (k ≤ 20) of clip(level − 6 dB, −6, 14),
     so a hit adds and a miss costs (this stops octave-down and octave-up errors);
  4. keep up to 3 candidate pitches that aren't the same sound (within 8 %, or at a 2:1/3:1 ratio).
Candidates are linked into tracks (≤ 7 % pitch change per hop: Doppler, throttle, and the
alternation between a multirotor's slightly different rotors). A track lasting ≥ 2.5 s with ≥ 4
harmonics is scored, and becomes an Event if its confidence is ≥ the threshold. Confidence drops for
a steady pitch (a motor on mains, a fan, hum: drones are never perfectly steady), for strong energy
between the harmonics (a piston engine's half-orders), and for a weak or short comb.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter

FRAME_S = 1.0
HOP_S = 0.25
F_LO, F_HI = 40.0, 4200.0
F0_LO, F0_HI, F0_STEP = 80.0, 600.0, 0.25
MAX_HARM = 20
HIT_DB = 6.0
MIN_HITS = 4
MIN_SCORE = 12.0
TRACK_JUMP = 0.07
GAP_S = 1.5
MIN_EVENT_S = 2.5
MIN_COVERAGE = 0.6  # share of an event's hops in which its comb was actually found
BACKGROUND_S = 120.0
MIN_WOBBLE = 0.0015  # 0.15 %: a mains-locked or steady-speed motor is steadier than this; a drone isn't
FULL_WOBBLE = 0.004
CONFIDENCE = 0.5


@dataclass
class Event:
    start_s: float
    end_s: float
    f0_hz: float
    harmonics: int
    level_db: float
    wobble: float
    confidence: float

    def to_record(self, node: str, t0_epoch: float) -> dict:
        def iso(s: float) -> str:
            return datetime.fromtimestamp(t0_epoch + s, timezone.utc).isoformat(timespec="milliseconds")

        return {"start": iso(self.start_s), "end": iso(self.end_s), "node": node,
                "confidence": round(self.confidence, 3), "f0_hz": round(self.f0_hz, 1),
                "harmonics": int(self.harmonics), "level_db": round(self.level_db, 1),
                "wobble": round(self.wobble, 4)}


@dataclass
class _Cand:
    f0: float
    score: float
    hits: int
    half: float  # share of in-between positions of the best finer comb (f0/n) that have a peak (0 = none)


@dataclass
class _Track:
    t: list = field(default_factory=list)
    f0: list = field(default_factory=list)
    hits: list = field(default_factory=list)
    score: list = field(default_factory=list)
    half: list = field(default_factory=list)
    level: list = field(default_factory=list)
    last: int = 0


class Detector:
    def __init__(self, sr: int, threshold: float = CONFIDENCE):
        self.sr = sr
        self.threshold = threshold
        self.frame = int(round(FRAME_S * sr))
        self.hop = int(round(HOP_S * sr))
        self.win = np.hanning(self.frame).astype(np.float32)
        freqs = np.fft.rfftfreq(self.frame, 1 / sr)
        self.band = (freqs >= F_LO) & (freqs <= F_HI)
        self.f = freqs[self.band]
        self.df = freqs[1]
        self.f0s = np.arange(F0_LO, F0_HI + 1e-9, F0_STEP)
        K = np.minimum(MAX_HARM, np.floor(4000.0 / self.f0s)).astype(int)
        ks = np.arange(1, MAX_HARM + 1)
        self.kmask = ks[None, :] <= K[:, None]
        self.hidx = np.clip(np.rint((self.f0s[:, None] * ks[None, :] - self.f[0]) / self.df).astype(int), 0, len(self.f) - 1)
        self.bg = np.zeros(len(self.f), np.float32)
        self.alpha = HOP_S / BACKGROUND_S
        self.buf = np.zeros(0, np.float32)
        self.n = 0  # frames processed
        self.tracks: list[_Track] = []
        self.pending: list[Event] = []  # finished tracks, held until nothing still running could join them
        self.gap = int(round(GAP_S / HOP_S))

    def buffered_samples(self) -> int:
        return len(self.buf)

    def feed(self, x: np.ndarray) -> list[Event]:
        out: list[Event] = []
        x = np.asarray(x, np.float32).ravel()
        i = 0
        while i < len(x):
            take = min(len(x) - i, self.frame - len(self.buf)) if len(self.buf) < self.frame else 0
            if take:
                self.buf = np.concatenate([self.buf, x[i:i + take]])
                i += take
            if len(self.buf) >= self.frame:
                out += self._frame(self.buf[: self.frame])
                self.buf = self.buf[self.hop:].copy()
        return out

    def flush(self) -> list[Event]:
        for tr in self.tracks:
            self._hold(self._finish(tr))
        out, self.pending, self.tracks = self.pending, [], []
        self.buf = np.zeros(0, np.float32)
        return out

    def _hold(self, e: Event | None) -> None:
        """One drone makes several tracks (its rotors, its octaves, Doppler up then down): events that
        overlap or nearly touch are one event, reported with its strongest track's pitch."""
        if e is None:
            return
        for i, p in enumerate(self.pending):
            if e.start_s <= p.end_s + GAP_S and p.start_s <= e.end_s + GAP_S:
                best = e if e.confidence > p.confidence else p
                merged = Event(min(e.start_s, p.start_s), max(e.end_s, p.end_s), best.f0_hz,
                               max(e.harmonics, p.harmonics), max(e.level_db, p.level_db), best.wobble,
                               max(e.confidence, p.confidence))
                del self.pending[i]
                return self._hold(merged)
        self.pending.append(e)

    def _release(self, t: float) -> list[Event]:
        """Events no running track can still join."""
        first = min((tr.t[0] for tr in self.tracks), default=np.inf)
        ready = [p for p in self.pending if p.end_s + GAP_S < first - FRAME_S / 2 and p.end_s + GAP_S < t]
        self.pending = [p for p in self.pending if p not in ready]
        return ready

    # -- per frame -------------------------------------------------------------------------------
    def _frame(self, x: np.ndarray) -> list[Event]:
        t = (self.n * self.hop + self.frame / 2) / self.sr
        X = np.abs(np.fft.rfft(x * self.win))[self.band]
        power = float(np.sum(X[(self.f >= 80) & (self.f <= 4000)] ** 2))
        level = 10 * np.log10(2 * power / (self.frame * np.sum(self.win ** 2) / self.frame) / self.frame + 1e-20)
        L = 20 * np.log10(X + 1e-12)
        W = L - median_filter(L, size=41, mode="nearest")
        self.bg += self.alpha * (W - self.bg)
        Wd = W - np.maximum(self.bg, 0)
        cands = self._candidates(Wd, L)
        self.n += 1
        return self._track(t, cands, level)

    def _candidates(self, W: np.ndarray, L: np.ndarray) -> list[_Cand]:
        W3 = maximum_filter1d(W, 3)
        H = W3[self.hidx]
        s = np.where(self.kmask, np.clip(H - HIT_DB, -6, 14), 0).sum(axis=1)
        hits = np.where(self.kmask, H >= HIT_DB, False).sum(axis=1)
        order = np.argsort(-s)
        chosen: list[_Cand] = []
        for j in order[:400]:
            if s[j] < MIN_SCORE or len(chosen) == 3:
                break
            if hits[j] < MIN_HITS:
                continue
            f0 = self.f0s[j]
            if any(_related(f0, c.f0) for c in chosen):
                continue
            hit = self.kmask[j] & (H[j] >= HIT_DB)
            f0r = self._refine(f0, L, hit)
            hh = self._finer_comb(f0r, W3)
            chosen.append(_Cand(f0r, float(s[j]), int(hits[j]), float(hh)))
        return chosen

    def _finer_comb(self, f0: float, W3: np.ndarray) -> float:
        """Is this pitch one order of a finer comb (spacing f0/n, down to 20 Hz)? The share of the
        in-between positions k·f0/n (k not a multiple of n, up to 1.5 kHz) that have a peak, best over n.
        A piston engine's firing orders fill them (~1.0); a multirotor leaves them mostly empty (~0.2)."""
        best = 0.0
        for n in range(2, int(f0 // 20) + 1):
            k = np.arange(1, int(1500 * n // f0) + 1)
            k = k[k % n != 0]
            if len(k) < 4:
                continue
            idx = np.rint((k * f0 / n - self.f[0]) / self.df).astype(int)
            idx = idx[(idx >= 0) & (idx < len(W3))]
            best = max(best, float((W3[idx] >= HIT_DB).mean()))
        return best

    def _refine(self, f0: float, L: np.ndarray, hit: np.ndarray) -> float:
        """Pitch to a fraction of a hertz: parabolic peak of each hit harmonic, divided by its number."""
        est = []
        for k in np.flatnonzero(hit) + 1:
            c = int(round((k * f0 - self.f[0]) / self.df))
            lo, hi = max(c - 2, 1), min(c + 3, len(L) - 1)
            if hi - lo < 3:
                continue
            p = lo + int(np.argmax(L[lo:hi]))
            if p <= 0 or p >= len(L) - 1:
                continue
            a, b, cc = L[p - 1], L[p], L[p + 1]
            d = 0.5 * (a - cc) / (a - 2 * b + cc) if (a - 2 * b + cc) != 0 else 0.0
            est.append((self.f[0] + (p + d) * self.df) / k)
        return float(np.median(est)) if est else float(f0)

    # -- tracking ----------------------------------------------------------------------------------
    def _track(self, t: float, cands: list[_Cand], level: float) -> list[Event]:
        used = set()
        for c in cands:
            best, bd = None, TRACK_JUMP
            for tr in self.tracks:
                if id(tr) in used:
                    continue
                d = abs(c.f0 - tr.f0[-1]) / tr.f0[-1]
                if d <= bd:
                    best, bd = tr, d
            if best is None:
                best = _Track()
                self.tracks.append(best)
            used.add(id(best))
            best.t.append(t)
            best.f0.append(c.f0)
            best.hits.append(c.hits)
            best.score.append(c.score)
            best.half.append(c.half)
            best.level.append(level)
            best.last = self.n
        alive = []
        for tr in self.tracks:
            if self.n - tr.last > self.gap:
                self._hold(self._finish(tr))
            else:
                alive.append(tr)
        self.tracks = alive
        return self._release(t)

    def _finish(self, tr: _Track) -> Event | None:
        if not tr.t or tr.t[-1] - tr.t[0] + HOP_S < MIN_EVENT_S:
            return None
        if len(tr.t) < MIN_COVERAGE * (tr.t[-1] - tr.t[0] + HOP_S) / HOP_S:
            return None  # a few scattered frames that happen to line up, not a sound that's there
        f0 = np.array(tr.f0)
        lf = np.log(f0)
        # wobble: pitch scatter around a slow trend (a fly-past's Doppler sweep is a trend, not wobble)
        tt = np.array(tr.t)
        trend = np.polyval(np.polyfit(tt, lf, 2), tt) if len(tt) >= 4 else lf.mean()
        wobble = float(np.std(lf - trend))
        hits = float(np.median(tr.hits))
        K = min(MAX_HARM, int(4000 // np.median(f0)))
        strength = 0.5 * min(1.0, hits / K / 0.6) + 0.5 * min(1.0, float(np.median(tr.score)) / (8.0 * K))
        steady = np.clip((wobble - MIN_WOBBLE) / (FULL_WOBBLE - MIN_WOBBLE), 0, 1)
        half = float(np.median(tr.half))
        engine = np.clip((0.7 - half) / 0.3, 0, 1)  # drones measured ~0.2, engines ~1.0
        dur = tt[-1] - tt[0] + HOP_S
        lasting = 0.7 + 0.3 * min(1.0, (dur - MIN_EVENT_S) / 5.0)
        conf = float(np.clip(0.4 + 0.6 * strength, 0, 1) * steady * engine * lasting)
        if conf < self.threshold:
            return None
        return Event(tt[0] - FRAME_S / 2, tt[-1] + FRAME_S / 2, float(np.median(f0)), int(round(hits)),
                     float(np.max(tr.level)), wobble, conf)


def _related(a: float, b: float) -> bool:
    """Same sound: within 8 % (a multirotor's rotors), or an octave/twelfth apart (a comb's own octaves)."""
    r = max(a, b) / min(a, b)
    return any(abs(r - m) / m < (0.08 if m == 1 else 0.03) for m in (1, 2, 3))


def detect_all(x: np.ndarray, sr: int, threshold: float = CONFIDENCE) -> list[Event]:
    d = Detector(sr, threshold)
    return d.feed(x) + d.flush()
