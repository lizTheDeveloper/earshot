# earshot v1 — design

## Why
A cheap way for people to notice small drones overhead — the citizen side of the mic networks
that cities and militaries already run — without building a neighbourhood eavesdropping network.

## Goals
- `earshot listen`: leave a laptop/Raspberry Pi listening to its mic; print and log an event
  when a multirotor drone's rotor sound is likely present.
- `earshot merge`: pool several people's event logs. When ≥3 nodes heard the same event, give a
  rough area from onset-time differences, plus the simpler "which node heard it loudest".
- Classical signal processing only. No models, no network code, no audio kept.

## Non-goals (v1)
- Live networking between nodes (see `docs/networking.md`).
- Identifying the drone model, or anything about people.
- Fixed-wing drones and engine-powered drones (different sound; maybe later).

## Anti-goals
- Ever writing audio, or anything from which speech could be recovered (spectrograms,
  per-frame spectra), to disk or the network.
- Claiming certainty: lawnmowers, leaf blowers, fans and music make harmonic combs too.

## How
- Mono 16 kHz, in memory. 1.0 s Hann frames, 0.25 s hop (1 Hz resolution).
- Spectrum whitening twice: subtract a running median across frequency (peaks stand out from the
  local noise floor) and a slow per-bin background (~2 min time constant), so anything that runs
  for a long time (fridge, HVAC, a neighbour's pump) fades into the background.
- Harmonic sum over f0 = 80–600 Hz: mean whitened level at k·f0, k = 1..K (k·f0 ≤ 4 kHz,
  K ≤ 20); count harmonics ≥ 6 dB over the local floor. Best f0 per frame.
- Track f0 across frames (≤ 4 % jump per hop allows Doppler and throttle changes); an event needs
  ≥ 2.5 s of continuous track with ≥ 4 harmonics; ends after 1.5 s without.
- Discriminators (each lowers confidence or rejects):
  - mains hum: f0 locked within 0.5 Hz of a 50/60 Hz multiple with almost no wobble → reject;
  - piston engines: firing produces half-order components (energy at odd multiples of f0/2);
    multirotors don't → penalise strong half-orders;
  - wobble: multirotor pitch is never perfectly steady (attitude control), fans/motors on mains
    nearly are → penalise very low wobble;
  - speech: voiced pitch jumps by >4 % between hops and stops within a second, so the
    continuity + persistence rule mostly rejects it.
- Event record (JSON line): `start`, `end` (UTC ISO), `node`, `confidence` (0–1), `f0_hz`,
  `harmonics`, `level_db` (dBFS, uncalibrated), `wobble`. Nothing else.
- Merge: cluster events from different nodes whose starts are within 5 s and f0 within 15 %
  (Doppler differs per node). Area: grid search for the point whose distances best explain the
  onset-time differences at 343 m/s, with the unknown emission time solved per point; reported
  with its residual. Onsets are only known to a hop (0.25 s ≈ 86 m of sound travel) plus clock
  error, so the area is rough by design. Needs NTP-synced clocks.

## Left to the builder
Thresholds, the confidence formula, file layout.
