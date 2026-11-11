# earshot

Leave a laptop or a Raspberry Pi listening, and earshot tells you when a small drone (a quadcopter or
other multirotor) is probably nearby. Several people can pool what their listeners heard to get a rough
idea of where it was.

- **It keeps no audio.** Sound is analysed in memory, one second at a time, and thrown away. The only
  thing earshot ever writes is a one-line summary per event: when, how confident, the rotor pitch, how
  many harmonics, how loud. No recordings, no spectrograms, nothing a voice could be recovered from.
- **It never uses the network.** There is no network code in it at all. Pooling is done by people
  sharing their event logs however they like.
- **No AI models.** It's classical signal processing you can read: about 300 lines.

Recording laws vary. In some places (Oregon, for example) recording a conversation needs everyone's
consent. earshot doesn't record, and is built that way so a listener can't become an eavesdropper,
but check the rules where you put one, and tell the people around it what it is.

## Install

Mac or Linux, with [uv](https://docs.astral.sh/uv/):

```sh
uv tool install git+https://github.com/lizTheDeveloper/earshot   # (not published yet: from a local clone, `uv tool install .`)
```

On Linux you may also need PortAudio: `sudo apt install libportaudio2`.

## Use

```sh
earshot listen --node roof --log roof.jsonl          # listen to the microphone; Ctrl-C to stop
earshot listen --file walk.wav --start 2026-11-10T20:00:00Z   # analyse a recording you already have
earshot devices                                      # list microphones; pick one with --device N
earshot merge roof.jsonl porch.jsonl park.jsonl --positions positions.json
```

A detection prints a line like:

```
21:43:02  possible drone  182 Hz, 15 harmonics, confidence 0.84, level -31 dBFS
```

and, with `--log`, appends a JSON line:

```json
{"start": "2026-11-10T05:43:02.250+00:00", "end": "2026-11-10T05:43:14.000+00:00", "node": "roof",
 "confidence": 0.84, "f0_hz": 182.3, "harmonics": 15, "level_db": -31.2, "wobble": 0.012}
```

`level_db` is relative to the microphone's full scale, not calibrated.

### Pooling (as a group)

Each person runs `earshot listen --node <their name> --log <file>` and shares the log file. One person
runs `earshot merge` on all of them. Events from different listeners that start within 5 s of each other
at a similar pitch are treated as the same drone. For each one, merge reports which listener heard it
loudest (the simplest "it was nearest here"), and with **three or more listeners whose positions are
given**, a rough area:

```json
{"roof": [45.5231, -122.6765], "porch": [45.5262, -122.6712], "park": [45.5208, -122.6698]}
```

The area comes from the differences in when each listener first heard it: sound travels about 343 m a
second, so a listener 343 m farther away hears the drone about a second later. It's rough by design:
earshot only knows when it first heard something to within a quarter of a second (about 86 m of sound
travel), and every listener's clock has to be right. Keep clocks synced (macOS and most Linux do this
automatically over NTP); a clock that's a second off moves the estimate by hundreds of metres. Positions
only need to be good to tens of metres. Whether you share listeners' positions is up to the group: a
position in a shared file says where someone lives.

## How it decides

Multirotor drones make a distinctive sound: each propeller blade passing makes a pulse, so each rotor
produces a tone (the blade-pass frequency, usually 80–600 Hz) plus a long ladder of harmonics, and the
rotors run at slightly different speeds that the flight controller keeps adjusting. earshot looks for
that, every quarter second:

1. finds harmonic ladders with a fundamental from 80 to 600 Hz and at least 4 harmonics clearly above
   the local noise floor;
2. follows each one over time, allowing for Doppler and throttle changes;
3. calls it a possible drone if it lasts at least 2.5 s, and its pitch wobbles the way a flight
   controller makes it wobble;
4. turns it down if the pitch is perfectly steady (fans, pumps, transformers, mains hum), or if the
   ladder is really part of a finer one (a piston engine's firing orders: mowers, leaf blowers,
   motorbikes, generators).

Sounds that never stop slowly fade into a background estimate, so a fridge or an air conditioner
doesn't keep tripping it. The other side: a drone hovering in the same place for several minutes
fades too.

## How well it works

Measured on **synthetic** sounds only (`uv run python scripts/evaluate.py`): simulated multirotors
with 2–6 rotors, 80–600 Hz, random wobble, half of them flying past, in pinkish background noise.

| drone loudness vs background (80 Hz–4 kHz) | detected |
|---|---|
| +5 dB | 30 / 30 |
| 0 dB | 30 / 30 |
| −5 dB | 26 / 30 |
| −10 dB | 10 / 30 |
| −15 dB | 0 / 30 |

False alarms on 2 hours of simulated non-drone sound (noise, talking, 50 and 60 Hz mains hum, fans at
80–400 Hz, piston engines firing at 25–160 Hz): **none**.

It has **not been tested on real recordings yet**. The one public drone-sound dataset we found forbids
security and surveillance uses, so we didn't use it. Real numbers need real drones; treat every
detection as "go and look", not as a fact.

## What it can't tell apart (yet)

- **A piston engine whose lowest firing order is above ~80 Hz** (some small two-strokes, RC planes,
  motorbikes revving) can look like a drone.
- **Electric motors that change speed**: e-bikes and scooters accelerating, power tools, some fans on
  variable controllers, an electric RC car.
- **Music** with a sustained note, especially synthesisers; a **wobbly whistle**.
- **Fixed-wing and engine-powered drones** don't make the multirotor sound and are mostly missed, apart
  from very loud ones.
- **Far-away drones**: a small quadcopter is usually audible over 100–300 m outdoors in a quiet place,
  much less near traffic.
- **A drone hovering for minutes** fades into the background estimate.

## Files

- `src/earshot/detect.py`: the detector (streaming, keeps one second of audio at most).
- `src/earshot/merge.py`: pooling and the rough area.
- `src/earshot/cli.py`: `listen`, `merge`, `devices`.
- `scripts/evaluate.py`: the synthetic numbers above.
- `docs/design.md`, `docs/networking.md`: how it's built, and how live pooling could work.

MIT licensed.
