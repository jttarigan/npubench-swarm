# Study 3, first device data: Galaxy Tab S6 Lite LTE (2026-10-13, 22:34 to 23:46 WIB)

Device: SM-P615, Exynos 9611 (4xA73 + 4xA53), Mali-G72 MP3, Android 13, no NPU
(NNAPI lists only `nnapi-reference`). App: the study 3 APK (md5 cd4d9034..., git
7db173f), model XL, TRACE on, LOOP on, three free runs per config, then REPLAY INPUT
of the first CPU F32 trace on each config; all driven over adb with
`tools/study3_drive.py` (no hands on the tablet; a bare tap is not a gesture).
Data: `runs_study3/s6lite/files/` (trace and replay .bin files are gitignored,
10 MB each), logs `runs_study3/s6lite/logcat_*.txt`, `session_log.txt`.
Mac replays of the same tablet trace (`analysis/policy_backends.py replay`, LiteRT
f32 and int8 on the M5): `runs_study3/s6lite/mac_replays/` (gitignored).

## Free runs: repeat identity

| config | runs | ticks | outcome | repeat identity |
|---|---|---|---|---|
| CPU F32 | 3 | 3724 | victory, 440 kills, hp 6 | byte-identical |
| CPU INT8 | 3 | 3702 | victory, 440 kills, hp 6 | byte-identical |
| GPU F32 (GPU delegate, OpenGL path) | 3 | 3724 | victory, 440 kills, hp 6 | byte-identical |

The tablet is a deterministic peer for itself on every backend, including the GPU
(NeonTide saw one GPU divergence in 24 launches on this tablet with its own shader
workload; three launches here were identical, so the question stays open for the
delegate path too). Free-run pairs against CPU F32: GPU F32 differs from tick 0 in
outputs and state but keeps the same wave starts, score and hp for all 3724 ticks
(enemy positions at most 0.036 px apart after 1200 ticks); INT8 is a different game
(wave starts drift, 3702 vs 3724 ticks, 138 px mean drift after 1200 ticks).

## Replays: outputs over identical inputs (123,354 enemy rows, one XL game)

Reference: the tablet's CPU F32 trace. `analysis/determinism.py matrix`,
`analysis/prevention.py table` (`matrix_xl*.csv`, `prevention_replay_xl*.csv`).

| peer | bit-exact values | mean abs diff | max abs diff | median ULP | fire flips / 1k | regime |
|---|---|---|---|---|---|---|
| tablet CPU F32 (replay of itself) | 100 % | 0 | 0 | 0 | 0 | identity |
| Mac M5 CPU, LiteRT f32 | 37.7 % | 8.0e-8 | 1.3e-6 | 2 | 0 | ULP (cross-chip CPU) |
| tablet GPU F32, Mali-G72 delegate (OpenGL) | 21.0 % | 9.8e-5 | 4.8e-3 | large | 0.016 | fp16 |
| tablet CPU INT8 | 14.8 % | 1.7e-2 | 0.88 | large | 5.6 | model |
| Mac M5 CPU, LiteRT int8 | 14.8 % | 1.7e-2 | 0.88 | large | 5.6 | model |
| tablet CPU INT8 vs Mac CPU INT8 (int8 against int8) | 98.7 % | 6.9e-4 | 0.11 | large | 0.04 | not bit-exact |

Prevention, rows still differing / rows recomputed on the CPU reference:

| peer | grid 2^-2 | grid 2^-3 | grid 2^-4 | grid 2^-3 + guard 1e-4 | grid 2^-3 + guard 1e-2 |
|---|---|---|---|---|---|
| Mac CPU f32 (ULP) | 0 % | 0 % | 0 % | 0 % at 0.27 % recomputed | 0 % at 24 % |
| tablet GPU (fp16) | 0.125 % | 0.25 % | 0.49 % | 0.155 % at 0.26 % | 0 % at 24 % |
| tablet INT8 (model) | 18 % | 34 % | 55 % | 20 % at 24 % | 16 % at 35 % |

The grid costs a steer RMS of 0.062 / 0.033 / 0.017 at 2^-2 / 2^-3 / 2^-4 on the
reference's own outputs.

## Reading, against the three regimes of `../prevention_offline.md`

- **Cross-chip CPU is the ULP regime, measured for real.** Exynos 9611 against Apple
  M5, both through LiteRT's XNNPACK float32 kernels, differ by 2 ULP median and 1.3e-6
  max, exactly the Mac's `np_f32` pair. A 2^-2 to 2^-4 output grid alone left zero
  differing rows in 123k; the 2^-3 grid plus a 1e-4 guard makes that a guarantee at
  0.27 % CPU recomputes. Two phones' CPUs can run lockstep with the cheap method.
- **The Mali-G72 through the TFLite GPU delegate lands in the fp16 regime**, where the
  Mac's `np_f16` emulation put it (Mac: mean 1.4e-4, max 4.9e-3; tablet: 9.8e-5,
  4.8e-3). OpenCL is unavailable on this device (`libvndksupport.so` missing,
  `clEnqueueReleaseEGLObjectsKHR` undefined), so the delegate used its OpenGL compute
  path; all 5 policy nodes were delegated. No grid holds (0.125 % rows at 2^-2 is
  ~150 rows per game, each a desync); the 0.01 guard holds at a quarter of the rows
  recomputed on the CPU, which is the Mac's prediction (24 % at 2^-3) to the decimal.
  The gesture model was rejected by the GPU (EXPAND_DIMS, Mean) and ran on the CPU,
  which has no effect without gestures.
- **INT8 is a different model**, 5.6 fire flips per 1000 rows against F32 on both
  chips, and the Mac int8 numbers match the tablet's to three digits. The new point:
  **int8 against int8 across chips is not bit-exact either** (98.7 % of values, 5
  flips in 123k rows, max 14/128). The int8 kernels on the two CPUs round differently
  somewhere, so "the same int8 model on every peer" is not a lockstep method without a
  guard. On the Poco, CPU INT8 versus NPU INT8 is the remaining int8 question.
  Correction (2026-10-15, POCO session 2): the 98.7 % is tablet against Mac LiteRT, a
  cross-build pair. The tablet's CPU INT8 replay and free game are byte-identical to
  the POCO's CPU INT8 ones, so int8 across the two Android chips is bit-exact
  (`../poco/README.md`).
- The GPU game stayed macroscopically identical to the CPU game for all 3724 ticks:
  a lockstep peer would have desynced at tick 0 while a spectator would see the same
  match. That is the "identity, not playability" point of the Mac writeup on a real
  accelerator.

## Session notes

Wireless adb dropped twice during the pulls (22:46 at 12 % of the 6th trace, ~23:15
at 13 % of the 8th), each time with `error: closed`; the second time the tablet left
`adb devices` while its mDNS record stayed and the port refused, until the user
toggled wireless debugging at ~23:43. Pull after every config next time and keep
transfers short. The tablet showed two mDNS transports for one device, the second
named `... (2)._adb-tls-connect._tcp` (a serial with a space), which
`tools/study3_device.py` now parses. The background logcat capture dies with its
transport. Battery 49 % at the start, 68 % at 23:46 on the charger; screen timeout
600000 ms as found; the app was left in the foreground on its menu.

## Tiny model repeat (2026-10-14, 20:29 to 20:44 WIB)

Same APK and protocol, model T (`tools/study3_drive.py taps cpu t f32 loop`,
then `loop3` per config), unattended over adb, tablet at 100 % on AC. The
replay file is this session's first CPU F32 tiny trace (tag b707785ae20c8a26).

| Config | Runs | Ticks | Repeat traces | Replay of the CPU F32 trace |
|---|---|---|---|---|
| CPU F32 | 3 | 4407 | byte-identical (md5 6060b1e4...) | 100 % bit-exact, max ULP 0 |
| CPU INT8 | 3 | 3956 | byte-identical (md5 6ac1a6ed...) | 0 % bit-exact, 1226 fire flips (different model) |
| GPU F32 | 3 | 4356 | byte-identical (md5 8a38f2d5...) | 0.018 % bit-exact, 10 fire flips |

All nine games end at wave 10 with score 440; HP left differs by config (5, 7, 6).
The tiny model reproduces the XL picture: each backend repeats itself byte for
byte, and CPU INT8 and the Mali-G72 GPU delegate leave the CPU F32 game from tick 0.
