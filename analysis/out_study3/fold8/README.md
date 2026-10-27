# Galaxy Z Fold 8: analysis (2026-10-21)

The Fold 8 session (2026-10-20, `runs_study3/fold8/README.md`) is part of the default
analysis since the user's OK to decision 7 (2026-10-21): `crosschip.md` and its CSVs,
`guard.md`, `guard_oos.csv` and `validation.txt` one folder up include it (as the last
device, so the earlier class ids stay: Adreno 840 = R20 and T11). Re-simulation:
`tools/hostsim/validate_trace.py` per game, appended to `validation.txt`.

## Results

- **Whole games** (`crosschip.md`): CPU F32 and CPU INT8 games of both models are
  byte-identical to the four other Android devices (classes A and B). GPU F32 (Adreno
  840) is its own class for both models (R), repeated 3/3. Outcomes: XL GPU F32 3,724
  ticks, hp 6 (the CPU outcome); tiny GPU F32 4,407 ticks, hp 5 (the CPU outcome; the
  Adreno 650 ends it 4,356, hp 6).
- **58 tiny CPU F32 games** (the app looped through the adb outage): 57 class A, one
  (20261020_194213) class Q. That game's only recorded gesture (a "slash", accepted)
  falls at tick 4,038, where its state leaves the reference game; its outputs leave
  at tick 4,062, and it still ends 4,407 ticks, hp 5. A game input, not the model.
- **Re-simulation** (`validation.txt`): 72 of 73 games re-simulate byte for byte on
  the Mac. The disturbed game diverges structurally at tick 4,062 because the host
  simulation does not replay gestures.
- **Replays over identical inputs**: CPU F32 joins R1/T1 and CPU INT8 joins R2/T2
  (bit-exact). GPU F32 is a new class, R20 (XL) and T11 (tiny):

  | pair | values differing | rows differing | max abs diff | fire flips |
  |---|---|---|---|---|
  | R1 vs R20 (CPU vs Adreno 840) | 303,420 | all 123,354 | 0.0105 | 4 |
  | R9 vs R20 (Adreno 650 vs 840) | 298,271 | all | 0.0060 | 0 |
  | R3 vs R20 (Mali-G72 vs Adreno 840) | 304,881 | all | 0.0124 | 6 |
  | T1 vs T11 (tiny, CPU vs Adreno 840) | 389,207 | all 129,744 | 0.0032 | 18 |
  | T7 vs T11 (tiny, Adreno 650 vs 840) | 25,393 | 15,268 | 0.00098 | 1 |

- **Arithmetic** (Table 5 method, 20,000 sampled rows, seed 0): Adreno 840 error
  against float64 mean 2.3e-4, p99 1.3e-3, max 8.6e-3, 16.9 % of values on the fp16
  grid (Adreno 650: 2.5e-4, 1.6e-3, 1.0e-2, 0.169). Between the fp16-sums-every-16
  (2.0e-4) and every-4 (3.5e-4) emulations, the same reading as the Adreno 650:
  partial fp16 accumulation. Regime label "fp16, wider".
- **Guard** (`guard.md`, Table 6 row): max diff 0.011; grid 2^-2 0.27 %, 2^-3 0.546 %,
  2^-4 1.11 %; 2^-2 with guard 1e-4 0.216 % at 0.11 %; guard 0.01: 0 at 11 %; guard
  0.02: 0 at 22 %. Like the other Qualcomm peers, 0.01 left no differing row in this
  game but only 0.02 exceeds the largest difference.
- **Drift** (Figure 3 measure, XL GPU vs CPU): 0.005 to 0.025 px per second, inside
  the 0.001 to 0.03 px the text gives for the other Android GPUs.
- **No NPU**: no vendor NNAPI driver (the app sees `nnapi-reference` only), so no NPU
  configs; the Hexagon is reachable only through Qualcomm's QNN/SNPE libraries.

