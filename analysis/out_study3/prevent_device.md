# Experiment C: prevention on the devices (`analysis/study3_prevent_device.py`)

Campaign 2026-10-18 17:11-17:43 WIB, spec `grid:2+fallback:0.02` (grid 2^-2, guard 0.02, portable recompute), both tablets on battery. Games: 3 per configuration.

## 1. Whole-game identity

| device | model | backend | prevention | games | distinct traces | ticks | end |
|---|---|---|---|---|---|---|---|
| Tab S6 Lite | tiny | CPU F32 | grid:2+fallback:0.02 | 3 | 1 | 4301 | wave 10 hp 6 |
| Tab S6 Lite | tiny | GPU F32 | grid:2+fallback:0.02 | 3 | 1 | 4301 | wave 10 hp 6 |
| Tab S6 Lite | XL | CPU F32 | grid:2+fallback:0.02 | 3 | 1 | 3697 | wave 10 hp 6 |
| Tab S6 Lite | XL | CPU F32 | off | 3 | 1 | 3724 | wave 10 hp 6 |
| Tab S6 Lite | XL | GPU F32 | grid:2+fallback:0.02 | 3 | 1 | 3697 | wave 10 hp 6 |
| Tab S7+ | tiny | CPU F32 | grid:2+fallback:0.02 | 3 | 1 | 4301 | wave 10 hp 6 |
| Tab S7+ | tiny | GPU F32 | grid:2+fallback:0.02 | 3 | 1 | 4301 | wave 10 hp 6 |
| Tab S7+ | tiny | NPU F32 | grid:2+fallback:0.02 | 3 | 1 | 4301 | wave 10 hp 6 |
| Tab S7+ | XL | CPU F32 | grid:2+fallback:0.02 | 3 | 1 | 3697 | wave 10 hp 6 |
| Tab S7+ | XL | CPU F32 | off | 4 | 1 | 3724 | wave 10 hp 6 |
| Tab S7+ | XL | GPU F32 | grid:2+fallback:0.02 | 3 | 1 | 3697 | wave 10 hp 6 |
| Tab S7+ | XL | NPU F32 | grid:2+fallback:0.02 | 3 | 1 | 3697 | wave 10 hp 6 |

- XL, prevented: 15 games over 5 device/backend pairs (Tab S6 Lite CPU, Tab S6 Lite GPU, Tab S7+ CPU, Tab S7+ GPU, Tab S7+ NPU): 4 distinct trace file(s) 2e81ac80, 3dbe39e8, 53a6dbd1, c597773d. Inputs and state identical on every tick: True; outputs equal in value on every row: True; output values differing only in the sign of a zero (-0 vs +0), per game against Tab S6 Lite CPU: 0 to 103.
- tiny, prevented: 15 games over 5 device/backend pairs (Tab S6 Lite CPU, Tab S6 Lite GPU, Tab S7+ CPU, Tab S7+ GPU, Tab S7+ NPU): 4 distinct trace file(s) 021f00c3, 0383802e, 07f214a5, d36bc7f7. Inputs and state identical on every tick: True; outputs equal in value on every row: True; output values differing only in the sign of a zero (-0 vs +0), per game against Tab S6 Lite CPU: 0 to 50.
- Control (Tab S6 Lite, XL CPU F32, prevention off): byte-identical to the study game 20261013_223444 (3 games).
- Controls on both tablets: 1 distinct trace file(s).

## 2. Mac peers along the device trajectory

Each recorded input row is run through the Mac backend, prevented with the same rule (portable kernel for guarded rows), and compared with the output the tablet's game received (Tab S6 Lite CPU game).

| model | Mac peer | rows | rows differing in value | rows differing only in the sign of a zero | recomputed | largest raw difference vs portable |
|---|---|---|---|---|---|---|
| XL | LiteRT CPU F32 (XNNPACK) | 121,392 | 0 | 0 | 22.3 % | 1e-06 |
| XL | emulated fp16 storage | 121,392 | 0 | 18 | 22.2 % | 0.0053 |
| tiny | LiteRT CPU F32 (XNNPACK) | 132,413 | 0 | 0 | 39.1 % | 4.6e-07 |
| tiny | emulated fp16 storage | 132,413 | 0 | 28 | 38.9 % | 0.0021 |

Device outputs all on the 2^-2 grid: XL: True; tiny: True.

## 3. Bit check of the portable kernel on the S6 Lite

- run_20261018_171200_replay_cpu_xlarge_f32_replay.bin: 123,354 of 123,354 rows bit-identical to pp_cli on the Mac (M5, clang -O2 -ffp-contract=off); largest difference 0.

## 4. Cost (ticks after warmup)

| device | model | backend | prevention | rows recomputed | prevent ms/tick median | p99 | max | ms per recomputed row | inference ms/tick median | tick ms median | missed deadlines |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Tab S6 Lite | tiny | CPU | grid:2+fallback:0.02 | 39.4 % | 0.030 | 0.124 | 0.98 | 0.0030 | 0.08 | 0.19 | 0 of 12603 |
| Tab S6 Lite | tiny | GPU | grid:2+fallback:0.02 | 39.4 % | 0.017 | 0.054 | 0.15 | 0.0016 | 13.38 | 13.45 | 5 of 12603 |
| Tab S6 Lite | XL | CPU | grid:2+fallback:0.02 | 22.5 % | 4.798 | 18.307 | 35.90 | 0.7734 | 11.50 | 16.53 | 5333 of 10791 |
| Tab S6 Lite | XL | CPU | off | 0.0 % | 0.000 | 0.000 | 0.00 | nan | 11.01 | 11.09 | 1689 of 10872 |
| Tab S6 Lite | XL | GPU | grid:2+fallback:0.02 | 22.5 % | 5.614 | 17.916 | 44.13 | 0.8387 | 10.07 | 14.69 | 4218 of 10791 |
| Tab S7+ | tiny | CPU | grid:2+fallback:0.02 | 39.4 % | 0.017 | 0.068 | 0.24 | 0.0017 | 0.05 | 0.10 | 0 of 12603 |
| Tab S7+ | tiny | GPU | grid:2+fallback:0.02 | 39.5 % | 0.025 | 0.102 | 0.23 | 0.0024 | 10.76 | 10.84 | 1 of 12603 |
| Tab S7+ | tiny | NPU | grid:2+fallback:0.02 | 39.2 % | 0.026 | 0.111 | 0.39 | 0.0026 | 3.54 | 3.62 | 0 of 12603 |
| Tab S7+ | XL | CPU | grid:2+fallback:0.02 | 22.5 % | 1.694 | 4.934 | 11.54 | 0.2425 | 3.49 | 5.38 | 1 of 10791 |
| Tab S7+ | XL | CPU | off | 0.0 % | 0.000 | 0.000 | 0.00 | nan | 3.93 | 3.97 | 0 of 14496 |
| Tab S7+ | XL | GPU | grid:2+fallback:0.02 | 22.5 % | 2.307 | 8.213 | 14.40 | 0.3576 | 6.21 | 9.43 | 2782 of 10791 |
| Tab S7+ | XL | NPU | grid:2+fallback:0.02 | 22.4 % | 2.584 | 8.729 | 14.96 | 0.3787 | 3.75 | 6.30 | 3 of 10791 |

Mean alive rows per tick: 31.7. prevent_ms times the whole prevention step (edge tests and grid rounding on every row, plus the portable recompute of the guarded rows).

## 5. Validation

Re-simulated on the Mac (tools/hostsim) from the recorded outputs: 9 of 9 distinct trace files reproduced byte for byte.
