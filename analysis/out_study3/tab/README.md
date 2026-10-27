# Study 3, Galaxy Tab S7+ (2026-10-16, 08:27 to 08:47 WIB)

Device: SM-T975, Snapdragon 865+ (SM8250), Adreno 650, Hexagon 698, Android 13
(T975XXS7DXH2). NNAPI lists qti-default, qti-dsp, qti-hta (type 4), qti-gpu (type 3)
and nnapi-reference; the app's picker (written for MediaTek names) took the first
type-4 device, `qti-default`. Screen 1752x2800, internal 240x534 at scale 5.
App: the study 3 APK (md5 cd4d9034...), fresh install (the app was not on the
tablet). Unattended over wireless adb with `tools/study3_drive.py --model SM_T975
--screen 1752 2800` (new per-model menu reference `tools/study3_menu_SM_T975.png`);
the user only plugged the charger in at 08:28. Battery 79 % to 90 %, 29 to 30 C.
Kemulusan Gerakan Tinggi (refresh_rate_mode 2) and Sleep Never, both as found and
left. Logs: `runs_study3/tab/seq_free_20261016.txt`, `seq_replay_20261016.txt`,
`logcat_20261016T012709Z.txt`, `session_log.txt`.

Delegation: GPU through OpenGL (OpenCL library lacks
`clEnqueueReleaseEGLObjectsKHR`, as on both Mali devices); NNAPI took 5 of 5 policy
nodes for F32 and INT8. The gesture model fell back to the CPU on GPU and NPU (no
effect without gestures).

## Free runs (XL, model trace classes as in `../crosschip.md`)

| config | runs | ticks | outcome | repeat identity | class |
|---|---|---|---|---|---|
| CPU F32 | 3 | 3724 | victory, 440, hp 6 | byte-identical | A (= Tab S6 Lite = POCO) |
| CPU INT8 | 3 | 3702 | victory, 440, hp 6 | byte-identical | B (= Tab S6 Lite = POCO = POCO NPU INT8) |
| GPU F32 (Adreno 650) | 4 | 3724 | victory, 440, hp 6 | byte-identical | F (own) |
| NPU F32 (qti-default) | 3 | 3725 | victory, 440, **hp 7** | byte-identical | G (own) |
| NPU INT8 (qti-default) | 3 | 3709 | victory, 440, hp 6 | byte-identical | H (own) |

GPU F32 ran 4 games (the stop taps landed after the 4th start); all 4 are complete
and identical. Median policy inference per tick (ms): CPU F32 4.2, CPU INT8 5.4,
GPU F32 12.6, NPU F32 4.1, NPU INT8 2.7.

## Replays of the Tab S6 Lite CPU F32 trace (tag 1c255b7430926177)

| peer (against CPU F32) | bit-exact values | max abs diff | fire flips |
|---|---|---|---|
| CPU F32 | 100 % | 0 | 0 |
| GPU F32, Adreno 650 | 18.1 % | 0.011 | 4 |
| NPU F32, qti-default | 19.6 % | 0.016 | 9 |
| CPU INT8 | 14.8 % | 0.88 | 692 (same values as both other phones) |
| NPU INT8, qti-default | 14.8 % | 0.90 | 676 |

NPU INT8 against CPU INT8 (both int8, same inputs): 59.1 % of values bit-exact,
151,216 values differ, mean 0.0078 (one 1/128 step), max 0.070 (9 steps), 16 fire
flips. No grid removes it (10,805 values still differ at 2^-2).

Prevention against CPU F32 (rows still differing / rows recomputed):

| peer | grid 2^-2 | grid 2^-3 | grid 2^-4 | grid 2^-3 + guard 1e-4 | grid 2^-3 + guard 0.01 |
|---|---|---|---|---|---|
| Adreno GPU F32 | 0.29 % | 0.58 % | 1.17 % | 0.46 % at 0.25 % | 0.001 % at 24 % |
| qti-default NPU F32 | 0.67 % | 1.34 % | 2.70 % | 1.13 % at 0.48 % | 0.002 % at 24 % |

## Reading

- **Three SoC vendors, one CPU game.** Snapdragon 865+ joins Exynos 9611 and
  Dimensity 8300: the CPU F32 game and the CPU INT8 game are byte-identical on all
  three, and the CPU F32 replay is bit-exact. Same APK, same LiteRT XNNPACK kernels.
- **Qualcomm's float paths are noisier than the Mali and MediaTek ones.** Max
  difference 0.011 (Adreno) and 0.016 (qti-default NPU) against about 0.005 on the
  Mali GPUs and the MediaTek APU, closer to the Mac's fp16-accumulate emulation
  (6.7e-3) or above it. The 0.01 guard that held exactly for the other fp16 peers
  leaves 1 to 3 rows of 123k here; the guard has to be about 0.02. Grids alone
  leave two to five times more rows than on the Mali GPUs.
- **The qti-default NPU F32 game ends differently**: hp 7 against 6, one tick
  longer, waves 8 and 10 start one and two ticks off. This is the first device
  pair in the study whose outcome differs; all others kept the same length and hp.
- **int8 on the Qualcomm NNAPI path is not the CPU's int8.** Unlike the MediaTek
  APU (1 value in 370k), qti-default differs from the CPU int8 kernels on 41 % of
  values, mostly by one quantization step, so the requantization rounding differs
  systematically. The free int8 games split at tick 0 and drift 130 px by tick
  1200 (different wave starts from wave 4), with the same final score and hp.
  "Use the int8 model everywhere" is a lockstep method on this evidence only if all
  peers run it on the CPU, or on accelerators that match it.
- Open: `qti-default` is Qualcomm's routing device; whether it ran on the Hexagon
  DSP, the HTA or the GPU is not visible from the app. A run pinned to `qti-dsp`
  would need an accelerator override in the app (a rebuild).

## Session 2: tiny model (2026-10-16, 15:10 to 15:31 WIB)

Same APK and install as session 1 (the app stayed installed), unattended over
wireless adb, the user away. Menu set to T + LOOP by taps (the app opened on M),
then session 1's free-run and replay sequences in one script; log `runs_study3/tab/seq_tiny_20261016.txt`, logcat
`logcat_20261016T081009Z.txt`. On AC, battery 100 %, 31 to 32 C. Display as found
and left: Tinggi (refresh_rate_mode 2), screen_off_timeout 300000, stay_on 0 (the
app holds the screen on). Replay input: the Tab S6 Lite tiny CPU F32 game
(tag b707785ae20c8a26), so the replays line up with the S6 Lite's and the Mac's.

| config | games | trace class (`../crosschip.md`) | ticks, hp |
|---|---|---|---|
| CPU F32 | 3 | A (= Tab S6 Lite) | 4407, 5 |
| CPU INT8 | 3 | B (= Tab S6 Lite) | 3956, 7 |
| GPU F32 (Adreno 650) | 3 | D | 4356, 6 |
| NPU F32 (qti-default) | 3 | E | 4356, 6 |
| NPU INT8 (qti-default) | 3 | F | 3933, 5 |

Replays (classes T1 to T8 in `../crosschip.md`):
- CPU F32 replay is bit-exact with the S6 Lite game (class T1); CPU INT8 equals the
  S6 Lite CPU INT8 (T2). The Mac's LiteRT build differs at ULP level (5.3e-7).
- Adreno F32: max 3.2e-3, 19 fire flips; qti-default F32: max 4.0e-3, 14 flips.
  Smaller than for XL (0.011 / 0.016), more flips.
- qti-default INT8 against CPU INT8: 36 % of values, up to 3 quantization steps
  (0.023), 160 fire flips (XL: 41 %, 16 flips).

Reading: every backend repeats; CPU F32 and INT8 games are chip-independent for a
second model. The fp16-class paths (Adreno, qti-default, and the Mali-G72 and Metal
GPUs) all end the tiny game at 4356 ticks with hp 6 instead of 4407 / hp 5, and
qti-default int8 ends with hp 5 instead of 7: with the tiny model, accelerator
differences change the outcome.
