# Study 3, POCO X6 Pro (2026-10-13 23:52 to 2026-10-14 00:09 WIB), partial session

Device: POCO X6 Pro, Dimensity 8300 (MT6897), Mali-G615 MC6, MediaTek APU
(NNAPI lists mtk-dsp_shim, mtk-mdla_shim, mtk-neuron_shim, nnapi-reference; the app
picks mtk-mdla_shim), Android 16 (SDK 36), peak refresh 60 Hz. App: the study 3 APK
(md5 cd4d9034...), fresh install after a keystore mismatch (the July build's 136 files
are in `runs_study3/poco/preinstall_backup_20261013T165150Z/`). Screen 1220x2712,
internal 240x534 at scale 5. Attended: the user tapped the menu (the POCO refuses
injected taps). The user started CPU F32 XL with TRACE and LOOP on and let it loop:
12 runs, 23:55:36 to 00:08:45, then Main ended the session (bedtime). No other config
and no replay ran; `replay_input.bin` (the Tab S6 Lite CPU F32 trace, tag
1c255b7430926177) is already on the phone.

Data: `runs_study3/poco/files/` (12 traces of 10.3 MB, gitignored; ticks and
coldstart CSVs committed), `logcat_20261013T165159Z.txt`, `session_log.txt`.

## Results

| comparison | rows | bit-exact | max abs diff | median ULP | fire flips |
|---|---|---|---|---|---|
| 12 POCO CPU F32 runs against each other | 12 x 3724 ticks | byte-identical, 1 distinct of 12 | 0 | 0 | 0 |
| POCO CPU F32 run 1 vs Tab S6 Lite CPU F32 run 1 (free runs, closed loop) | 123,354 | **100 % for the whole game**, state identical on all 3724 ticks | 0 | 0 | 0 |
| POCO CPU F32 trace replayed on the Mac M5 (LiteRT f32) | 123,354 | 37.7 % of values | 1.3e-6 | 2 | 0 |

The headline: **two different Android SoCs, Dimensity 8300 on Android 16 and Exynos
9611 on Android 13, play the whole XL game bit-identically on their CPUs** with the same
APK and no prevention method at all: same outputs, same enemy positions, same wave
starts (283, 646, ... 3329), same 440 kills and hp 6 at tick 3723. The LiteRT/XNNPACK
float32 kernels in the APK give the same bits on both ARM64 CPUs. The Mac's LiteRT
(TensorFlow 2.20 Python wheel, Apple silicon) differs from both phones by the same 2
ULP median and 1.3e-6 max, so the cross-chip ULP regime measured earlier is a
cross-build difference between kernels, not a property of the phone chips.

For a lockstep game shipped as one APK, the CPU float32 path is therefore a usable
common peer across chips on this evidence (two chips, one game each); the ULP guard of
`../prevention_offline.md` is the insurance for a third chip or a different build.

## Session 2 (2026-10-15, 13:39 to 15:29 WIB): the other four configs and the replays

Attended, same APK, model XL, TRACE on. The user picked each config and tapped START
(LOOP off, three taps) for CPU INT8, GPU F32 and NPU F32; NPU INT8 ran with LOOP on and
the Mac ended the loop with `am force-stop` right after the 4th run started (no partial
files are left). One more CPU F32 free run came from a START tap meant as a replay.
Then REPLAY INPUT of the Tab S6 Lite CPU F32 trace (tag 1c255b7430926177) on all five
configs, CPU INT8 twice. Delegation: GPU through OpenGL (OpenCL missing, as on the
tablet), 5 of 5 policy nodes; NPU through NNAPI `mtk-mdla_shim`, 5 of 5 policy nodes for
F32 and INT8, the gesture model too. Phone 43 % at the start, 99 % at the end on the
charger, battery 27 to 33 C.

### Free runs

| config | runs | ticks | outcome | repeat identity | trace md5 |
|---|---|---|---|---|---|
| CPU F32 | 12 + 1 | 3724 | victory, 440 kills, hp 6 | byte-identical | 2f5efb0d... (= Tab S6 Lite) |
| CPU INT8 | 3 | 3702 | victory, 440 kills, hp 6 | byte-identical | 49112f98... (= Tab S6 Lite) |
| GPU F32 (Mali-G615) | 3 | 3724 | victory, 440 kills, hp 6 | byte-identical | 561c3e4e... |
| NPU F32 (MediaTek APU) | 3 | 3724 | victory, 440 kills, hp 6 | byte-identical | 3ebbec5f... |
| NPU INT8 (MediaTek APU) | 3 | 3702 | victory, 440 kills, hp 6 | byte-identical | 49112f98... (= CPU INT8) |

Every backend, the NPU included, repeats itself byte for byte. Three whole-game
identities hold across hardware: CPU F32 POCO = CPU F32 tablet, CPU INT8 POCO = CPU
INT8 tablet, and NPU INT8 = CPU INT8 on the POCO. The GPU games of the two Mali
generations (G615 here, G72 on the tablet) split at tick 340 by up to 1.7e-4 in one
output, drift by at most 3e-5 px, and reconverge at the next wave start: the state is
identical again from tick 3329 to the end. NPU F32 leaves the CPU F32 game at tick 0 and
stays split (drift up to 0.016 px), with the same wave starts, score and hp.

### Replays: outputs over identical inputs (123,354 enemy rows)

Reference: the tablet's CPU F32 trace, which is also the POCO's own CPU F32 game
(`matrix_xl.csv`, `prevention_replay_xl.csv`).

| peer | bit-exact values | mean abs diff | max abs diff | median ULP | fire flips | regime |
|---|---|---|---|---|---|---|
| POCO CPU F32 | 100 % | 0 | 0 | 0 | 0 | identity |
| POCO GPU F32 (Mali-G615, OpenGL delegate) | 20.96 % | 9.8e-5 | 4.8e-3 | 745 | 2 | fp16 |
| POCO NPU F32 (APU via NNAPI) | 20.87 % | 1.3e-4 | 4.4e-3 | 2565 | 1 | fp16 |
| POCO CPU INT8, POCO NPU INT8 | 14.78 % | 1.7e-2 | 0.88 | large | 692 | model |
| Mac M5 LiteRT f32 (from the tablet session) | 37.7 % | 8.0e-8 | 1.3e-6 | 2 | 0 | ULP (cross-build) |

Peer against peer, same inputs:

| pair | values that differ | rows that differ | max abs diff | grid 2^-2 / 2^-3 / 2^-4, rows still differing |
|---|---|---|---|---|
| CPU INT8 POCO vs CPU INT8 tablet | 0 | 0 | 0 | 0 / 0 / 0 |
| NPU INT8 vs CPU INT8, POCO | 1 (tick 1089, 120/128 vs 121/128) | 1 | 1/128 | 0 / 0 / 0 |
| GPU F32 POCO vs GPU F32 tablet | 19 | 13 | 1.1e-4 | 0 / 0 / 0 |
| NPU F32 vs GPU F32, POCO | 81 % | 100 % | 2.4e-3 | 184 / 431 / 976 |
| CPU INT8 POCO vs Mac LiteRT int8 | 1.26 % | 3.7 % | 0.11 | (cross-build) |

Prevention against the CPU F32 reference (rows still differing / rows recomputed on
the CPU):

| peer | grid 2^-2 | grid 2^-3 | grid 2^-4 | grid 2^-3 + guard 1e-4 | grid 2^-3 + guard 1e-2 |
|---|---|---|---|---|---|
| POCO GPU (fp16) | 0.125 % | 0.25 % | 0.49 % | 0.155 % at 0.26 % | 0 % at 24 % |
| POCO NPU F32 (fp16) | 0.146 % | 0.34 % | 0.81 % | 0.076 % at 0.57 % | 0 % at 24 % |
| POCO INT8, CPU or NPU (model) | 18 % | 34 % | 55 % | 20 % at 24 % | 16 % at 35 % |

## Reading

- **The APU's F32 path is the fp16 regime.** The MediaTek NNAPI driver runs the float
  model at reduced precision: 2565 ULP median, max 4.4e-3, the same size as the Mali
  delegates. Against the CPU it needs the 0.01 guard (24 % of rows recomputed), the
  Mac's prediction for fp16 peers; no grid alone holds.
- **INT8 is reproducible across Android chips and nearly across processors.** The
  int8 game is byte-identical on the Dimensity 8300 CPU, the Exynos 9611 CPU and the
  POCO's APU (whole free runs, same md5). Over the tablet's inputs the APU differs from
  the CPU in one value out of 370,062: one quantization step (1/128) in the second
  output at tick 1089, a requantization rounding difference on an input the free games
  never produced. The Mac's LiteRT int8 differs from both phones on 1.3 % of values, so
  the earlier "int8 is not bit-exact across chips" (tablet vs Mac) is a cross-build
  effect, like the 2-ULP F32 gap. An all-int8 lockstep game on these two phones would
  have desynced once (APU peer) or never (CPU peers) in this game; a 2^-2 to 2^-5 grid
  removed that one row.
- **Same backend on two GPU generations is close to the ULP regime.** Mali-G615 and
  Mali-G72 through the same OpenGL delegate differ in 19 values (13 rows) of 370,062,
  by at most 1.1e-4, and every grid tested (2^-2, 2^-3, 2^-4, 2^-5, 2^-7) removed all of them. NeonTide's
  observation that the G72 driver fuses `fma()` where the G615 does not fits this
  picture of rare last-bit splits. A cross-device GPU-only lockstep with a grid looks
  feasible on this evidence (two chips, one game), while GPU against CPU stays fp16.
- **Different accelerators on one phone do not agree with each other.** NPU F32 and
  GPU F32 on the POCO differ on every row; both are fp16-class but round differently.

## Session notes

adb's own mDNS listed nothing on the office Wi-Fi although `dns-sd` saw the phone;
`adb connect` to the address from `dns-sd -L` / `-G v4` worked, after which the mDNS
transport appeared too. The background logcat capture died silently once (restart with
`tools/study3_device.py poco logcat stop/start`). After `am force-stop` the app menu
resets the model to M. Replays take 2 to 3 s each on this phone. The phone also holds a
12:20 CPU MEDIUM game and a sweep the user had left running; neither is study data.
