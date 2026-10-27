# NPU Bench Swarm

A 2D neon swarm shooter that is really a **latency benchmark instrument** for
neural game mechanics on a mobile NPU (MediaTek APU 780 / Dimensity 8300-Ultra,
Poco X6 Pro). See `PROJECT_BRIEF.md` for the research design. The instrument
comes before the game: every measurement lands in `adb pull`-able CSVs.

## Layout

```
app/                          Android app (all game logic in native C++)
  src/main/java/...           MainActivity.java — launcher shim ONLY
  src/main/cpp/src/           engine, renderer (GLES3 + low-res bloom), game,
                              input, gesture, nn (TFLite C API), csv logger
  src/main/cpp/third_party/   LiteRT 1.4.2 headers (vendored from official AARs)
  src/main/jniLibs/arm64-v8a/ libtensorflowlite_jni.so, libtensorflowlite_gpu_jni.so
                              (pinned; SHA256 in this README)
  src/main/assets/models/     12 .tflite models produced by training/train_all.py
  src/main/cpp/src/trace.*     trace and replay recording (reproducibility study)
  src/main/cpp/src/prevention.* on-device prevention: rounding grid, guard, portable kernel
training/train_all.py         policy-net (5 sizes) + gesture-CNN training/export
analysis/                     analysis scripts; study3_*.py for the reproducibility study
ios/                          iOS harness for the reproducibility study (see ios/README.md)
tools/hostsim/                host build of the game simulation (re-simulates recordings)
tools/portable_policy/        portable scalar C kernel of the policy model
tools/x86kit/                 replay and comparison kit for an x86 PC
tools/study3_*.{py,sh}        device sessions over adb for the reproducibility study
```

## Pinned inference runtime

`com.google.ai.edge.litert:litert:1.4.2` + `litert-gpu:1.4.2` (the continuation
of `org.tensorflow:tensorflow-lite` after its relocation). Both `.so`s are
vendored in `jniLibs` so the APK always bundles this exact runtime — no Play
Services, no version drift between experiment sessions.

```
E2328CAF39606382A13DC08F06ED03EB0119A490208D375E3F42B80E4DABC465  libtensorflowlite_jni.so
228BDE7ACE97FD32EFC8905990617C65D4382AED9AB0648035741FE52D46886B  libtensorflowlite_gpu_jni.so
```

Backends: CPU (XNNPACK, 4 threads — recorded per row), GPU delegate
(`TfLiteGpuDelegateV2`, sustained-speed + min-latency), NPU (`NNAPI delegate`,
`disallow_nnapi_cpu=1`, pinned to the first NNAPI device of type ACCELERATOR,
delegate fallback disabled so backend failures are loud, `allow_fp16=1`).
NNAPI is deprecated as of Android 15 — a discussion point for the paper
(vendor path: MediaTek NeuroPilot).

## Build

Requirements: Android SDK + NDK 28.2.13676358 + CMake 3.22.1, JDK 17+
(`gradle.properties` points at Android Studio's JBR), Python 3.13 + TF 2.21
for retraining.

```powershell
# models (only when retraining; exports into app assets + verifies via host TFLite)
& "$env:USERPROFILE\tfvenv\Scripts\python.exe" training\train_all.py

# APK
.\gradlew.bat assembleRelease
# -> %USERPROFILE%\.npubench-build\app\outputs\apk\release\app-release.apk
# (build output lives OUTSIDE the repo: OneDrive's cloud reparse tags break
#  Gradle snapshotting of hard-linked .so files — see build.gradle.kts)

# install (sideload; release build is debug-signed on purpose)
adb install -r npubench-swarm-v0.1.apk
```

## Running experiments

Start screen: pick **backend** (CPU/GPU/NPU), **model** (T/S/M/L/XL =
1x16 … 3x512), **precision** (F32/INT8), **glow** (ON/OFF control run), then:

- **START RUN** — plays the level. Wave number == swarm size == inference
  batch size (8 → 80 over 10 waves); the level design IS the load sweep.
  Waves auto-advance after 40 s so the sweep completes even unattended.
- **BENCH ALL BACKENDS** — cold-start + 200-iteration steady-state
  micro-bench of the selected model at batch 32 on all three backends.

Controls: drag in the bottom band to move the ship; draw above it to cast —
circle = bomb, straight slash = beam, zigzag = shield.

The debug HUD (top-left) shows backend/model/N, last + rolling-p95 inference
ms, deadline misses, fps, battery temp, thermal headroom — watch it for
throttling during sustained runs.

## Data

CSVs land in the app external files dir:

```
adb pull /sdcard/Android/data/com.npubench.swarm/files ./runs
```

Per run id (`YYYYMMDD_HHMMSS_kind_backend_model_prec`):

- `run_<id>_ticks.csv` — per 60 Hz tick: `tick_total_ms, feat_ms,
  policy_infer_ms, sim_ms, render_ms, gpu_render_ms, glow_on,
  deadline_missed(>16.667ms), fps, warmup, batch_mode, battery_temp_c,
  thermal_headroom, thermal_status, …`
- `run_<id>_gestures.csv` — per stroke: class, confidence, `infer_ms`,
  `finger_up_to_effect_ms` (motion-event timestamp → world-state change).
- `run_<id>_coldstart.csv` — model load / delegate create / interpreter
  create / allocate / first & second inference; bench rows add steady
  mean/std/p50/p95/p99; every row carries the NNAPI device list.

Measurement discipline (from the brief): all timing native
`clock_gettime(CLOCK_MONOTONIC)`; CSVs buffered in memory and flushed only at
wave/run boundaries; first 10 ticks after every (re)compile flagged `warmup=1`;
per-wave batch resize is timed and noted on the wave's first tick row
(`resize` vs `recreate` in `batch_mode`).

`render_ms` is the CPU-side encode time of the previous frame;
`gpu_render_ms` comes from `GL_EXT_disjoint_timer_query` when available. The
bloom chain runs entirely at internal resolution (240 x ~534 scene, half-res
blur) so render cost is resolution-fixed by construction — the glow OFF
control run demonstrates it doesn't perturb tick timing.

## Model I/O contracts

- Feature spec **PF1** (per enemy, 14 floats) — defined in
  `app/src/main/cpp/src/common.h`, mirrored in `training/train_all.py`.
- Gesture spec **GS1** (64 resampled points, centroid-centered, half-max-bbox
  scaled) — `gesture.cpp` ↔ `train_all.py`.
- Policy output: `[N x 3]` tanh (steer_x, steer_y, fire).
- Ops kept NNAPI-friendly on purpose: FULLY_CONNECTED / CONV_2D / MAX_POOL_2D /
  MEAN / TANH / SOFTMAX / RESHAPE.

## Reproducibility study

A second study uses the same game to ask whether the enemy-AI model gives bit-identical
outputs, and therefore the same game, on different chips and backends, as deterministic
lockstep multiplayer needs. It covers the CPU, GPU and NPU paths of five Android devices,
two iPhones, a Mac and an x86 PC.

- Recording: with tracing on, the app writes every tick's feature rows, model outputs and
  game state (`*_trace.bin`). A replay feeds a recorded trace's feature rows to any backend
  over identical inputs (`*_replay.bin`).
- Re-simulation: `tools/hostsim/build.sh` builds the game on the Mac;
  `tools/hostsim/validate_trace.py` re-simulates each recorded game from its recorded
  outputs and checks the trace byte for byte.
- Prevention: `app/src/main/cpp/src/prevention.cpp` rounds the model outputs to a coarse
  grid and recomputes rows near a grid edge with the portable kernel
  (`tools/portable_policy/`), so every peer takes the same decision.
- Analysis: every table and figure comes from the recordings through `analysis/study3_*.py`
  (`study3_crosschip.py` for the cross-device comparison, `study3_mechanism.py`,
  `study3_guard.py`, `study3_seeds.py` with `cosim.py` and `policy_backends.py`,
  `study3_figures.py` and others); outputs go to `analysis/out_study3/`.
- iOS: `ios/README.md` (LiteRT for iOS is not in git; set your own signing team).
- x86: `tools/x86kit/README.md`.

The recordings (436 trace and replay files, 4.0 GB), the APKs, the models and the iOS
runtime archives are on Zenodo: https://doi.org/10.5281/zenodo.23660012. Unzip each
`runs_study3_<device>.zip` in the repository root to get `runs_study3/<device>/`, where
the analysis scripts expect it. Device serial numbers in that record are replaced by
`SERIAL-<device>`.

## Known caveats

- With `disallow_nnapi_cpu=1` a not-fully-delegable model makes NNAPI opt out
  silently (TFLite CPU kernels serve) — detectable via timing + captured
  TFLite error-reporter text in the coldstart `note` column. Models were kept
  to plain ops so full delegation is expected; verify on first NPU run.
- Tiny models may legitimately run FASTER on CPU than NPU (dispatch overhead):
  that is a finding, not a bug.
- Battery temperature arrives via the Java shim (`ACTION_BATTERY_CHANGED`,
  explicitly allowed by the brief) — never on a measured path.
