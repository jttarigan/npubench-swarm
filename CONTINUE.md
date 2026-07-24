# CONTINUE — session state (updated 2026-07-24, session 2)

Read this + `PROJECT_BRIEF.md` + `README.md` to resume from any machine.

## Where the project stands

Build-order steps 1–5 are DONE. The full 2×2 money matrix (CPU/NPU × F32/INT8,
XL model, 10-wave in-game sweeps) is measured, plus sustained back-to-back
series on both backends. Step 6 (`analysis/analyze.py`) is written and being
validated (check `analysis/out/` exists and regenerates cleanly).

## Results (all CSVs pulled into `runs/`)

Per-wave steady-state policy inference, XL (~330k params), N = 8→80:

- **CPU F32**: mean 2.5→6.0 ms scaling with N; p99 up to 11.5 ms (70% of the
  16.7 ms budget); jitter high (p95 ≈ 3–4× mean). 0 deadline misses.
- **NPU F32**: mean 0.91–0.98 ms FLAT across N; p99 ≤ 1.6 ms. 0 misses.
- **CPU INT8**: mean 5.1→7.6 ms — SLOWER than CPU F32 (~2× at small N,
  ~1.25× at N=80). Confirmed not thermal (CPU F32 at 39–40 °C ≈ same as at
  34 °C). Per-tensor quant on XNNPACK loses to f32 sgemm here — a finding.
- **NPU INT8**: mean 0.95–1.00 ms flat — no gain over F32 (allow_fp16=1 means
  the f32 model already runs fp16 on the MDLA; dispatch floor dominates).
- **NNAPI batch-resize recompile: ~98–118 ms at EVERY wave boundary** (vs
  0.05–0.66 ms CPU resize), precision-independent. ≈7 dropped-tick stall at
  each swarm-size change. Headline deployment finding: dynamic shapes are
  hostile to the NPU path — pad to max batch or pre-compile.
- **INT8 per-channel rejection**: mtk-mdla_shim throws ANEURALNETWORKS_BAD_DATA
  ("while adding operation") on per-channel quantized weights; NNAPI never
  supported per-channel FULLY_CONNECTED at all. Fixed by per-tensor export
  (`conv._experimental_disable_per_channel = True` in train_all.py). Failure
  evidence preserved: `runs/run_20260724_1634{41,45}_*_coldstart.csv`.
- **Thermal/sustained**: 3× CPU + 2× NPU back-to-back runs (~50 s each,
  waves clear fast). Zero throttling (thermal_status 0, headroom 0.69–0.83);
  CPU series pinned at 39 °C; NPU series COOLED during play (38.0→37.7 °C).
  NPU offload is a thermal win too.
- Cold start NPU: delegate 7–22 ms + interpreter create 185 ms (int8) /
  259 ms (f32), first infer ~3–4 ms vs ~1 ms steady.
- Repetitions: CPU F32 ×4, NPU F32 ×3, CPU INT8 ×1, NPU INT8 ×1.

## NEXT STEPS (in order)

1. **INT8 reps**: 2 more game runs each of CPU+XL+INT8 and NPU+XL+INT8 so all
   four cells have n≥3 (~4 min of play).
2. Validate `analysis/analyze.py` end-to-end (should emit tick_stats.csv,
   summary.md + 6 PNGs into `analysis/out/`); iterate on chart quality for
   the paper figures.
3. Optional: true 10-min sustained session per backend (≈12 back-to-back
   runs) if reviewers will want a longer thermal claim than ~3–5 min.
4. Optional: ~130k-param policy variant (e.g. 3x256) to pin the static
   crossover (bench said ≈130k params at batch 32, between large and xlarge).
5. Optional cosmetics: '|' renders as '?' in the 5x7 font (menu device list).
6. Write the paper. Story arc: crossover chart + flat-NPU scaling + tail/jitter
   analysis + reconfiguration/cold-start delays + quantization anti-findings
   (per-channel rejected; int8 no faster on NPU, slower on CPU) + thermal.

## Machine setup notes (both machines now validated)

- Work machine: see session-1 notes below. Home machine additionally:
  - adb NOT on PATH — use `%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe`.
  - No tfvenv: system `python` (3.10) has TF 2.21 + numpy — train_all.py runs
    directly (28 s).
  - **Debug keystores differ between machines** → `adb install -r` fails with
    INSTALL_FAILED_UPDATE_INCOMPATIBLE after building on the other machine.
    Must uninstall first, WHICH WIPES on-device CSVs — `adb pull` everything
    before uninstalling. Xiaomi then gates the fresh install: user must
    confirm the on-phone dialog (INSTALL_FAILED_USER_RESTRICTED otherwise).
  - OneDrive placeholders: on a freshly-synced machine run `attrib -U +P /S /D`
    in repo root first, else git/file reads fail on unhydrated files.
- Android SDK + NDK 28.2.13676358 + CMake 3.22.1 + platform android-36;
  Android Studio JBR as JDK (`gradle.properties` hardcodes the path — edit if
  Studio lives elsewhere).
- Build: `.\gradlew.bat assembleRelease` → APK OUTSIDE repo at
  `%USERPROFILE%\.npubench-build\...` (OneDrive breaks Gradle in-tree);
  repo-root `npubench-swarm-v0.1.apk` is the current installable copy
  (contains the per-tensor int8 models; f32 models byte-identical to
  session-1 APK — f32 assets were restored from git after re-export).
- Phone: Poco X6 Pro via USB; data pull:
  `adb pull /sdcard/Android/data/com.npubench.swarm/files ./runs`.

## Hard-won build facts (do not rediscover)

- OneDrive reparse tags break AGP hard-links → all build output + staged
  jniLibs live in `%USERPROFILE%\.npubench-build\` (wired via syncLitertLibs
  + `-DLITERT_LIB_DIR`).
- games-activity 4.x: needs androidx.appcompat + AppCompat theme, kotlin-bom
  1.8.22, linker flag `-u Java_..._GameActivity_initializeNativeCode`.
- PS 5.1 `-Encoding utf8` writes a BOM that breaks Gradle settings files.
- Batch on SPATIAL axis ([1,N,1,14]); never move N to batch axis (GL delegate
  rejects it). In-game inference always runs the wave's full batch.
- INT8 exports MUST be per-tensor (see finding above) — the flag is already
  set in train_all.py; don't remove it.
- Training seeded (SEED=7) but NOT bit-reproducible across machines/numpy
  versions — re-export changes f32 bytes too. To keep f32 assets identical to
  measured runs: `git checkout -- app/src/main/assets/models/*_f32.tflite`
  after retraining (already done for the current tree).

## Model I/O contracts (frozen)

- PF1 feature spec: block comment in `app/src/main/cpp/src/common.h`
  (mirrored in `training/train_all.py`).
- GS1 gesture spec: 64 arc-length-resampled points, centroid-centred,
  half-max-bbox scaled; classes none/circle/slash/zigzag.
