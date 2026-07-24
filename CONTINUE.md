# CONTINUE — session state (updated 2026-07-24)

Read this + `PROJECT_BRIEF.md` + `README.md` to resume from any machine.

## Where the project stands

Build-order steps 1–4 are DONE and validated on the Poco X6 Pro:

- Full native C++ app (GameActivity 4.4.2, GLES3 low-res bloom pipeline,
  60 Hz fixed tick, CSV logger, debug HUD) — builds clean.
- LiteRT 1.4.2 C API vendored + pinned; backends CPU / GPU delegate / NNAPI.
- 12 models trained + in assets (policy tiny/small/medium/large/xlarge x
  f32/int8 as 1x1-CONV form `[1,N,1,14]`, gesture CNN x f32/int8).
- Gameplay (tuned with the user): ship fixed at centre, autofire, enemies
  spawn offscreen and converge (policy-driven, speed cap 55), whole screen is
  gesture canvas. Circle = damage zone 6–10 s (radius = drawn size, 75%->30%
  alpha gradient, DPS 3.0 centre -> 25% rim, dissolves bullets after 1 s in).
  Slash = 2 s beam wall (2.5 DPS, destroys enemy bullets on touch).
  Zigzag = 4 s shield. HP 10 shown as circular ring around ship, refills
  every wave. Waves 8→80 enemies over 10 waves, 40 s auto-advance.
- APK: `npubench-swarm-v0.1.apk` in repo root (debug-signed, installable).

## Results so far (in `runs/`)

- Bench (batch 32, f32, steady mean ms): medium CPU 0.037 / GPU 0.795 / NPU
  0.485; large CPU 0.131 / NPU 0.522; xlarge CPU 2.146 / GPU 2.190 /
  NPU **0.517**. NPU flat ~0.5 ms across 100x params (dispatch floor);
  **CPU->NPU crossover ≈ 130k params at batch 32** (between large and xlarge).
- NPU compile cold-start: 108–141 ms (grows with model), first infer ~2–3 ms.
- Full in-game 10-wave sweep on CPU/medium: inference 0.078→0.209 ms
  (N=8→80), 0 deadline misses, render_ms flat vs N, no throttling (31.7 C).
- Device facts: NNAPI devices mtk-dsp_shim / mtk-mdla_shim / mtk-neuron_shim
  (+ nnapi-reference); we pin **mtk-mdla_shim** (rank mdla>neuron>other).
  HyperOS libOpenCL is broken (missing clEnqueueReleaseEGLObjectsKHR) so the
  GPU delegate always falls back to OpenGL — captured in CSV notes; GL
  rejects batch>1, hence the 1x1-conv model form. GPU timer queries
  (GL_EXT_disjoint_timer_query) unavailable on this Mali driver -> -1 column.

## NEXT STEPS (in order)

1. **The money runs**: full 10-wave game runs CPU+XL+F32 and NPU+XL+F32
   (user plays; ~5 min each). Expect CPU deadline misses late, NPU flat.
2. INT8 pair (CPU+XL+INT8, NPU+XL+INT8) — quantization axis.
3. Thermal: one 10+ min sustained session per backend (replay runs
   back-to-back; battery temp + AThermal headroom are already per-tick).
4. Step 6: write `analysis/analyze.py` (only `analysis/quick_summary.py`
   exists) — paper tables + charts: latency-vs-batch per backend, crossover
   chart across model sizes, latency-over-time thermal plot, gesture delay
   distributions. CSV schemas are in README.
5. Optional: add ~130k-param policy variant (e.g. 3x256) to pin the static
   crossover; retrain script is fast (~35 s).
6. Optional cosmetics: '|' renders as '?' in the 5x7 font (menu device list).

## Machine setup to build (new machine)

- Android SDK + **NDK 28.2.13676358** (`app/build.gradle.kts` ndkVersion is
  authoritative) + CMake 3.22.1 + platform android-36; Android Studio's
  JBR (17+) as JDK.
- `gradle.properties` hardcodes `org.gradle.java.home` to
  `C:\Program Files\Android\Android Studio\jbr` — EDIT if Studio lives
  elsewhere at home.
- `local.properties` (gitignored/machine-specific) must point sdk.dir at the
  local SDK — Android Studio regenerates it on open.
- Build: `.\gradlew.bat assembleRelease` → APK lands OUTSIDE the repo at
  `%USERPROFILE%\.npubench-build\app\outputs\apk\release\` (OneDrive breaks
  Gradle if outputs live in the synced tree — do NOT move build dirs back).
- Retraining only: `python -m venv %USERPROFILE%\tfvenv` (SHORT path — long
  paths break TF install), `pip install -r training\requirements.txt`, run
  `training\train_all.py` (writes straight into app assets, ~35 s).
- Phone: enable Developer options → USB debugging + **Install via USB**
  (Xiaomi gate). Install: `adb install -r npubench-swarm-v0.1.apk`.
  Pull data: `adb pull /sdcard/Android/data/com.npubench.swarm/files ./runs`.

## Hard-won build facts (do not rediscover)

- OneDrive stamps cloud reparse tags on synced files; AGP hard-links .so
  files and Gradle then fails "not a regular file". Fix (already wired):
  all build output + a `syncLitertLibs` staged copy of jniLibs live in
  `%USERPROFILE%\.npubench-build\`. CMake gets the staged dir via
  `-DLITERT_LIB_DIR`.
- games-activity 4.x: needs `androidx.appcompat` dep + an AppCompat theme
  (`Theme.NpuBench`), kotlin-bom 1.8.22 to dedupe stdlib, and the linker flag
  `-u Java_com_google_androidgamesdk_GameActivity_initializeNativeCode`.
- PS 5.1 `-Encoding utf8` writes a BOM that breaks Gradle settings files.
- Batch handling: models are exported with N on the SPATIAL axis
  ([1,N,1,14]); `NnModel::setBatch` resizes axis 1 for rank-4 inputs.
  Never move N back to the batch axis — GL delegate rejects it.
- In-game the inference always runs at the wave's full batch (dead enemies
  = zero rows); `alive` is a separate CSV column.

## Model I/O contracts (frozen)

- PF1 feature spec: see block comment in `app/src/main/cpp/src/common.h`
  (mirrored in `training/train_all.py`). Player near centre, enemy pos may
  exceed [-1,1] (offscreen spawns), teacher fires at dist<0.85 & t_sin>0.35.
- GS1 gesture spec: 64 arc-length-resampled points, centroid-centred,
  half-max-bbox scaled; classes none/circle/slash/zigzag.
