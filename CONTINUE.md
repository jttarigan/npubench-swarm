# CONTINUE — session state (updated 2026-07-24, session 2 END)

Read this + `PROJECT_BRIEF.md` + `README.md` to resume from any machine.

## Where the project stands

ALL MEASUREMENT IS DONE. Build-order steps 1–6 complete: three backends
in-game (CPU/GPU/NPU), both precisions, crossover bracketed by bench with
l192/l256 variants, sustained LOOP series on all backends, and
`analysis/analyze.py` validated over the full dataset (156 files, 121k tick
rows -> `analysis/out/`: tick_stats.csv, summary.md, 6 charts). What remains
is chart polish + WRITING THE PAPER.

App additions this session (all in the current APK + installed on device):
7 model variants (T/S/M/L/192/256/XL), LOOP auto-replay toggle (unattended
deterministic runs), gesture-CNN CPU fallback when a backend rejects it
(GPU rejects EXPAND_DIMS; gesture CSV logs actual backend), '|' font glyph.

## Results (all CSVs pulled into `runs/`)

Per-wave steady-state policy inference, XL (534,531 params — NOT ~330k as
earlier notes said), N = 8→80:

- **In-game backend profiles at XL/F32**: NPU ~0.9–1.0 ms FLAT; CPU 2.5→6 ms
  scaling with N; GPU ~7.4 ms FLAT (batch-insensitive like NPU but 7.5x
  slower; OpenGL-fallback path). Three distinct shapes — key figure.
- **Crossover pinned by l192/l256 bench (batch 32, F32)**: CPU wins at 78k
  (0.21 vs 0.55 ms), NPU wins at 136k (0.51 vs 1.02 ms) → crossover ≈ 100k
  params. INT8 crossover: CPU int8 beats NPU even at XL in MICRO-BENCH
  (0.35 vs 0.57 ms) — but see inversion below.
- **HEADLINE: micro-bench vs in-situ inversion.** CPU XL INT8: bench 0.35 ms
  but IN-GAME 5.1–7.6 ms (~18x degradation from contention with sim/render);
  CPU F32 degrades only ~2x (2.15 -> ~4.5); NPU ~1.7x (0.57 -> 0.97). Isolated
  microbenchmarks invert the precision ranking vs in-game measurement — the
  core justification for the game-as-instrument method. NPU is not just fast
  but load-immune (predictability argument).
- **NNAPI batch-resize recompile ~98–118 ms at EVERY wave boundary** (vs
  0.05–0.66 ms CPU), precision-independent. ≈7 dropped ticks per swarm-size
  change: dynamic shapes are hostile to NPU — pad or pre-compile.
- **INT8 per-channel rejection**: mtk-mdla_shim throws ANEURALNETWORKS_BAD_DATA
  on per-channel weights; NNAPI never supported per-channel FULLY_CONNECTED.
  Fixed via per-tensor export (flag in train_all.py). Evidence:
  `runs/run_20260724_1634{41,45}_*_coldstart.csv`.
- **LOOP mode = deterministic instrument**: every unattended run is EXACTLY
  3624 ticks to victory (autofire clears all 10 waves, no player variance).
  Unattended CPU runs show the first real deadline misses (1–6/run, from
  higher alive-counts raising sim load).
- **Thermal**: ~8.5 min sustained CPU (8 LOOP runs) flat 35.9 C; ~4.5 min NPU
  flat 35.0 C; GPU series 36–37 C. Zero throttling anywhere (status 0);
  NPU sessions COOL the device. Battery temp + headroom per-tick in CSVs.
- Cold start NPU: delegate 7–22 ms + interpreter create 185 ms (int8) /
  259 ms (f32), first infer ~3–4 ms vs ~1 ms steady.
- Repetitions at XL (final): CPU F32 x19 (incl. 4 glow-OFF control runs),
  NPU F32 x8, GPU F32 x6, CPU INT8 x5, NPU INT8 x8. Dataset: 66 game runs,
  159,735 tick rows, 161 coldstart rows, 1,306 gestures (2026-07-24/25).
- Glow-OFF control DONE: render-encode and inference stats unchanged vs
  glow ON — closes the brief's control-run promise.
- In-situ inversion final number: CPU XL INT8 degrades 15.1x (0.35 bench ->
  5.30 ms in-game at N=32); NPU 1.5-1.8x.

## Manuscript status (paper/manuscript.docx)

Journal manuscript draft written by desktop Claude, then verified/filled
here against
primary data: Table 1 variants, Android 16/HyperOS 3.0 build, dates, rep
counts, Tables 2-4 fully updated to the final 66-run dataset, 13x->15.1x
inversion propagated through abstract/intro/discussion/conclusions, glow-off
control sentence added, airplane-mode claim corrected (never used), energy
bracket resolved as future work. Backups: manuscript_peerj_cs.backup*.docx;
patch scripts kept in paper/ for provenance. Author metadata, funding
(none), licenses (MIT code / CC BY 4.0 data+models), and GitHub URL are
FILLED. Public repo: https://github.com/jttarigan/npubench-swarm — pushed
from the `public` branch (filtered history: no paper/, no AI co-author
trailers, author normalized to full name; re-filter from main + force-push
to update). Zenodo deposit folder ready at %USERPROFILE%\zenodo_deposit
(299 files, 54.5 MB, checksums + full DATA_README with CSV dictionaries).

REMAINING: (1) user uploads zenodo_deposit/ contents to zenodo.org →
receives DOI; (2) patch DOI into manuscript ([Zenodo DOI: 10.5281/
zenodo.XXXXXXX]) and DATA_README.md, regenerate SHA256SUMS, re-zip/upload;
(3) Acknowledgements section still says [Optional.] — write or delete;
(4) delete the drafting-note paragraph on the manuscript title page
("Manuscript draft prepared for submission... placeholders...") before
submission; (5) final human proofread, then submit to the target journal
(first choice recorded in the manuscript itself; artifact/repo materials are
deliberately journal- and title-neutral so a resubmission elsewhere under a
different title requires no artifact changes).

## NEXT STEPS (in order)

1. **Write the paper.** Story arc: (a) crossover chart with l192/l256 pins
   (≈100k params F32); (b) three in-game backend profiles (NPU flat/low,
   CPU scaling, GPU flat/high); (c) tail/jitter schedulability analysis;
   (d) micro-bench vs in-situ INVERSION (CPU INT8 0.35 ms bench vs 5–7.6 ms
   in-game) — the method's core justification; (e) delay costs: cold start +
   ~110 ms NNAPI recompile per shape change; (f) deployment friction:
   per-channel rejection, broken OpenCL, gesture-CNN GPU rejection;
   (g) thermal: no throttling, NPU cools the device.
2. Chart polish for camera-ready: consider log-y for crossover.png (budget
   line compresses it), label the crossover point, maybe split latency_vs_N
   into mean and p95 panels. Iterate `analysis/analyze.py`.
3. Optional extra data (cheap now with LOOP): more INT8 reps, longer NPU/GPU
   thermal sessions, glow-OFF control runs (brief promises one; still unrun
   at XL — 1 LOOP session with GLOW OFF would close it).
4. Optional: gesture-delay deep dive (gesture CSVs already rich: 912 strokes).

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
