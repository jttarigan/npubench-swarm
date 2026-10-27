# PROJECT BRIEF — NPU Game Benchmark (Research Project)

## READ THIS FIRST

**This is a research project, not just a game.** The game is a benchmark instrument.
The deliverable is a Scopus-indexed conference proceeding paper. **Data collection is
the top priority** — every architectural decision must preserve clean, timestamped,
exportable measurements. When in doubt between "nicer gameplay" and "cleaner data,"
choose cleaner data. A feature that cannot be measured is not worth building.

- Target paper: a latency-benchmarking study of neural game mechanics on a
  mobile NPU (MediaTek APU 780); title and venue are decided at submission.
- Timeline: 1–2 days of development + measurement. Keep scope brutally minimal.
- Metrics: **delay** (input-to-effect latency, cold-start/init latency) and
  **processing time** (per-inference time, per-tick time), across CPU / GPU / NPU
  execution backends.

## Target device

- **Poco X6 Pro** — MediaTek **Dimensity 8300-Ultra**, NPU = **APU 780**,
  GPU = Mali-G615 MC6, display up to 120 Hz.
- Runs as a normal installed Android app (NOT Termux — Termux cannot reach the NPU).
- NPU access via **LiteRT/TFLite NNAPI delegate** (note in paper: NNAPI is deprecated
  as of Android 15; mention MediaTek NeuroPilot as vendor path — a discussion point,
  not necessarily an implementation requirement).
- Follow-up studies may add other phones and tablets. The current device list is
  kept outside this repo.

## The game (benchmark harness)

A single-level 2D pixel-art **swarm shooter with gesture-cast abilities**. Touch-only.

- Player drags a ship at the bottom of the screen (one-finger drag).
- Enemy swarms attack in waves. **Wave number = swarm size = inference batch size**:
  e.g. wave 1 = 8 enemies, scaling to ~80 by wave 8. The level design IS the load sweep.
- Player draws gestures to fire abilities: **3 gesture classes only**
  (circle = bomb, straight slash = beam, zigzag = shield).
- One level, waves escalate, game over on death or after final wave. Nothing else.
  No menus beyond a start screen with backend selection (CPU/GPU/NPU) — that
  selector is required for experiments.

## Two neural workloads (the point of the paper)

1. **Enemy policy net** (every tick, batched): small MLP, input = per-enemy feature
   vector (~12–16 floats: own position/velocity, delta to player, deltas to 2–3
   nearest neighbors), batched as [N × features], output = steering/attack decision.
   Trained offline (imitation of a scripted heuristic is fine — behavior quality is
   NOT the research question; latency is). Export to .tflite.
2. **Gesture CNN** (sporadic, on input path): on finger-up, resample stroke to fixed
   length (32–64 normalized points), tiny 1D-CNN classifier, 3 classes + none.
   Small self-recorded dataset is fine (record in-app, dump to storage).

These give two contrasting inference profiles: batched every-tick vs. sporadic
input-path. Additionally build a **model-size sweep**: export the policy net at
3–5 widths/depths (e.g. tiny/small/medium/large) to find the **CPU-vs-NPU crossover
point** — expected headline figure of the paper. NOTE: tiny models may legitimately
run FASTER on CPU than NPU (dispatch overhead) — that is a finding, not a bug.
Also log and report any NNAPI **delegate fallbacks to CPU** (which ops, which models).

## Tech stack (decided — do not revisit)

- **All native C++.** Kotlin/Java is a launcher shim only (manifest + GameActivity).
  No Kotlin overlays, no Java game logic, no JNI hops inside measured paths.
- **GameActivity** (NDK) for lifecycle + native touch input queue.
- **OpenGL ES 3.0**: one texture atlas, GL_NEAREST (pixel-art look), render to small
  internal framebuffer (~400×240), blit upscaled. One batched sprite draw + blit.
  Rendering must stay cheap — the paper's story is inference cost, not render cost.
- **Art direction: neon synthwave pixel glow** (game will launch publicly later, so
  it must look good, but never at the cost of measurement cleanliness):
  - Dark purple/navy background, neon cyan/magenta/blue palette, emissive sprites,
    optional perspective grid backdrop + sparse star particles.
  - **Pixelated bloom, done INSIDE the low-res pipeline**: render emissive sprites
    to the ~400×240 framebuffer → bright-pass extract → separable blur at ½ or ¼
    res (~200×120) → additive composite back at low res → THEN one GL_NEAREST
    upscale to screen. Never blur at native screen resolution — that produces a
    smooth "Photoshop glow" that breaks the pixel-art look AND costs more GPU.
  - Bloom cost is resolution-fixed and independent of entity count → render time
    stays flat as swarm size N grows. This is deliberate and research-critical:
    inference must remain the only per-frame cost that scales with N.
  - Add a **glow on/off runtime toggle** and log it; one control run with glow off
    demonstrates rendering does not perturb tick timing.
  - Log **`render_ms` as its own CSV column** (see logging section) to show render
    cost is constant across waves while inference scales.
- **LiteRT/TFLite C API** in the same .so; delegates: none (CPU, fixed thread count
  — record it), GPU delegate, NNAPI delegate. Backend switchable at runtime or at
  minimum via start-screen selection; backend must be recorded in every log row.
- **Fixed-timestep simulation at 60 Hz** → hard, citable 16.6 ms tick budget,
  independent of the 120 Hz display. Render interpolation optional; sim tick is the
  measured deadline.
- Inference **synchronous in-tick** first (honest deadline semantics). Async
  one-tick-behind worker mode is an optional stretch toggle (nice extra subsection).
- Offline training in Python (PyTorch or TF), export .tflite, ship in APK assets.
  Include the training + export scripts in the repo — reproducibility matters for
  the paper.

## Data collection (NON-NEGOTIABLE requirements)

- All timestamps native-side via `clock_gettime(CLOCK_MONOTONIC)`. Never wall clock,
  never Java-side timing.
- **Per-tick CSV log** (buffered in memory, flushed to app-external storage on
  wave end / game end so logging doesn't perturb timing). Columns at minimum:
  `run_id, backend, model_variant, wave, batch_size(N), tick_index,
  tick_total_ms, policy_infer_ms, render_ms, glow_on(0/1), deadline_missed(0/1),
  fps, thermal_note`.
- **Per-gesture CSV log**: `run_id, backend, gesture_class, confidence,
  stroke_points, infer_ms, finger_up_to_effect_ms`.
- **Cold-start log**: model load ms + first-inference ms per model per backend
  (this is the "delay" metric alongside input-path delay).
- **Warmup discipline**: discard first K inferences (e.g. 10) from steady-state
  stats; log them separately as warmup.
- Repetitions: every (backend × model_variant × wave) cell needs enough ticks for
  stable stats — report mean, std, p50, p95, p99, and deadline-miss rate.
- **Thermal/sustained run**: one long session per backend (10+ min) to plot
  latency-over-time and show throttling. Log battery temp if accessible
  (sysfs/BatteryManager via the shim is acceptable for this one non-latency value).
- **Debug HUD from day one** (tiny bitmap font): backend, model variant, N,
  last inference ms, rolling p95, deadline misses, fps. Needed to see throttling
  live during runs.
- CSVs must land somewhere `adb pull`-able (app external files dir). Include a
  small Python analysis script that ingests the CSVs and emits the paper's tables
  + charts (latency vs batch size per backend; crossover chart across model sizes;
  latency-over-time thermal plot; gesture delay distributions).

## Experiment matrix (what the game must support end-to-end)

| Factor | Levels |
|---|---|
| Backend | CPU (fixed threads), GPU delegate, NNAPI (NPU) |
| Policy model size | 3–5 variants (sweep for crossover) |
| Batch size N | via waves, ~8 → ~80 |
| Workload type | batched policy (per tick) vs gesture CNN (sporadic) |
| Scheduling (stretch) | sync in-tick vs async one-behind |

## Known pitfalls (learned in planning — do not rediscover)

- Termux cannot reach NNAPI/NPU → must be a normal installed app.
- Tiny models can lose to CPU on NPU dispatch overhead → expected, it's a finding.
- NNAPI delegate may silently fall back to CPU for unsupported ops (esp. anything
  exotic; keep models to plain dense/conv/relu ops) → detect and log fallbacks.
- Keep gesture classes at 3 — each class costs self-recorded training data.
- Don't let rendering cost grow — it contaminates tick timing.
- INT8 quantization is likely needed for the best NPU path; export both float32 and
  int8 variants if time allows, and record which is used in every log row.

## Suggested build order (fits 1–2 days)

1. GameActivity + GLES skeleton, fixed 60 Hz loop, touch input, debug HUD, CSV logger.
2. LiteRT C API integration + backend selector; benchmark a dummy model on all three
   backends; verify NPU actually engages (timing difference + fallback logging).
3. Python: train scripted-imitation policy net (all size variants) + gesture CNN;
   export .tflite; drop into assets.
4. Game: player ship, swarm waves driven by policy net, gestures → abilities.
5. Runs: full experiment matrix, thermal session, `adb pull` CSVs.
6. Analysis script → tables + charts.

Steps 1, 2, and the logging in particular must be solid before gameplay polish of
any kind — **the instrument comes before the game.**
