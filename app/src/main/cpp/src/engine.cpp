#include "engine.h"

#include <game-activity/GameActivityEvents.h>

#include <algorithm>
#include <cstring>

std::atomic<float> Engine::batteryTemp{-1.f};

namespace {
constexpr uint32_t kCyanCol = 0x50FFF0FF;
constexpr uint32_t kMagCol = 0xFF3CDCFF;
constexpr uint32_t kDimCol = 0x9080C0FF;
constexpr uint32_t kWhiteCol = 0xFFFFFFFF;

enum BtnGroup { GB_BACKEND, GB_MODEL, GB_PREC, GB_GLOW, GB_LOOP, GB_START, GB_BENCH };
}  // namespace

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

void Engine::onWindowInit(ANativeWindow* window, AAssetManager* am, const char* filesDir) {
    assets_ = am;
    if (!renderer_.init(window)) {
        LOGE("renderer init failed");
        return;
    }
    input_.attach(&renderer_);
    if (filesDir) csv_.configure(filesDir);
    if (!thermal_) thermal_ = AThermal_acquireManager();
    lastFrameNs_ = now_ns();
    accumulatorNs_ = 0;
    fpsWindowStartNs_ = now_ns();
    // enumerate NNAPI devices once, early (also logs to logcat for the paper)
    nnapiDeviceList();
}

void Engine::onWindowLost() {
    csv_.flush();
    renderer_.destroy();
}

void Engine::onPause() { csv_.flush(); }

void Engine::handleMotionEvent(const GameActivityMotionEvent& ev) { input_.handle(ev); }

// ---------------------------------------------------------------------------
// run flow
// ---------------------------------------------------------------------------

std::string Engine::makeRunId(const char* kind) const {
    // wall clock is fine for an ID (never used for measurement)
    time_t t = time(nullptr);
    struct tm tmv;
    localtime_r(&t, &tmv);
    char buf[96];
    snprintf(buf, sizeof(buf), "%04d%02d%02d_%02d%02d%02d_%s_%s_%s_%s",
             tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday, tmv.tm_hour, tmv.tm_min,
             tmv.tm_sec, kind, backendName(backend_), kModelVariants[variant_],
             precisionName(prec_));
    return buf;
}

std::string Engine::policyAssetPath() const {
    return std::string("models/policy_") + kModelVariants[variant_] + "_" +
           precisionName(prec_) + ".tflite";
}
std::string Engine::gestureAssetPath() const {
    return std::string("models/gesture_") + precisionName(prec_) + ".tflite";
}

void Engine::startGame() {
    state_ = AppState::LOADING;
    loadingStep_ = 0;
}

void Engine::finishLoading() {
    runId_ = makeRunId("game");
    csv_.startRun(runId_);

    if (!policy_.load(assets_, policyAssetPath(), backend_)) {
        maybeWriteColdRowsForFailure();
        return;
    }
    if (!gesture_.load(assets_, gestureAssetPath(), backend_)) {
        // GPU delegate rejects the gesture CNN (EXPAND_DIMS unsupported); the
        // policy net is the measured workload, so fall back gesture-only to
        // CPU rather than failing the whole run. Actual backend lands in CSVs.
        if (backend_ != Backend::CPU &&
            gesture_.load(assets_, gestureAssetPath(), Backend::CPU)) {
            LOGW("gesture model fell back to CPU (backend %s rejected it)",
                 backendName(backend_));
        } else {
            maybeWriteColdRowsForFailure();
            return;
        }
    }
    coldPolicyPending_ = coldGesturePending_ = true;

    game_.startRun(renderer_.internalW(), renderer_.internalH());
    tickIndex_ = 0;
    missCount_ = 0;
    sumPolicyMs_ = 0;
    cntPolicy_ = 0;
    rollN_ = rollPos_ = 0;
    rollP95_ = 0;
    gestureCooldown_ = 0.f;
    beginWave(0);
    if (state_ == AppState::ERRORS) return;
    accumulatorNs_ = 0;
    lastFrameNs_ = now_ns();
    state_ = AppState::PLAYING;
}

// helper lives out-of-class-decl for clarity; writes cold rows for a failed load
void Engine::maybeWriteColdRowsForFailure() {
    ColdRow row{};
    row.run_id = runId_.c_str();
    row.phase = "cold";
    row.backend = backendName(backend_);
    row.model = kModelVariants[variant_];
    row.precision = precisionName(prec_);
    row.batch = 1;
    row.load_model_ms = policy_.cold.load_ms;
    row.delegate_create_ms = policy_.cold.delegate_ms;
    row.interpreter_create_ms = policy_.cold.create_ms;
    row.allocate_ms = policy_.cold.alloc_ms;
    row.first_infer_ms = -1;
    row.second_infer_ms = -1;
    row.steady_iters = 0;
    row.nnapi_devices = nnapiDeviceList().c_str();
    std::string note = "LOAD_FAILED: " + policy_.cold.note + " | " + policy_.errors() +
                       " | gesture: " + gesture_.cold.note + " " + gesture_.errors();
    row.note = note.c_str();
    csv_.add(row);
    csv_.flush();
    enterError("MODEL LOAD FAILED\n" + policy_.cold.note + "\n" + gesture_.cold.note +
               "\nBACKEND: " + backendName(backend_));
}

void Engine::beginWave(int idx) {
    game_.startWave(idx);
    double setupMs = 0;
    std::string how;
    if (!policy_.setBatch(kWaveTable[idx], &setupMs, &how)) {
        csv_.flush();
        enterError("BATCH SETUP FAILED AT WAVE " + std::to_string(idx + 1) + "\n" +
                   policy_.errors());
        return;
    }
    waveSetupMs_ = setupMs;
    batchHow_ = how;
    warmupLeft_ = kWarmupTicks;   // delegate may have recompiled: re-warm
    pendingTickNote_ = "wave_start";
    LOGI("wave %d: N=%d setup %.2fms (%s)", idx + 1, kWaveTable[idx], setupMs, how.c_str());
}

void Engine::enterError(const std::string& msg) {
    errorMsg_ = msg;
    state_ = AppState::ERRORS;
    LOGE("%s", msg.c_str());
}

void Engine::endRun(AppState finalState) {
    maybeWriteColdRows(true);
    csv_.flush();
    state_ = finalState;
    // ~2 s at 120 fps; never auto-replay an errored run
    replayCountdown_ = autoReplay_ && finalState != AppState::ERRORS ? 240 : 0;
}

void Engine::maybeWriteColdRows(bool force) {
    auto write = [&](NnModel& m, const char* modelName, bool& pending) {
        if (!pending) return;
        if (!force && m.invokeCount() < 2) return;
        ColdRow row{};
        row.run_id = runId_.c_str();
        row.phase = "cold";
        row.backend = backendName(m.backend());
        row.model = modelName;
        row.precision = precisionName(prec_);
        row.batch = m.batch();
        row.load_model_ms = m.cold.load_ms;
        row.delegate_create_ms = m.cold.delegate_ms;
        row.interpreter_create_ms = m.cold.create_ms;
        row.allocate_ms = m.cold.alloc_ms;
        row.first_infer_ms = m.cold.first_ms;
        row.second_infer_ms = m.cold.second_ms;
        row.steady_iters = 0;
        row.nnapi_devices = nnapiDeviceList().c_str();
        std::string note = m.errors();
        row.note = note.c_str();
        csv_.add(row);
        pending = false;
    };
    write(policy_, kModelVariants[variant_], coldPolicyPending_);
    write(gesture_, "gesture", coldGesturePending_);
}

// ---------------------------------------------------------------------------
// gestures
// ---------------------------------------------------------------------------

void Engine::handleStroke(const Stroke& s) {
    GestureResult res = classifyStroke(gesture_, s.pts);
    if (!res.ok) return;   // too short — not a gesture attempt

    bool onCooldown = gestureCooldown_ > 0.f;
    bool confident = res.conf >= kGestureConfThreshold && res.cls != 0;
    bool accepted = confident && !onCooldown;

    double fingerUpToEffect = -1.0;
    if (accepted) {
        Vec2 centroid{0, 0}, lo = s.pts[0], hi = s.pts[0];
        for (const Vec2& p : s.pts) {
            centroid += p;
            lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y);
            hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y);
        }
        centroid = centroid * (1.f / (float)s.pts.size());
        float drawnRadius = std::max(hi.x - lo.x, hi.y - lo.y) * 0.5f;
        switch (res.cls) {
            case 1: game_.castZone(centroid, drawnRadius); break;
            case 2: game_.castBeam(s.pts); break;
            case 3: game_.castShield(); break;
        }
        gestureCooldown_ = kGestureCooldownS;
        // effect latency: motion-event finger-up -> world state mutated
        fingerUpToEffect = ns_to_ms(now_ns() - s.fingerUpNs);
        if (fingerUpToEffect < 0) fingerUpToEffect = -1.0;  // clock oddity: mark invalid
    }

    GestureRow row{};
    row.run_id = runId_.c_str();
    row.backend = backendName(gesture_.backend());  // may be CPU fallback
    row.precision = precisionName(prec_);
    row.gesture_class = kGestureNames[res.cls];
    row.confidence = res.conf;
    row.stroke_points = (int)s.pts.size();
    row.resample_ms = res.resampleMs;
    row.infer_ms = res.inferMs;
    row.finger_up_to_effect_ms = fingerUpToEffect;
    row.accepted = accepted ? 1 : 0;
    row.note = !confident ? "low_conf_or_none" : (onCooldown ? "cooldown" : "");
    csv_.add(row);
    maybeWriteColdRows(false);
}

// ---------------------------------------------------------------------------
// the measured tick
// ---------------------------------------------------------------------------

void Engine::doTick() {
    const int64_t t0 = now_ns();

    // 1. features (CPU cost, scales ~N^2 from neighbour search — logged separately)
    int64_t t = now_ns();
    const int alive = game_.buildFeatures(featBuf_);
    // Inference always runs at the wave's fixed batch: dead enemies become
    // zero rows (outputs ignored). Keeps the measured workload constant across
    // the whole wave — one clean (backend x model x N) cell per wave.
    const int batch = policy_.batch();
    if (alive < batch)
        memset(featBuf_ + alive * kFeatureDim, 0,
               sizeof(float) * (size_t)(batch - alive) * kFeatureDim);
    const double featMs = ns_to_ms(now_ns() - t);

    // 2. batched policy inference — THE measured workload
    double policyMs = 0;
    if (!policy_.invoke(featBuf_, batch * kFeatureDim, outBuf_, batch * kPolicyOut,
                        &policyMs)) {
        csv_.flush();
        enterError("POLICY INVOKE FAILED\n" + policy_.errors());
        return;
    }

    // 3. world update
    t = now_ns();
    if (alive > 0) game_.applyPolicy(outBuf_, alive);
    game_.simulate(1.f / (float)kTickHz);
    if (gestureCooldown_ > 0) gestureCooldown_ -= 1.f / (float)kTickHz;
    const double simMs = ns_to_ms(now_ns() - t);

    const double tickTotalMs = ns_to_ms(now_ns() - t0);
    const bool missed = tickTotalMs > kTickBudgetMs;
    if (missed) missCount_++;

    // rolling HUD stats (steady ticks only)
    const bool warm = warmupLeft_ > 0;
    if (!warm) {
        rollBuf_[rollPos_] = policyMs;
        rollPos_ = (rollPos_ + 1) % 120;
        if (rollN_ < 120) rollN_++;
        sumPolicyMs_ += policyMs;
        cntPolicy_++;
        if ((tickIndex_ % 30) == 0 && rollN_ > 10) {
            double tmp[120];
            memcpy(tmp, rollBuf_, sizeof(double) * rollN_);
            std::sort(tmp, tmp + rollN_);
            rollP95_ = tmp[(int)(0.95 * (rollN_ - 1))];
        }
    }

    // wave / death transitions (evaluated before logging so the row carries the note)
    const char* note = pendingTickNote_;
    pendingTickNote_ = "";
    char noteBuf[128];
    if (note[0] != '\0') {
        snprintf(noteBuf, sizeof(noteBuf), "%s;setup_ms=%.3f;mode=%s", note, waveSetupMs_,
                 batchHow_.c_str());
        note = noteBuf;
    }

    TickRow row{};
    row.run_id = runId_.c_str();
    row.backend = backendName(backend_);
    row.model_variant = kModelVariants[variant_];
    row.precision = precisionName(prec_);
    row.batch_mode = batchHow_.c_str();
    row.wave = game_.waveIndex() + 1;
    row.batch_size = batch;
    row.alive = alive;
    row.tick_index = tickIndex_;
    row.warmup = warm ? 1 : 0;
    row.tick_total_ms = tickTotalMs;
    row.feat_ms = featMs;
    row.policy_infer_ms = policyMs;
    row.sim_ms = simMs;
    row.render_ms = renderCpuMs_;
    row.gpu_render_ms = renderGpuMs_;
    row.glow_on = glow_ ? 1 : 0;
    row.deadline_missed = missed ? 1 : 0;
    row.fps = fps_;
    row.cpu_threads = kCpuThreads;
    row.battery_temp_c = batteryTemp.load();
    row.thermal_headroom = thermalHeadroom_;
    row.thermal_status = thermalStatus_;
    row.note = note;
    csv_.add(row);

    if (warm) warmupLeft_--;
    tickIndex_++;
    maybeWriteColdRows(false);

    if (game_.playerDead()) {
        endRun(AppState::GAMEOVER);
        return;
    }
    if (game_.waveCleared() || game_.waveTimedOut()) {
        const int next = game_.waveIndex() + 1;
        csv_.flush();   // wave boundary: safe flush point (outside measured tick)
        if (next >= kNumWaves) {
            endRun(AppState::VICTORY);
        } else {
            pendingTickNote_ = game_.waveCleared() ? "wave_start" : "wave_start_after_timeout";
            beginWave(next);
        }
    }
}

// ---------------------------------------------------------------------------
// thermal
// ---------------------------------------------------------------------------

void Engine::sampleThermal() {
    if (!thermal_) return;
    if (++thermalPollCounter_ % 60 != 0) return;   // ~2x per second at 120 fps
    thermalStatus_ = (int)AThermal_getCurrentThermalStatus(thermal_);
    float hr = AThermal_getThermalHeadroom(thermal_, 0);
    thermalHeadroom_ = std::isnan(hr) ? -1.f : hr;
}

// ---------------------------------------------------------------------------
// frame
// ---------------------------------------------------------------------------

void Engine::frame() {
    if (!renderer_.ready()) return;
    sampleThermal();

    // fps window
    fpsFrames_++;
    int64_t nowT = now_ns();
    if (nowT - fpsWindowStartNs_ > 500000000LL) {
        fps_ = fpsFrames_ * 1e9 / (double)(nowT - fpsWindowStartNs_);
        fpsFrames_ = 0;
        fpsWindowStartNs_ = nowT;
    }

    // input routing by state
    auto taps = input_.takeTaps();
    auto strokes = input_.takeStrokes();
    switch (state_) {
        case AppState::MENU:
            for (const Vec2& p : taps) handleMenuTap(p);
            break;
        case AppState::PLAYING:
            for (const Stroke& s : strokes) {
                handleStroke(s);
                if (state_ != AppState::PLAYING) break;
            }
            break;
        case AppState::GAMEOVER:
        case AppState::VICTORY:
            if (!taps.empty()) {
                replayCountdown_ = 0;   // a tap always cancels the loop
                state_ = AppState::MENU;
            } else if (replayCountdown_ > 0 && --replayCountdown_ == 0) {
                startGame();
            }
            break;
        case AppState::ERRORS:
            if (!taps.empty()) state_ = AppState::MENU;
            break;
        case AppState::BENCH:
            if (benchIdx_ >= 3 && !taps.empty()) state_ = AppState::MENU;
            break;
        default:
            break;
    }

    // fixed-timestep simulation
    if (state_ == AppState::PLAYING) {
        int64_t now = now_ns();
        accumulatorNs_ += now - lastFrameNs_;
        lastFrameNs_ = now;
        if (accumulatorNs_ > 4 * kTickNs) accumulatorNs_ = kTickNs;  // hitch/pause guard
        int guard = 0;
        while (accumulatorNs_ >= kTickNs && guard < 3 && state_ == AppState::PLAYING) {
            doTick();
            accumulatorNs_ -= kTickNs;
            guard++;
        }
    } else {
        lastFrameNs_ = now_ns();
    }

    // render
    renderer_.beginFrame();
    switch (state_) {
        case AppState::MENU:
            renderMenu();
            break;
        case AppState::LOADING:
            renderLoading();
            break;
        case AppState::PLAYING: {
            game_.render(renderer_);
            // gesture trail (world phase -> it glows)
            const auto& live = input_.liveStroke();
            for (size_t i = 1; i < live.size(); i++)
                renderer_.line(live[i - 1], live[i], 2.f, rgba(80, 255, 240, 160), true);
            renderer_.applyBloom(glow_);
            renderPlayingOverlay();
            break;
        }
        case AppState::BENCH:
            renderBench();
            break;
        case AppState::GAMEOVER:
            renderEndScreen("GAME OVER");
            break;
        case AppState::VICTORY:
            renderEndScreen("SWEEP COMPLETE");
            break;
        case AppState::ERRORS:
            renderErrors();
            break;
    }
    // MENU and PLAYING call applyBloom themselves (they have world content);
    // the remaining screens do it here so every frame runs the pipeline once.
    if (state_ != AppState::PLAYING && state_ != AppState::MENU)
        renderer_.applyBloom(glow_);
    renderer_.endFrame(&renderCpuMs_, &renderGpuMs_);

    // deferred heavy work AFTER a frame was shown
    if (state_ == AppState::LOADING) {
        if (loadingStep_ == 0) {
            loadingStep_ = 1;         // one frame of "LOADING..." first
        } else {
            finishLoading();
        }
    } else if (state_ == AppState::BENCH && benchIdx_ < 3) {
        benchStep();
    }
}

// ---------------------------------------------------------------------------
// bench: cold + steady per backend for the current model selection
// ---------------------------------------------------------------------------

void Engine::startBench() {
    state_ = AppState::BENCH;
    benchIdx_ = 0;
    benchLines_.clear();
    runId_ = makeRunId("bench");
    csv_.startRun(runId_);
}

void Engine::benchStep() {
    Backend b = (Backend)benchIdx_;
    constexpr int kBenchBatch = 32;
    constexpr int kBenchWarmup = 10;
    constexpr int kBenchIters = 200;

    NnModel m;
    ColdRow row{};
    row.run_id = runId_.c_str();
    row.phase = "bench";
    row.backend = backendName(b);
    row.model = kModelVariants[variant_];
    row.precision = precisionName(prec_);
    row.batch = kBenchBatch;
    row.nnapi_devices = nnapiDeviceList().c_str();

    char line[160];
    std::string note;
    if (!m.load(assets_, policyAssetPath(), b)) {
        note = "LOAD_FAILED: " + m.cold.note + " " + m.errors();
        row.note = note.c_str();
        row.load_model_ms = m.cold.load_ms;
        row.first_infer_ms = -1;
        csv_.add(row);
        snprintf(line, sizeof(line), "%s: FAILED (%s)", backendName(b), m.cold.note.c_str());
        benchLines_.push_back(line);
        benchIdx_++;
        if (benchIdx_ >= 3) csv_.flush();
        return;
    }

    double setupMs = 0;
    std::string how;
    if (!m.setBatch(kBenchBatch, &setupMs, &how)) {
        note = "SET_BATCH_FAILED: " + m.errors();
        row.note = note.c_str();
        csv_.add(row);
        snprintf(line, sizeof(line), "%s: BATCH FAILED", backendName(b));
        benchLines_.push_back(line);
        benchIdx_++;
        if (benchIdx_ >= 3) csv_.flush();
        return;
    }

    // deterministic pseudo-features
    uint32_t s = 0x1234567u;
    float in[kBenchBatch * kFeatureDim];
    for (float& v : in) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        v = ((s & 0xFFFF) / 32768.f) - 1.f;
    }
    float out[kBenchBatch * kPolicyOut];

    std::vector<double> samples;
    samples.reserve(kBenchIters);
    double ms = 0;
    bool ok = true;
    for (int i = 0; i < kBenchWarmup + kBenchIters && ok; i++) {
        ok = m.invoke(in, kBenchBatch * kFeatureDim, out, kBenchBatch * kPolicyOut, &ms);
        if (ok && i >= kBenchWarmup) samples.push_back(ms);
    }

    SteadyStats st = computeStats(samples);
    row.load_model_ms = m.cold.load_ms;
    row.delegate_create_ms = m.cold.delegate_ms;
    row.interpreter_create_ms = m.cold.create_ms;
    row.allocate_ms = m.cold.alloc_ms;
    row.first_infer_ms = m.cold.first_ms;
    row.second_infer_ms = m.cold.second_ms;
    row.steady_iters = st.iters;
    row.steady_mean_ms = st.mean;
    row.steady_std_ms = st.stddev;
    row.steady_p50_ms = st.p50;
    row.steady_p95_ms = st.p95;
    row.steady_p99_ms = st.p99;
    note = (ok ? "" : "INVOKE_FAILED ") + m.errors() + ";setup=" + how;
    row.note = note.c_str();
    csv_.add(row);

    snprintf(line, sizeof(line), "%s: 1ST %.2f  MEAN %.3f  P95 %.3f", backendName(b),
             m.cold.first_ms, st.mean, st.p95);
    benchLines_.push_back(line);
    LOGI("bench %s", line);

    benchIdx_++;
    if (benchIdx_ >= 3) csv_.flush();
}

// ---------------------------------------------------------------------------
// UI rendering
// ---------------------------------------------------------------------------

void Engine::menuButton(int group, int index, float x, float y, float w, float h,
                        const char* label, bool selected) {
    btns_.push_back({x, y, w, h, group, index});
    uint32_t border = selected ? kCyanCol : kDimCol;
    uint32_t fill = selected ? rgba(20, 90, 90, 200) : rgba(20, 16, 44, 200);
    renderer_.rect(x, y, w, h, fill);
    renderer_.rect(x, y, w, 1, border);
    renderer_.rect(x, y + h - 1, w, 1, border);
    renderer_.rect(x, y, 1, h, border);
    renderer_.rect(x + w - 1, y, 1, h, border);
    renderer_.textCentered((int)(x + w / 2), (int)(y + h / 2 - 3), label,
                           selected ? kWhiteCol : kDimCol);
}

void Engine::handleMenuTap(Vec2 p) {
    for (const Btn& b : btns_) {
        if (p.x < b.x || p.x > b.x + b.w || p.y < b.y || p.y > b.y + b.h) continue;
        switch (b.group) {
            case GB_BACKEND: backend_ = (Backend)b.index; break;
            case GB_MODEL: variant_ = b.index; break;
            case GB_PREC: prec_ = (Precision)b.index; break;
            case GB_GLOW: glow_ = b.index == 0; break;
            case GB_LOOP: autoReplay_ = !autoReplay_; break;
            case GB_START: startGame(); break;
            case GB_BENCH: startBench(); break;
        }
        return;
    }
}

void Engine::renderMenu() {
    const int W = renderer_.internalW();
    btns_.clear();

    // glowing title (world phase -> bloomed)
    renderer_.textCentered(W / 2, 30, "NPU BENCH", kCyanCol, 3);
    renderer_.textCentered(W / 2, 56, "SWARM", kMagCol, 3);
    renderer_.applyBloom(glow_);

    renderer_.textCentered(W / 2, 84, "NEURAL LATENCY INSTRUMENT", kDimCol);

    auto rowLabel = [&](int y, const char* s) { renderer_.text(10, y + 5, s, kDimCol); };

    int y = 108;
    rowLabel(y, "BACKEND");
    const char* bnames[3] = {"CPU", "GPU", "NPU"};
    for (int i = 0; i < 3; i++)
        menuButton(GB_BACKEND, i, 68.f + i * 56.f, (float)y, 52, 18, bnames[i],
                   (int)backend_ == i);

    y = 136;
    rowLabel(y, "MODEL");
    const char* mnames[kNumVariants] = {"T", "S", "M", "L", "192", "256", "XL"};
    for (int i = 0; i < kNumVariants; i++)
        menuButton(GB_MODEL, i, 68.f + i * 24.f, (float)y, 22, 18, mnames[i], variant_ == i);

    y = 164;
    rowLabel(y, "PREC");
    menuButton(GB_PREC, 0, 68, (float)y, 52, 18, "F32", prec_ == Precision::F32);
    menuButton(GB_PREC, 1, 124, (float)y, 52, 18, "INT8", prec_ == Precision::INT8);

    y = 192;
    rowLabel(y, "GLOW");
    menuButton(GB_GLOW, 0, 68, (float)y, 52, 18, "ON", glow_);
    menuButton(GB_GLOW, 1, 124, (float)y, 52, 18, "OFF", !glow_);
    menuButton(GB_LOOP, 0, 184, (float)y, 48, 18, "LOOP", autoReplay_);

    menuButton(GB_START, 0, (float)(W / 2 - 70), 228, 140, 26, "START RUN", true);
    menuButton(GB_BENCH, 0, (float)(W / 2 - 90), 264, 180, 20, "BENCH ALL BACKENDS", false);

    // instrument info
    int iy = 300;
    renderer_.text(8, iy, "TFLITE C API 1.4.2  THREADS 4", kDimCol);
    iy += 10;
    renderer_.text(8, iy, "TICK 60HZ  BUDGET 16.6MS", kDimCol);
    iy += 14;
    renderer_.text(8, iy, "NNAPI DEVICES:", kDimCol);
    iy += 10;
    // wrap device list
    const std::string& dev = nnapiDeviceList();
    for (size_t off = 0; off < dev.size() && iy < renderer_.internalH() - 40; off += 38) {
        renderer_.text(8, iy, dev.substr(off, 38).c_str(), kDimCol);
        iy += 9;
    }
    iy += 8;
    renderer_.text(8, iy, "CSV: ANDROID/DATA/COM.NPUBENCH.SWARM", kDimCol);
    iy += 10;
    renderer_.text(8, iy, "AUTOFIRE ON - DRAW ANYWHERE TO CAST", kDimCol);
    iy += 10;
    renderer_.text(8, iy, "O=BOMB  /=BEAM  W=SHIELD", kDimCol);
}

void Engine::renderLoading() {
    const int W = renderer_.internalW(), H = renderer_.internalH();
    renderer_.textCentered(W / 2, H / 2 - 20, "LOADING MODEL", kCyanCol, 2);
    char buf[80];
    snprintf(buf, sizeof(buf), "%s %s %s", backendName(backend_), kModelVariants[variant_],
             precisionName(prec_));
    renderer_.textCentered(W / 2, H / 2 + 4, buf, kDimCol);
}

void Engine::renderPlayingOverlay() {
    char buf[96];
    // debug HUD (brief: from day one)
    snprintf(buf, sizeof(buf), "%s %s %s W%d N%d", backendName(backend_),
             kModelVariants[variant_], precisionName(prec_), game_.waveIndex() + 1,
             game_.waveSize());
    renderer_.text(4, 4, buf, kCyanCol);
    double last = rollN_ > 0 ? rollBuf_[(rollPos_ + 119) % 120] : 0.0;
    snprintf(buf, sizeof(buf), "INF %.2f P95 %.2f MS", last, rollP95_);
    renderer_.text(4, 13, buf, kCyanCol);
    snprintf(buf, sizeof(buf), "MISS %d FPS %.0f", missCount_, fps_);
    renderer_.text(4, 22, buf, missCount_ > 0 ? kMagCol : kCyanCol);
    snprintf(buf, sizeof(buf), "T %.1fC HR %.2f S%d", batteryTemp.load(), thermalHeadroom_,
             thermalStatus_);
    renderer_.text(4, 31, buf, kDimCol);

    // right side: hp + score
    snprintf(buf, sizeof(buf), "HP %d", game_.hp());
    renderer_.text(renderer_.internalW() - 40, 4, buf, kMagCol);
    snprintf(buf, sizeof(buf), "%d", game_.score());
    renderer_.text(renderer_.internalW() - 40, 13, buf, kWhiteCol);
}

void Engine::renderEndScreen(const char* title) {
    const int W = renderer_.internalW(), H = renderer_.internalH();
    renderer_.textCentered(W / 2, H / 2 - 60, title, kMagCol, 2);
    char buf[96];
    snprintf(buf, sizeof(buf), "SCORE %d  WAVE %d", game_.score(), game_.waveIndex() + 1);
    renderer_.textCentered(W / 2, H / 2 - 30, buf, kWhiteCol);
    double mean = cntPolicy_ > 0 ? sumPolicyMs_ / cntPolicy_ : 0.0;
    snprintf(buf, sizeof(buf), "MEAN INF %.3f MS  P95 %.3f", mean, rollP95_);
    renderer_.textCentered(W / 2, H / 2 - 16, buf, kCyanCol);
    snprintf(buf, sizeof(buf), "DEADLINE MISSES %d", missCount_);
    renderer_.textCentered(W / 2, H / 2 - 2, buf, missCount_ > 0 ? kMagCol : kCyanCol);
    renderer_.textCentered(W / 2, H / 2 + 24, "CSV FLUSHED", kDimCol);
    if (replayCountdown_ > 0)
        renderer_.textCentered(W / 2, H / 2 + 48, "LOOP: REPLAYING - TAP TO STOP", kWhiteCol);
    else
        renderer_.textCentered(W / 2, H / 2 + 48, "TAP FOR MENU", kWhiteCol);
}

void Engine::renderBench() {
    const int W = renderer_.internalW();
    renderer_.textCentered(W / 2, 40, "BACKEND BENCH", kCyanCol, 2);
    char buf[96];
    snprintf(buf, sizeof(buf), "MODEL %s %s  BATCH 32", kModelVariants[variant_],
             precisionName(prec_));
    renderer_.textCentered(W / 2, 64, buf, kDimCol);
    int y = 90;
    for (const std::string& l : benchLines_) {
        renderer_.text(8, y, l.c_str(), kWhiteCol);
        y += 12;
    }
    if (benchIdx_ < 3) {
        snprintf(buf, sizeof(buf), "RUNNING %s...", backendName((Backend)benchIdx_));
        renderer_.text(8, y, buf, kCyanCol);
    } else {
        renderer_.textCentered(W / 2, y + 16, "DONE - CSV FLUSHED", kCyanCol);
        renderer_.textCentered(W / 2, y + 32, "TAP FOR MENU", kWhiteCol);
    }
}

void Engine::renderErrors() {
    const int W = renderer_.internalW();
    renderer_.textCentered(W / 2, 60, "BACKEND ERROR", kMagCol, 2);
    int y = 90;
    // crude wrap of the error text
    for (size_t off = 0; off < errorMsg_.size() && y < renderer_.internalH() - 30;) {
        size_t nl = errorMsg_.find('\n', off);
        std::string lineStr = errorMsg_.substr(off, std::min(nl == std::string::npos
                                                                 ? errorMsg_.size() - off
                                                                 : nl - off,
                                                             (size_t)38));
        renderer_.text(6, y, lineStr.c_str(), kWhiteCol);
        y += 10;
        off += lineStr.size();
        if (nl != std::string::npos && off >= nl) off = nl + 1;
    }
    renderer_.textCentered(W / 2, renderer_.internalH() - 24, "TAP FOR MENU", kWhiteCol);
}
