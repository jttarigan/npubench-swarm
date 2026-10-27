#include "Harness.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <vector>

#include <TensorFlowLiteC/TensorFlowLiteC.h>
#include <TensorFlowLiteC/c_api_experimental.h>
#include <TensorFlowLiteCCoreML/TensorFlowLiteCCoreML.h>
#include <TensorFlowLiteCMetal/TensorFlowLiteCMetal.h>

#include "game.h"
#include "trace.h"

// game.cpp's render() references these; the harness never draws (as tools/hostsim).
void Renderer::sprite(SpriteId, Vec2, float, uint32_t, bool, float) {}
void Renderer::rect(float, float, float, float, uint32_t, bool) {}
void Renderer::line(Vec2, Vec2, float, uint32_t, bool) {}
void Renderer::text(int, int, const char*, uint32_t, int) {}
int Renderer::textWidth(const char*, int) const { return 0; }
void Renderer::textCentered(int, int, const char*, uint32_t, int) {}

const char* backendTag(IosBackend b) {
    switch (b) {
        case IosBackend::CPU: return "cpu";
        case IosBackend::GPU: return "gpu";
        case IosBackend::NPU: return "npu";
    }
    return "?";
}

std::string liteRtVersion() { return TfLiteVersion(); }

bool g_legacyDelegateOrder = false;

namespace {

constexpr int kMaxBatch = 96;

// The floating-point control register of the calling thread (rounding mode,
// flush-to-zero). The game's arithmetic depends on it, so the harness logs
// any change a backend makes to it during inference.
inline uint64_t fpcr() { return __builtin_arm_rsr64("fpcr"); }

double nowMs() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// Games take about a second here, so the id carries milliseconds and a
// per-process sequence number (two runs in one second must not share files:
// TraceWriter appends).
std::string runId(const char* kind, const HarnessConfig& c) {
    static int seq = 0;
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    tm tmv;
    localtime_r(&ts.tv_sec, &tmv);
    char buf[160];
    snprintf(buf, sizeof(buf), "%04d%02d%02d_%02d%02d%02d%03ld_%03d_%s_%s_%s_%s",
             tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday, tmv.tm_hour, tmv.tm_min,
             tmv.tm_sec, ts.tv_nsec / 1000000, ++seq, kind, backendTag(c.backend),
             c.variant.c_str(), c.int8 ? "int8" : "f32");
    return buf;
}

// LiteRT model with the Android app's I/O conventions (nn.cpp): input shape
// [1, N, 1, 14], int8 I/O quantized with lroundf + clamp, (q - zp) * scale.
class Model {
public:
    ~Model() { destroy(); if (model_) TfLiteModelDelete(model_); }

    bool load(const std::string& path, IosBackend backend, const LogFn& log) {
        backend_ = backend;
        log_ = log;
        model_ = TfLiteModelCreateFromFile(path.c_str());
        if (!model_) { log("model load failed: " + path); return false; }
        return create(8);
    }

    // Same rule as NnModel::setBatch: resize + allocate; the delegates get a
    // fresh interpreter (Core ML and Metal compile for a fixed shape).
    bool setBatch(int n) {
        if (n == batch_) return true;
        if (backend_ == IosBackend::CPU) {
            int dims[4] = {1, n, 1, kFeatureDim};
            if (TfLiteInterpreterResizeInputTensor(interp_, 0, dims, 4) == kTfLiteOk &&
                TfLiteInterpreterAllocateTensors(interp_) == kTfLiteOk) {
                batch_ = n;
                return cache();
            }
            log_("cpu resize failed, recreating");
        }
        return create(n);
    }

    bool invoke(const float* in, int inCount, float* out, int outCount, double* ms) {
        double t0 = nowMs();
        if (inInt8_) {
            int8_t* dst = (int8_t*)TfLiteTensorData(in_);
            const float inv = 1.f / inScale_;
            for (int i = 0; i < inCount; i++) {
                int q = (int)lroundf(in[i] * inv) + inZero_;
                dst[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
            }
        } else if (TfLiteTensorCopyFromBuffer(in_, in, sizeof(float) * inCount) != kTfLiteOk) {
            return false;
        }
        if (TfLiteInterpreterInvoke(interp_) != kTfLiteOk) return false;
        if (outInt8_) {
            const int8_t* src = (const int8_t*)TfLiteTensorData(out_);
            for (int i = 0; i < outCount; i++) out[i] = ((int)src[i] - outZero_) * outScale_;
        } else if (TfLiteTensorCopyToBuffer(out_, out, sizeof(float) * outCount) != kTfLiteOk) {
            return false;
        }
        *ms = nowMs() - t0;
        return true;
    }

    int batch() const { return batch_; }

private:
    bool create(int n) {
        destroy();
        if (backend_ == IosBackend::GPU) {
            TFLGpuDelegateOptions o = TFLGpuDelegateOptionsDefault();
            o.allow_precision_loss = true;   // as the Android GPU delegate (fp16 allowed)
            o.wait_type = TFLGpuDelegateWaitTypePassive;
            o.enable_quantization = true;
            delegate_ = TFLGpuDelegateCreate(&o);
        } else if (backend_ == IosBackend::NPU) {
            TfLiteCoreMlDelegateOptions o{};
            o.enabled_devices = TfLiteCoreMlDelegateDevicesWithNeuralEngine;
            o.coreml_version = 0;            // highest available
            o.max_delegated_partitions = 0;  // all
            o.min_nodes_per_partition = 1;
            delegate_ = TfLiteCoreMlDelegateCreate(&o);
        }
        if (backend_ != IosBackend::CPU && !delegate_) {
            log_(std::string(backendTag(backend_)) + " delegate create failed");
            return false;
        }
        opts_ = TfLiteInterpreterOptionsCreate();
        TfLiteInterpreterOptionsSetNumThreads(opts_, 4);   // kCpuThreads on Android
        // The delegate is applied AFTER the input is resized to this batch.
        // Adding it at creation and resizing afterwards (the Android app's
        // order) let the Metal delegate write into freed host memory on its
        // first invoke on iOS (2026-10-16: enemy fields of the game were
        // overwritten with fp16-precision values in some launches).
        if (delegate_ && g_legacyDelegateOrder) {
            TfLiteInterpreterOptionsAddDelegate(opts_, delegate_);
            TfLiteInterpreterOptionsSetEnableDelegateFallback(opts_, false);
        }
        interp_ = TfLiteInterpreterCreate(model_, opts_);
        if (!interp_) { log_("interpreter create failed"); return false; }
        int dims[4] = {1, n, 1, kFeatureDim};
        if (TfLiteInterpreterResizeInputTensor(interp_, 0, dims, 4) != kTfLiteOk ||
            TfLiteInterpreterAllocateTensors(interp_) != kTfLiteOk) {
            log_("allocate failed at batch " + std::to_string(n));
            return false;
        }
        if (delegate_ && !g_legacyDelegateOrder &&
            TfLiteInterpreterModifyGraphWithDelegate(interp_, delegate_) != kTfLiteOk) {
            log_(std::string(backendTag(backend_)) + " delegate not applied");
            return false;
        }
        batch_ = n;
        return cache();
    }

    bool cache() {
        in_ = TfLiteInterpreterGetInputTensor(interp_, 0);
        out_ = TfLiteInterpreterGetOutputTensor(interp_, 0);
        if (!in_ || !out_) return false;
        inInt8_ = TfLiteTensorType(in_) == kTfLiteInt8;
        outInt8_ = TfLiteTensorType(out_) == kTfLiteInt8;
        TfLiteQuantizationParams qi = TfLiteTensorQuantizationParams(in_);
        TfLiteQuantizationParams qo = TfLiteTensorQuantizationParams(out_);
        inScale_ = qi.scale != 0.f ? qi.scale : 1.f;
        inZero_ = qi.zero_point;
        outScale_ = qo.scale != 0.f ? qo.scale : 1.f;
        outZero_ = qo.zero_point;
        return true;
    }

    void destroy() {
        if (interp_) TfLiteInterpreterDelete(interp_);
        interp_ = nullptr;
        if (opts_) TfLiteInterpreterOptionsDelete(opts_);
        opts_ = nullptr;
        if (delegate_) {
            if (backend_ == IosBackend::GPU) TFLGpuDelegateDelete(delegate_);
            else TfLiteCoreMlDelegateDelete(delegate_);
        }
        delegate_ = nullptr;
        batch_ = -1;
    }

    IosBackend backend_ = IosBackend::CPU;
    LogFn log_;
    TfLiteModel* model_ = nullptr;
    TfLiteInterpreterOptions* opts_ = nullptr;
    TfLiteInterpreter* interp_ = nullptr;
    TfLiteDelegate* delegate_ = nullptr;
    TfLiteTensor* in_ = nullptr;
    const TfLiteTensor* out_ = nullptr;
    bool inInt8_ = false, outInt8_ = false;
    float inScale_ = 1.f, outScale_ = 1.f;
    int inZero_ = 0, outZero_ = 0;
    int batch_ = -1;
};

std::string modelPath(const HarnessConfig& c, const std::string& dir) {
    return dir + "/policy_" + c.variant + (c.int8 ? "_int8" : "_f32") + ".tflite";
}

}  // namespace

bool runGame(const HarnessConfig& cfg, const std::string& modelDir, const std::string& outDir,
             const LogFn& log) {
    Model m;
    if (!m.load(modelPath(cfg, modelDir), cfg.backend, log)) return false;
    const std::string id = runId("game", cfg);
    const std::string prefix = outDir + "/run_" + id;
    TraceWriter trace;
    remove((prefix + "_trace.bin").c_str());
    trace.start(prefix + "_trace.bin", "NPBT");
    FILE* csv = fopen((prefix + "_ticks.csv").c_str(), "w");
    fprintf(csv, "run_id,backend,model_variant,precision,tick_index,wave,batch_size,alive,"
                 "policy_infer_ms,out_hash,state_hash,note\n");

    static float feat[kMaxBatch * kFeatureDim], out[kMaxBatch * kPolicyOut];
    std::vector<uint8_t> state;
    Game g;
    g.startRun(kInternalW, kInternalH);
    g.startWave(0);
    const char* endNote = "tick_limit";
    int tick = 0;
    const uint64_t fpcrStart = fpcr();
    int fpcrChanges = 0;
    uint64_t state0 = 0;
    for (; tick < 20000; tick++) {
        const uint64_t fpcrTick = fpcr();
        if (fpcrTick != fpcrStart && fpcrChanges++ < 3) {
            char b[96];
            snprintf(b, sizeof(b), "FPCR at tick %d start: %llx (run start %llx)", tick,
                     (unsigned long long)fpcrTick, (unsigned long long)fpcrStart);
            log(b);
        }
        const int alive = g.buildFeatures(feat);
        const int batch = kWaveTable[g.waveIndex()];
        if (!m.setBatch(batch)) { fclose(csv); return false; }
        memset(feat + alive * kFeatureDim, 0,
               sizeof(float) * (size_t)(batch - alive) * kFeatureDim);
        double ms = 0;
        if (!m.invoke(feat, batch * kFeatureDim, out, batch * kPolicyOut, &ms)) {
            log("invoke failed at tick " + std::to_string(tick));
            fclose(csv);
            return false;
        }
        const uint64_t fpcrAfter = fpcr();
        if (fpcrAfter != fpcrStart && fpcrChanges++ < 3) {
            char b[96];
            snprintf(b, sizeof(b), "FPCR after invoke at tick %d: %llx (run start %llx)", tick,
                     (unsigned long long)fpcrAfter, (unsigned long long)fpcrStart);
            log(b);
        }
        // self-check on the first ticks: the same step on a copy of the game
        // (taken before the step) must give the same state
        const bool check = tick < 3;
        Game shadow;
        std::vector<float> outCopy;
        if (check) { shadow = g; outCopy.assign(out, out + (size_t)alive * kPolicyOut); }
        if (alive > 0) g.applyPolicy(out, alive);
        g.simulate(1.f / (float)kTickHz);
        if (check) {
            if (alive > 0) shadow.applyPolicy(outCopy.data(), alive);
            shadow.simulate(1.f / (float)kTickHz);
            std::vector<uint8_t> a, b;
            g.appendState(a);
            shadow.appendState(b);
            const bool outSame = memcmp(outCopy.data(), out, outCopy.size() * sizeof(float)) == 0;
            if (a != b || !outSame) {
                char m[160];
                snprintf(m, sizeof(m), "SELF-CHECK tick %d: state %s, out buffer %s (hash now %016llx)",
                         tick, a == b ? "same" : "DIFFERS", outSame ? "same" : "CHANGED",
                         (unsigned long long)fnv1a64(out, outCopy.size() * sizeof(float)));
                log(m);
            }
        }

        state.clear();
        g.appendState(state);
        const uint64_t oh = fnv1a64(out, sizeof(float) * (size_t)alive * kPolicyOut);
        const uint64_t sh = fnv1a64(state.data(), state.size());
        trace.addTick(tick, g.waveIndex() + 1, batch, alive, feat, out, g.aliveIndices(), state);
        if (tick == 0) state0 = sh;

        const char* note = "";
        bool end = false;
        if (g.playerDead()) { endNote = note = "game_over"; end = true; }
        else if (g.waveCleared() || g.waveTimedOut()) {
            trace.flush();
            if (g.waveIndex() + 1 >= kNumWaves) { endNote = note = "victory"; end = true; }
            else g.startWave(g.waveIndex() + 1);
        }
        fprintf(csv, "%s,%s,%s,%s,%d,%d,%d,%d,%.4f,%016llx,%016llx,%s\n", id.c_str(),
                backendTag(cfg.backend), cfg.variant.c_str(), cfg.int8 ? "int8" : "f32", tick,
                g.waveIndex() + 1, batch, alive, ms, (unsigned long long)oh,
                (unsigned long long)sh, note);
        if (end) break;
    }
    trace.flush();
    fclose(csv);
    char buf[160];
    snprintf(buf, sizeof(buf), "%s: %d ticks, wave %d, score %d, hp %d, %s, fpcr %llx, state0 %016llx",
             id.c_str(), tick + 1, g.waveIndex() + 1, g.score(), g.hp(), endNote,
             (unsigned long long)fpcrStart, (unsigned long long)state0);
    log(buf);
    return true;
}

bool runReplay(const HarnessConfig& cfg, const std::string& modelDir,
               const std::string& inputPath, const std::string& outDir, const LogFn& log) {
    TraceReader in;
    if (!in.load(inputPath)) { log("replay input: " + in.err()); return false; }
    Model m;
    if (!m.load(modelPath(cfg, modelDir), cfg.backend, log)) return false;
    const std::string id = runId("replay", cfg);
    const std::string prefix = outDir + "/run_" + id;
    TraceWriter outBin;
    remove((prefix + "_replay.bin").c_str());
    outBin.start(prefix + "_replay.bin", "NPBR");
    FILE* csv = fopen((prefix + "_replay.csv").c_str(), "w");
    char src[64];
    snprintf(src, sizeof(src), "replay_input.bin#%016llx", (unsigned long long)in.fileHash());
    fprintf(csv, "run_id,backend,model_variant,precision,source,tick_index,wave,batch_size,"
                 "alive,policy_infer_ms,out_hash\n");
    static float feat[kMaxBatch * kFeatureDim], out[kMaxBatch * kPolicyOut];
    for (const TraceTick& t : in.ticks()) {
        if (t.batch < t.alive || t.batch > kMaxBatch || !m.setBatch(t.batch)) {
            log("replay: bad batch or setup failure");
            fclose(csv);
            return false;
        }
        memcpy(feat, t.feat, sizeof(float) * (size_t)t.alive * kFeatureDim);
        memset(feat + t.alive * kFeatureDim, 0,
               sizeof(float) * (size_t)(t.batch - t.alive) * kFeatureDim);
        double ms = 0;
        if (!m.invoke(feat, t.batch * kFeatureDim, out, t.batch * kPolicyOut, &ms)) {
            log("replay invoke failed");
            fclose(csv);
            return false;
        }
        fprintf(csv, "%s,%s,%s,%s,%s,%d,%d,%d,%d,%.4f,%016llx\n", id.c_str(),
                backendTag(cfg.backend), cfg.variant.c_str(), cfg.int8 ? "int8" : "f32", src,
                t.tick, t.wave, t.batch, t.alive, ms,
                (unsigned long long)fnv1a64(out, sizeof(float) * (size_t)t.alive * kPolicyOut));
        outBin.addReplayTick(t.tick, t.wave, t.batch, t.alive, out);
    }
    outBin.flush();
    fclose(csv);
    log(id + ": " + std::to_string(in.ticks().size()) + " ticks from " + src);
    return true;
}
