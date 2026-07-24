#include "nn.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstdarg>
#include <cstring>

#include "tflite/c/c_api_experimental.h"
#include "tflite/delegates/gpu/delegate.h"
#include "tflite/delegates/nnapi/nnapi_delegate_c_api.h"

// ---------------------------------------------------------------------------
// NNAPI device enumeration (dlopen so absence can't crash us)
// ---------------------------------------------------------------------------
namespace {

struct NnapiRuntime {
    std::string deviceList;
    std::string acceleratorName;
    NnapiRuntime() {
        void* lib = dlopen("libneuralnetworks.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib) {
            deviceList = "libneuralnetworks_unavailable";
            return;
        }
        using GetCountFn = int (*)(uint32_t*);
        using GetDeviceFn = int (*)(uint32_t, void**);
        using GetNameFn = int (*)(const void*, const char**);
        using GetTypeFn = int (*)(const void*, int32_t*);
        using GetFlFn = int (*)(const void*, int64_t*);
        auto getCount = (GetCountFn)dlsym(lib, "ANeuralNetworks_getDeviceCount");
        auto getDevice = (GetDeviceFn)dlsym(lib, "ANeuralNetworks_getDevice");
        auto getName = (GetNameFn)dlsym(lib, "ANeuralNetworksDevice_getName");
        auto getType = (GetTypeFn)dlsym(lib, "ANeuralNetworksDevice_getType");
        auto getFl = (GetFlFn)dlsym(lib, "ANeuralNetworksDevice_getFeatureLevel");
        if (!getCount || !getDevice || !getName) {
            deviceList = "nnapi_symbols_missing";
            return;
        }
        uint32_t count = 0;
        getCount(&count);
        // Prefer the dedicated deep-learning engine over other accelerators:
        // on MediaTek the MDLA is the APU's NN engine; "neuron" is the Neuron
        // adapter; DSP/VPU accelerators rank last. Rank 3 beats 2 beats 1.
        int bestRank = 0;
        for (uint32_t i = 0; i < count; i++) {
            void* dev = nullptr;
            if (getDevice(i, &dev) != 0 || !dev) continue;
            const char* name = "?";
            getName(dev, &name);
            int32_t type = -1;
            if (getType) getType(dev, &type);
            int64_t fl = -1;
            if (getFl) getFl(dev, &fl);
            char buf[160];
            // type: 1=OTHER 2=CPU 3=GPU 4=ACCELERATOR
            snprintf(buf, sizeof(buf), "%s(type%d;fl%lld)", name, type, (long long)fl);
            if (!deviceList.empty()) deviceList += "|";
            deviceList += buf;
            if (type == 4) {
                int rank = 1;
                if (strstr(name, "neuron")) rank = 2;
                if (strstr(name, "mdla")) rank = 3;
                if (rank > bestRank) {
                    bestRank = rank;
                    acceleratorName = name;
                }
            }
        }
        if (deviceList.empty()) deviceList = "no_nnapi_devices";
        deviceList += "|pick=" + (acceleratorName.empty() ? "none" : acceleratorName);
        LOGI("NNAPI devices: %s", deviceList.c_str());
    }
};

NnapiRuntime& nnapiRuntime() {
    static NnapiRuntime rt;
    return rt;
}

}  // namespace

const std::string& nnapiDeviceList() { return nnapiRuntime().deviceList; }
const std::string& nnapiAcceleratorName() { return nnapiRuntime().acceleratorName; }

SteadyStats computeStats(std::vector<double> s) {
    SteadyStats st;
    if (s.empty()) return st;
    st.iters = (int)s.size();
    double sum = 0;
    for (double v : s) sum += v;
    st.mean = sum / st.iters;
    double var = 0;
    for (double v : s) var += (v - st.mean) * (v - st.mean);
    st.stddev = std::sqrt(var / st.iters);
    std::sort(s.begin(), s.end());
    auto pct = [&](double p) {
        double idx = p * (st.iters - 1);
        int lo = (int)idx;
        int hi = lo + 1 < st.iters ? lo + 1 : lo;
        double f = idx - lo;
        return s[lo] * (1 - f) + s[hi] * f;
    };
    st.p50 = pct(0.50);
    st.p95 = pct(0.95);
    st.p99 = pct(0.99);
    return st;
}

// ---------------------------------------------------------------------------
// NnModel
// ---------------------------------------------------------------------------

void NnModel::reporterTrampoline(void* ud, const char* fmt, va_list args) {
    auto* self = static_cast<NnModel*>(ud);
    char buf[512];
    vsnprintf(buf, sizeof(buf), fmt, args);
    LOGW("TFLite: %s", buf);
    if (self->err_.size() < 4096) {
        self->err_ += buf;
        self->err_ += " / ";
    }
}

bool NnModel::load(AAssetManager* am, const std::string& assetPath, Backend backend) {
    unload();
    backend_ = backend;
    cold = ColdStartInfo{};
    err_.clear();

    int64_t t0 = now_ns();
    AAsset* asset = AAssetManager_open(am, assetPath.c_str(), AASSET_MODE_BUFFER);
    if (!asset) {
        LOGE("asset not found: %s", assetPath.c_str());
        cold.note = "asset_not_found:" + assetPath;
        return false;
    }
    size_t len = AAsset_getLength(asset);
    modelBuf_.resize(len);
    memcpy(modelBuf_.data(), AAsset_getBuffer(asset), len);
    AAsset_close(asset);

    model_ = TfLiteModelCreateWithErrorReporter(modelBuf_.data(), modelBuf_.size(),
                                                &NnModel::reporterTrampoline, this);
    cold.load_ms = ns_to_ms(now_ns() - t0);
    if (!model_) {
        cold.note = "model_parse_failed";
        return false;
    }

    std::string how;
    if (!createInterpreter(1, &cold.delegate_ms, &cold.create_ms, &cold.alloc_ms, &how)) {
        cold.note = "interpreter_create_failed:" + err_;
        return false;
    }
    curBatch_ = 1;
    invokes_ = 0;
    LOGI("model %s loaded on %s: load %.2fms delegate %.2fms create %.2fms alloc %.2fms",
         assetPath.c_str(), backendName(backend_), cold.load_ms, cold.delegate_ms,
         cold.create_ms, cold.alloc_ms);
    return true;
}

bool NnModel::createInterpreter(int batch, double* delegateMs, double* createMs,
                                double* allocMs, std::string* how) {
    destroyInterpreter();

    // --- delegate ---
    int64_t t0 = now_ns();
    if (backend_ == Backend::GPU) {
        TfLiteGpuDelegateOptionsV2 gopts = TfLiteGpuDelegateOptionsV2Default();
        gopts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
        gopts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        delegate_ = TfLiteGpuDelegateV2Create(&gopts);
        if (!delegate_) {
            err_ += "gpu_delegate_create_failed / ";
            return false;
        }
    } else if (backend_ == Backend::NPU) {
        auto* nopts = new TfLiteNnapiDelegateOptions(TfLiteNnapiDelegateOptionsDefault());
        nnapiOptsHolder_ = nopts;
        nopts->execution_preference =
            TfLiteNnapiDelegateOptions::ExecutionPreference::kSustainedSpeed;
        nopts->disallow_nnapi_cpu = 1;  // honest NPU: never let NNAPI's own CPU serve
        nopts->allow_fp16 = 1;          // fp32 models may run fp16 on the APU (recorded)
        if (!nnapiAcceleratorName().empty())
            nopts->accelerator_name = nnapiAcceleratorName().c_str();
        delegate_ = TfLiteNnapiDelegateCreate(nopts);
        if (!delegate_) {
            err_ += "nnapi_delegate_create_failed / ";
            return false;
        }
    }
    *delegateMs = ns_to_ms(now_ns() - t0);

    // --- interpreter ---
    t0 = now_ns();
    opts_ = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetErrorReporter(opts_, &NnModel::reporterTrampoline, this);
    TfLiteInterpreterOptionsSetNumThreads(opts_, kCpuThreads);
    if (delegate_) {
        TfLiteInterpreterOptionsAddDelegate(opts_, delegate_);
        // fail loudly instead of silently benchmarking CPU as "GPU"/"NPU"
        TfLiteInterpreterOptionsSetEnableDelegateFallback(opts_, false);
    }
    interp_ = TfLiteInterpreterCreate(model_, opts_);
    *createMs = ns_to_ms(now_ns() - t0);
    if (!interp_) {
        err_ += "interpreter_create_null / ";
        return false;
    }

    // capture original input dims once
    if (inputDims_.empty()) {
        TfLiteTensor* in = TfLiteInterpreterGetInputTensor(interp_, 0);
        int nd = TfLiteTensorNumDims(in);
        for (int i = 0; i < nd; i++) inputDims_.push_back(TfLiteTensorDim(in, i));
    }

    // --- resize (optional) + allocate ---
    // policy nets are [1, N, 1, feat] (N on the spatial axis, so the GPU
    // delegate's GL backend accepts it); rank-2 models batch on axis 0
    const int axis = inputDims_.size() == 4 ? 1 : 0;
    t0 = now_ns();
    if (batch != inputDims_[axis]) {
        std::vector<int> dims = inputDims_;
        dims[axis] = batch;
        if (TfLiteInterpreterResizeInputTensor(interp_, 0, dims.data(), (int32_t)dims.size()) !=
            kTfLiteOk) {
            err_ += "resize_failed_in_create / ";
            return false;
        }
    }
    if (TfLiteInterpreterAllocateTensors(interp_) != kTfLiteOk) {
        err_ += "allocate_failed / ";
        return false;
    }
    *allocMs = ns_to_ms(now_ns() - t0);
    if (how) *how = "recreate";
    return cacheTensors();
}

bool NnModel::cacheTensors() {
    inTensor_ = TfLiteInterpreterGetInputTensor(interp_, 0);
    outTensor_ = TfLiteInterpreterGetOutputTensor(interp_, 0);
    if (!inTensor_ || !outTensor_) {
        err_ += "tensor_lookup_failed / ";
        return false;
    }
    inInt8_ = TfLiteTensorType(inTensor_) == kTfLiteInt8;
    outInt8_ = TfLiteTensorType(outTensor_) == kTfLiteInt8;
    TfLiteQuantizationParams qi = TfLiteTensorQuantizationParams(inTensor_);
    TfLiteQuantizationParams qo = TfLiteTensorQuantizationParams(outTensor_);
    inScale_ = qi.scale != 0.f ? qi.scale : 1.f;
    inZero_ = qi.zero_point;
    outScale_ = qo.scale != 0.f ? qo.scale : 1.f;
    outZero_ = qo.zero_point;
    return true;
}

bool NnModel::setBatch(int n, double* setupMs, std::string* how) {
    if (!interp_) return false;
    if (n == curBatch_) {
        *setupMs = 0.0;
        *how = "noop";
        return true;
    }
    int64_t t0 = now_ns();
    std::vector<int> dims = inputDims_;
    dims[inputDims_.size() == 4 ? 1 : 0] = n;
    bool ok =
        TfLiteInterpreterResizeInputTensor(interp_, 0, dims.data(), (int32_t)dims.size()) ==
            kTfLiteOk &&
        TfLiteInterpreterAllocateTensors(interp_) == kTfLiteOk && cacheTensors();
    if (ok) {
        *how = "resize";
    } else {
        LOGW("resize to batch %d failed on %s; recreating interpreter", n,
             backendName(backend_));
        double dMs, cMs, aMs;
        if (!createInterpreter(n, &dMs, &cMs, &aMs, how)) return false;
    }
    curBatch_ = n;
    *setupMs = ns_to_ms(now_ns() - t0);
    return true;
}

bool NnModel::invoke(const float* in, int inCount, float* out, int outCount,
                     double* inferMs) {
    if (!interp_) return false;
    int64_t t0 = now_ns();

    if (inInt8_) {
        int8_t* dst = (int8_t*)TfLiteTensorData(inTensor_);
        float invScale = 1.f / inScale_;
        for (int i = 0; i < inCount; i++) {
            int q = (int)lroundf(in[i] * invScale) + inZero_;
            dst[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
        }
    } else {
        if (TfLiteTensorCopyFromBuffer(inTensor_, in, (size_t)inCount * sizeof(float)) !=
            kTfLiteOk) {
            err_ += "copy_in_failed / ";
            return false;
        }
    }

    if (TfLiteInterpreterInvoke(interp_) != kTfLiteOk) {
        err_ += "invoke_failed / ";
        return false;
    }

    if (outInt8_) {
        const int8_t* src = (const int8_t*)TfLiteTensorData(outTensor_);
        for (int i = 0; i < outCount; i++) out[i] = ((int)src[i] - outZero_) * outScale_;
    } else {
        if (TfLiteTensorCopyToBuffer(outTensor_, out, (size_t)outCount * sizeof(float)) !=
            kTfLiteOk) {
            err_ += "copy_out_failed / ";
            return false;
        }
    }

    double ms = ns_to_ms(now_ns() - t0);
    *inferMs = ms;
    invokes_++;
    if (invokes_ == 1) cold.first_ms = ms;
    if (invokes_ == 2) cold.second_ms = ms;
    return true;
}

void NnModel::destroyInterpreter() {
    if (interp_) TfLiteInterpreterDelete(interp_);
    interp_ = nullptr;
    if (opts_) TfLiteInterpreterOptionsDelete(opts_);
    opts_ = nullptr;
    if (delegate_) {
        if (backend_ == Backend::GPU) TfLiteGpuDelegateV2Delete(delegate_);
        else if (backend_ == Backend::NPU) TfLiteNnapiDelegateDelete(delegate_);
        delegate_ = nullptr;
    }
    if (nnapiOptsHolder_) {
        delete static_cast<TfLiteNnapiDelegateOptions*>(nnapiOptsHolder_);
        nnapiOptsHolder_ = nullptr;
    }
    inTensor_ = nullptr;
    outTensor_ = nullptr;
}

void NnModel::unload() {
    destroyInterpreter();
    if (model_) TfLiteModelDelete(model_);
    model_ = nullptr;
    modelBuf_.clear();
    modelBuf_.shrink_to_fit();
    inputDims_.clear();
    curBatch_ = 1;
    invokes_ = 0;
}
