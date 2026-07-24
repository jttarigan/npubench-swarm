#pragma once
// LiteRT/TFLite C API wrapper: one model + one backend + timing.
// Backends: CPU (fixed threads), GPU delegate, NNAPI delegate (NPU path).
// All timestamps CLOCK_MONOTONIC via now_ns(). No allocation in invoke().

#include <android/asset_manager.h>

#include <string>
#include <vector>

#include "common.h"
#include "tflite/c/c_api.h"

struct ColdStartInfo {
    double load_ms = -1, delegate_ms = -1, create_ms = -1, alloc_ms = -1;
    double first_ms = -1, second_ms = -1;   // filled by the first two invoke() calls
    std::string note;
};

struct SteadyStats {
    int iters = 0;
    double mean = 0, stddev = 0, p50 = 0, p95 = 0, p99 = 0;
};
SteadyStats computeStats(std::vector<double> samples);

// "name(type,featureLevel);..." — enumerated once via dlopen(libneuralnetworks)
const std::string& nnapiDeviceList();
// preferred dedicated accelerator device name ("" if none found)
const std::string& nnapiAcceleratorName();

class NnModel {
public:
    ~NnModel() { unload(); }

    bool load(AAssetManager* am, const std::string& assetPath, Backend backend);
    void unload();
    bool loaded() const { return interp_ != nullptr; }
    Backend backend() const { return backend_; }
    int invokeCount() const { return invokes_; }
    const std::string& errors() const { return err_; }
    ColdStartInfo cold;

    // Set batch dimension of input 0. Tries ResizeInputTensor first, falls back
    // to full interpreter recreation (both timed). how = "resize" | "recreate".
    bool setBatch(int n, double* setupMs, std::string* how);
    int batch() const { return curBatch_; }

    // Float in / float out end-to-end (int8 models quantize/dequantize inside;
    // that cost is part of the honest per-inference latency).
    bool invoke(const float* in, int inCount, float* out, int outCount, double* inferMs);

private:
    bool createInterpreter(int batch, double* delegateMs, double* createMs, double* allocMs,
                           std::string* how);
    void destroyInterpreter();
    bool cacheTensors();
    static void reporterTrampoline(void* ud, const char* fmt, va_list args);

    std::vector<char> modelBuf_;
    TfLiteModel* model_ = nullptr;
    TfLiteInterpreterOptions* opts_ = nullptr;
    TfLiteInterpreter* interp_ = nullptr;
    TfLiteDelegate* delegate_ = nullptr;   // GPU or NNAPI, else null
    void* nnapiOptsHolder_ = nullptr;      // keeps NNAPI options alive for delegate lifetime

    Backend backend_ = Backend::CPU;
    int curBatch_ = 1;
    int invokes_ = 0;
    std::vector<int> inputDims_;   // original dims of input 0 (batch at [0])

    TfLiteTensor* inTensor_ = nullptr;
    const TfLiteTensor* outTensor_ = nullptr;
    bool inInt8_ = false, outInt8_ = false;
    float inScale_ = 1.f, outScale_ = 1.f;
    int inZero_ = 0, outZero_ = 0;

    std::string err_;
};
