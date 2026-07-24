#pragma once
// Shared config + tiny utilities. This file freezes the benchmark contract:
// every constant that affects measurements or the model I/O spec lives here.

#include <android/log.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <string>
#include <vector>

#define LOG_TAG "NPUBench"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// --- timing: CLOCK_MONOTONIC only, never wall clock (PROJECT_BRIEF) ---
inline int64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}
inline double ns_to_ms(int64_t ns) { return static_cast<double>(ns) * 1e-6; }

// --- simulation / measurement contract ---
constexpr int     kTickHz        = 60;
constexpr int64_t kTickNs        = 16666667;        // 16.666 ms hard tick budget
constexpr double  kTickBudgetMs  = 16.666667;
constexpr int     kInternalW     = 240;             // low-res pixel pipeline width (portrait)
constexpr int     kWarmupTicks   = 10;              // first K inferences logged as warmup
constexpr int     kCpuThreads    = 4;               // fixed XNNPACK/CPU thread count, recorded per row
constexpr int     kWaveTable[]   = {8, 16, 24, 32, 40, 48, 56, 64, 72, 80};
constexpr int     kNumWaves      = 10;              // wave number == batch size sweep
constexpr float   kWaveTimeoutS  = 40.0f;           // wave auto-advances so the sweep always completes
constexpr float   kGestureCooldownS = 0.8f;
constexpr float   kGestureConfThreshold = 0.55f;

// --- model I/O specs (must match training/train_all.py) ---
// Feature spec "PF1", one row per enemy, batched [N x 14]:
//   0 posx_n  = x / W * 2 - 1
//   1 posy_n  = y / H * 2 - 1
//   2 velx_n  = vx / 120
//   3 vely_n  = vy / 120
//   4 dpx     = (player.x - x) / W
//   5 dpy     = (player.y - y) / H
//   6 dist    = hypot(dpx, dpy)
//   7..12     3 nearest neighbours: ((nx-x)/W)/0.25, ((ny-y)/H)/0.25, clamped [-1,1]; absent -> 0
//   13 t_sin  = sin(2*pi * t / 3s)
// Output [N x 3], all tanh: steer_x, steer_y, fire(>0.35 shoots)
// NOTE: models are exported in 1x1-conv form — tensor shape [1, N, 1, 14] /
// [1, N, 1, 3] — identical memory layout to [N x 14] / [N x 3]; N rides the
// spatial axis because the GPU delegate's GL backend rejects batch != 1.
constexpr int kFeatureDim = 14;
constexpr int kPolicyOut  = 3;
constexpr float kPolicyFireThreshold = 0.35f;

// Gesture spec "GS1": stroke resampled to 64 equidistant points,
// centroid at origin, scaled by half of max bbox dimension -> [64 x 2].
// Output: softmax over {none, circle, slash, zigzag}.
constexpr int kGesturePts     = 64;
constexpr int kGestureClasses = 4;
inline const char* kGestureNames[kGestureClasses] = {"none", "circle", "slash", "zigzag"};

enum class Backend { CPU = 0, GPU = 1, NPU = 2 };
enum class Precision { F32 = 0, INT8 = 1 };

inline const char* backendName(Backend b) {
    switch (b) {
        case Backend::CPU: return "cpu";
        case Backend::GPU: return "gpu";
        case Backend::NPU: return "npu";
    }
    return "?";
}
inline const char* precisionName(Precision p) { return p == Precision::F32 ? "f32" : "int8"; }

inline const char* kModelVariants[] = {"tiny", "small", "medium", "large", "xlarge"};
constexpr int kNumVariants = 5;

// --- tiny vector math ---
struct Vec2 {
    float x = 0.f, y = 0.f;
    Vec2() = default;
    Vec2(float x_, float y_) : x(x_), y(y_) {}
    Vec2 operator+(Vec2 o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(Vec2 o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(float s) const { return {x * s, y * s}; }
    Vec2& operator+=(Vec2 o) { x += o.x; y += o.y; return *this; }
    float len() const { return std::sqrt(x * x + y * y); }
    float len2() const { return x * x + y * y; }
    Vec2 normalized() const {
        float l = len();
        return l > 1e-6f ? Vec2{x / l, y / l} : Vec2{0.f, 0.f};
    }
};

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
