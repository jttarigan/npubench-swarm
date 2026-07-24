#pragma once
// Buffered CSV logging (PROJECT_BRIEF "Data collection", non-negotiable):
// rows accumulate in memory and hit the filesystem only on flush() —
// wave end / game end / pause — so logging never perturbs tick timing.

#include <string>

struct TickRow {
    const char* run_id;
    const char* backend;
    const char* model_variant;
    const char* precision;
    const char* batch_mode;      // "resize" | "padmax"
    int wave;
    int batch_size;              // inference batch (constant per wave)
    int alive;                   // alive enemies this tick (<= batch_size)
    int tick_index;
    int warmup;                  // 1 = inside warmup window, excluded from steady stats
    double tick_total_ms;
    double feat_ms;
    double policy_infer_ms;
    double sim_ms;
    double render_ms;            // CPU-side render encode of the previous frame
    double gpu_render_ms;        // GL_EXT_disjoint_timer_query, -1 if unavailable
    int glow_on;
    int deadline_missed;
    double fps;
    int cpu_threads;
    double battery_temp_c;       // -1 if unknown
    double thermal_headroom;     // AThermal_getThermalHeadroom, -1 if unknown
    int thermal_status;          // AThermal status enum, -1 if unknown
    const char* note;
};

struct GestureRow {
    const char* run_id;
    const char* backend;
    const char* precision;
    const char* gesture_class;   // predicted label
    double confidence;
    int stroke_points;           // raw points before resampling
    double resample_ms;
    double infer_ms;
    double finger_up_to_effect_ms;
    int accepted;                // 0 if below confidence / on cooldown
    const char* note;
};

struct ColdRow {
    const char* run_id;
    const char* phase;           // "cold" | "bench"
    const char* backend;
    const char* model;
    const char* precision;
    int batch;
    double load_model_ms;
    double delegate_create_ms;
    double interpreter_create_ms;
    double allocate_ms;
    double first_infer_ms;
    double second_infer_ms;
    int steady_iters;
    double steady_mean_ms;
    double steady_std_ms;
    double steady_p50_ms;
    double steady_p95_ms;
    double steady_p99_ms;
    const char* nnapi_devices;
    const char* note;
};

class CsvLogger {
public:
    // dir = app external files dir (adb pull-able without permissions)
    void configure(const std::string& dir);
    // starts a new run: resets buffers, arms the three per-run files
    void startRun(const std::string& runId);
    const std::string& runId() const { return runId_; }
    const std::string& dir() const { return dir_; }

    void add(const TickRow& r);
    void add(const GestureRow& r);
    void add(const ColdRow& r);

    void flush();   // append buffers to disk, clear

private:
    void flushOne(const std::string& path, std::string& buf, const char* header, bool& headerDone);

    std::string dir_, runId_;
    std::string ticks_, gestures_, cold_;
    bool ticksHeader_ = false, gesturesHeader_ = false, coldHeader_ = false;
};
