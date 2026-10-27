#include "csvlog.h"

#include <cstdio>
#include <cstring>

#include "common.h"

namespace {
constexpr const char* kTicksHeader =
    "run_id,backend,model_variant,precision,batch_mode,wave,batch_size,alive,tick_index,warmup,"
    "tick_total_ms,feat_ms,policy_infer_ms,sim_ms,render_ms,gpu_render_ms,glow_on,"
    "deadline_missed,fps,cpu_threads,battery_temp_c,thermal_headroom,thermal_status,"
    "out_hash,state_hash,prevent,prevent_rows,prevent_ms,note\n";
constexpr const char* kReplayHeader =
    "run_id,backend,model_variant,precision,source,tick_index,wave,batch_size,alive,"
    "policy_infer_ms,out_hash\n";
constexpr const char* kGesturesHeader =
    "run_id,backend,precision,gesture_class,confidence,stroke_points,resample_ms,infer_ms,"
    "finger_up_to_effect_ms,accepted,note\n";
constexpr const char* kColdHeader =
    "run_id,phase,backend,model,precision,batch,load_model_ms,delegate_create_ms,"
    "interpreter_create_ms,allocate_ms,first_infer_ms,second_infer_ms,steady_iters,"
    "steady_mean_ms,steady_std_ms,steady_p50_ms,steady_p95_ms,steady_p99_ms,"
    "nnapi_devices,note\n";

// CSV-sanitize free text fields (notes, device lists)
std::string safe(const char* s) {
    std::string out = s ? s : "";
    for (char& c : out) {
        if (c == ',' ) c = ';';
        if (c == '\n' || c == '\r') c = ' ';
    }
    return out;
}
}  // namespace

void CsvLogger::configure(const std::string& dir) { dir_ = dir; }

void CsvLogger::startRun(const std::string& runId) {
    flush();  // don't lose a previous run's tail
    runId_ = runId;
    ticks_.clear();     ticks_.reserve(1 << 20);
    gestures_.clear();  gestures_.reserve(1 << 14);
    cold_.clear();      cold_.reserve(1 << 14);
    replay_.clear();
    ticksHeader_ = gesturesHeader_ = coldHeader_ = replayHeader_ = false;
    LOGI("CSV run started: %s (dir %s)", runId_.c_str(), dir_.c_str());
}

void CsvLogger::add(const TickRow& r) {
    char buf[512];
    snprintf(buf, sizeof(buf),
             "%s,%s,%s,%s,%s,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%.2f,%d,%.1f,%.4f,%d,"
             "%016llx,%016llx,%s,%d,%.4f,%s\n",
             r.run_id, r.backend, r.model_variant, r.precision, r.batch_mode, r.wave,
             r.batch_size, r.alive, r.tick_index, r.warmup, r.tick_total_ms, r.feat_ms,
             r.policy_infer_ms, r.sim_ms, r.render_ms, r.gpu_render_ms, r.glow_on,
             r.deadline_missed, r.fps, r.cpu_threads, r.battery_temp_c, r.thermal_headroom,
             r.thermal_status, (unsigned long long)r.out_hash,
             (unsigned long long)r.state_hash, safe(r.prevent).c_str(), r.prevent_rows,
             r.prevent_ms, safe(r.note).c_str());
    ticks_ += buf;
}

void CsvLogger::add(const GestureRow& r) {
    char buf[384];
    snprintf(buf, sizeof(buf), "%s,%s,%s,%s,%.4f,%d,%.4f,%.4f,%.4f,%d,%s\n",
             r.run_id, r.backend, r.precision, r.gesture_class, r.confidence,
             r.stroke_points, r.resample_ms, r.infer_ms, r.finger_up_to_effect_ms,
             r.accepted, safe(r.note).c_str());
    gestures_ += buf;
}

void CsvLogger::add(const ColdRow& r) {
    char buf[768];
    snprintf(buf, sizeof(buf),
             "%s,%s,%s,%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%s,%s\n",
             r.run_id, r.phase, r.backend, r.model, r.precision, r.batch, r.load_model_ms,
             r.delegate_create_ms, r.interpreter_create_ms, r.allocate_ms, r.first_infer_ms,
             r.second_infer_ms, r.steady_iters, r.steady_mean_ms, r.steady_std_ms,
             r.steady_p50_ms, r.steady_p95_ms, r.steady_p99_ms,
             safe(r.nnapi_devices).c_str(), safe(r.note).c_str());
    cold_ += buf;
}

void CsvLogger::add(const ReplayRow& r) {
    char buf[384];
    snprintf(buf, sizeof(buf), "%s,%s,%s,%s,%s,%d,%d,%d,%d,%.4f,%016llx\n", r.run_id,
             r.backend, r.model_variant, r.precision, safe(r.source).c_str(), r.tick_index,
             r.wave, r.batch_size, r.alive, r.policy_infer_ms,
             (unsigned long long)r.out_hash);
    replay_ += buf;
}

void CsvLogger::flushOne(const std::string& path, std::string& buf, const char* header,
                         bool& headerDone) {
    if (buf.empty()) return;
    FILE* f = fopen(path.c_str(), "ab");
    if (!f) {
        LOGE("CSV flush failed: %s", path.c_str());
        return;
    }
    if (!headerDone) {
        fwrite(header, 1, strlen(header), f);
        headerDone = true;
    }
    fwrite(buf.data(), 1, buf.size(), f);
    fclose(f);
    buf.clear();
}

void CsvLogger::flush() {
    if (dir_.empty() || runId_.empty()) return;
    flushOne(dir_ + "/run_" + runId_ + "_ticks.csv", ticks_, kTicksHeader, ticksHeader_);
    flushOne(dir_ + "/run_" + runId_ + "_gestures.csv", gestures_, kGesturesHeader,
             gesturesHeader_);
    flushOne(dir_ + "/run_" + runId_ + "_coldstart.csv", cold_, kColdHeader, coldHeader_);
    flushOne(dir_ + "/run_" + runId_ + "_replay.csv", replay_, kReplayHeader, replayHeader_);
}
