#include "trace.h"

#include <cstdio>
#include <cstring>

#include "common.h"

uint64_t fnv1a64(const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

void TraceWriter::start(const std::string& path, const char magic[4]) {
    path_ = path;
    buf_.clear();
    buf_.reserve(1 << 22);
    headerDone_ = false;
    memcpy(magic_, magic, 4);
}

void TraceWriter::putTickHeader(int tick, int wave, int batch, int alive) {
    put<uint32_t>(buf_, (uint32_t)tick);
    put<uint16_t>(buf_, (uint16_t)wave);
    put<uint16_t>(buf_, (uint16_t)batch);
    put<uint16_t>(buf_, (uint16_t)alive);
    put<uint16_t>(buf_, 0);
}

void TraceWriter::addTick(int tick, int wave, int batch, int alive, const float* feat,
                          const float* out, const std::vector<int>& enemyIdx,
                          const std::vector<uint8_t>& stateBlock) {
    if (path_.empty()) return;
    putTickHeader(tick, wave, batch, alive);
    putBytes(buf_, feat, sizeof(float) * (size_t)alive * kFeatureDim);
    putBytes(buf_, out, sizeof(float) * (size_t)alive * kPolicyOut);
    for (int k = 0; k < alive; k++) put<uint16_t>(buf_, (uint16_t)enemyIdx[k]);
    putBytes(buf_, stateBlock.data(), stateBlock.size());
}

void TraceWriter::addReplayTick(int tick, int wave, int batch, int alive, const float* out) {
    if (path_.empty()) return;
    putTickHeader(tick, wave, batch, alive);
    putBytes(buf_, out, sizeof(float) * (size_t)alive * kPolicyOut);
}

void TraceWriter::flush() {
    if (path_.empty() || buf_.empty()) return;
    FILE* f = fopen(path_.c_str(), "ab");
    if (!f) {
        LOGE("trace flush failed: %s", path_.c_str());
        return;
    }
    if (!headerDone_) {
        fwrite(magic_, 1, 4, f);
        const uint32_t hdr[3] = {1, (uint32_t)kFeatureDim, (uint32_t)kPolicyOut};
        fwrite(hdr, sizeof(uint32_t), 3, f);
        headerDone_ = true;
    }
    fwrite(buf_.data(), 1, buf_.size(), f);
    fclose(f);
    buf_.clear();
}

namespace {
template <typename T>
bool take(const std::vector<uint8_t>& d, size_t& off, T* v) {
    if (off + sizeof(T) > d.size()) return false;
    memcpy(v, d.data() + off, sizeof(T));
    off += sizeof(T);
    return true;
}
}  // namespace

bool TraceReader::load(const std::string& path) {
    data_.clear();
    ticks_.clear();
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        err_ = "cannot open " + path;
        return false;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    data_.resize(len > 0 ? (size_t)len : 0);
    size_t got = data_.empty() ? 0 : fread(data_.data(), 1, data_.size(), f);
    fclose(f);
    if (got != data_.size() || data_.size() < 16 || memcmp(data_.data(), "NPBT", 4) != 0) {
        err_ = "not an NPBT trace: " + path;
        return false;
    }
    fileHash_ = fnv1a64(data_.data(), data_.size());
    uint32_t ver, fd, od;
    size_t off = 4;
    take(data_, off, &ver);
    take(data_, off, &fd);
    take(data_, off, &od);
    if (ver != 1 || fd != (uint32_t)kFeatureDim || od != (uint32_t)kPolicyOut) {
        err_ = "trace version/dims mismatch";
        return false;
    }
    while (off < data_.size()) {
        uint32_t tick;
        uint16_t wave, batch, alive, rsv, E;
        if (!take(data_, off, &tick) || !take(data_, off, &wave) || !take(data_, off, &batch) ||
            !take(data_, off, &alive) || !take(data_, off, &rsv)) {
            err_ = "truncated tick header";
            return false;
        }
        TraceTick t{(int)tick, wave, batch, alive,
                    reinterpret_cast<const float*>(data_.data() + off)};
        // skip feat, out, enemyIdx, then the variable-size state block
        off += (size_t)alive * (kFeatureDim + kPolicyOut) * sizeof(float) +
               (size_t)alive * sizeof(uint16_t);
        if (!take(data_, off, &E)) {
            err_ = "truncated tick body";
            return false;
        }
        off += (size_t)E * 9 + 2 + 4 + 2 + 4;
        if (off > data_.size()) {
            err_ = "truncated state block";
            return false;
        }
        ticks_.push_back(t);
    }
    return !ticks_.empty();
}
