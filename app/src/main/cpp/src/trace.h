#pragma once
// Determinism traces (study 3). Little-endian binary, buffered in memory and
// appended to disk only at the CSV flush points, like the CSV logger.
//
// Trace file  run_<id>_trace.bin  (free-running game):
//   header  "NPBT" u32 version=1 u32 featDim u32 outDim
//   per tick:
//     u32 tick  u16 wave(1-based)  u16 batch  u16 alive  u16 reserved
//     f32 feat[alive*featDim]   policy inputs, alive rows only
//     f32 out[alive*outDim]     policy outputs, alive rows only
//     u16 enemyIdx[alive]       row -> enemy slot
//     state block (Game::appendState, after this tick's simulate):
//       u16 E; E x (f32 x, f32 y, u8 alive); i16 hp; i32 score;
//       u16 enemyBullets; u32 rngState
//
// Replay file  run_<id>_replay.bin  (fixed-input replay of a trace):
//   header  "NPBR" u32 version=1 u32 featDim u32 outDim
//   per tick: u32 tick u16 wave u16 batch u16 alive u16 reserved
//             f32 out[alive*outDim]
//
// The tick CSV carries FNV-1a 64 hashes of out[] (alive rows) and of the
// state block, so two runs can be compared without the binary files.

#include <cstdint>
#include <string>
#include <vector>

uint64_t fnv1a64(const void* data, size_t len);

inline void putBytes(std::vector<uint8_t>& b, const void* p, size_t n) {
    const uint8_t* s = static_cast<const uint8_t*>(p);
    b.insert(b.end(), s, s + n);
}
template <typename T>
inline void put(std::vector<uint8_t>& b, T v) { putBytes(b, &v, sizeof(T)); }

class TraceWriter {
public:
    // magic "NPBT" or "NPBR"; empty path = disabled
    void start(const std::string& path, const char magic[4]);
    bool active() const { return !path_.empty(); }
    void stop() { path_.clear(); buf_.clear(); }

    void addTick(int tick, int wave, int batch, int alive, const float* feat,
                 const float* out, const std::vector<int>& enemyIdx,
                 const std::vector<uint8_t>& stateBlock);
    void addReplayTick(int tick, int wave, int batch, int alive, const float* out);

    void flush();

private:
    void putTickHeader(int tick, int wave, int batch, int alive);
    std::string path_;
    std::vector<uint8_t> buf_;
    bool headerDone_ = false;
    char magic_[4] = {0, 0, 0, 0};
};

// One tick of a trace file, as needed to replay it.
struct TraceTick {
    int tick, wave, batch, alive;
    const float* feat;   // points into the reader's buffer; may be unaligned
};

class TraceReader {
public:
    // Loads and indexes a whole "NPBT" file. Returns false with err() set.
    bool load(const std::string& path);
    const std::vector<TraceTick>& ticks() const { return ticks_; }
    const std::string& err() const { return err_; }
    uint64_t fileHash() const { return fileHash_; }   // FNV-1a 64 of the whole file

private:
    std::vector<uint8_t> data_;
    uint64_t fileHash_ = 0;
    std::vector<TraceTick> ticks_;
    std::string err_;
};
