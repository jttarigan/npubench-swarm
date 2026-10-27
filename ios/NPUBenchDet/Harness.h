// Study 3 iOS harness: the game's own simulation (game.cpp) and trace writer
// (trace.cpp), unchanged, driven headless with the enemy-AI model run through
// LiteRT on the CPU (XNNPACK), the GPU (Metal delegate) or Core ML (Neural
// Engine). Mirrors Engine::doTick and Engine::replayStep of the Android app:
// features -> padded batch -> policy -> applyPolicy -> simulate -> trace.
#pragma once

#include <functional>
#include <string>

enum class IosBackend { CPU = 0, GPU = 1, NPU = 2 };

struct HarnessConfig {
    IosBackend backend = IosBackend::CPU;
    bool int8 = false;
    std::string variant = "xlarge";   // model file policy_<variant>_<f32|int8>.tflite
};

using LogFn = std::function<void(const std::string&)>;

// Free-running game in LOOP-equivalent autoplay (no player input). Writes
// run_<id>_game_<backend>_<variant>_<prec>_trace.bin and _ticks.csv to outDir.
// Returns false and logs the reason on failure.
bool runGame(const HarnessConfig& cfg, const std::string& modelDir, const std::string& outDir,
             const LogFn& log);

// Fixed-input replay of inputPath (an NPBT trace from any device). Writes
// run_<id>_replay_<backend>_<variant>_<prec>_replay.bin and _replay.csv.
bool runReplay(const HarnessConfig& cfg, const std::string& modelDir,
               const std::string& inputPath, const std::string& outDir, const LogFn& log);

std::string liteRtVersion();

// true: add the delegate at interpreter creation and resize afterwards (the
// Android app's order; corrupts host memory with the Metal delegate on iOS).
extern bool g_legacyDelegateOrder;
const char* backendTag(IosBackend b);
