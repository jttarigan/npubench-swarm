#pragma once
// Orchestrator: app states, start-screen config, the measured fixed-timestep
// loop, debug HUD, cold-start bench, CSV wiring, thermal sampling.

#include <android/asset_manager.h>
#include <android/thermal.h>

#include <atomic>
#include <string>
#include <vector>

#include "common.h"
#include "csvlog.h"
#include "game.h"
#include "gesture.h"
#include "input.h"
#include "nn.h"
#include "prevention.h"
#include "renderer.h"
#include "trace.h"

struct ANativeWindow;
struct GameActivityMotionEvent;

enum class AppState { MENU, LOADING, PLAYING, BENCH, GAMEOVER, VICTORY, ERRORS, REPLAY };

class Engine {
public:
    void onWindowInit(ANativeWindow* window, AAssetManager* am, const char* filesDir);
    void onWindowLost();
    void onPause();
    void setFocused(bool f) { focused_ = f; }
    bool animating() const { return focused_ && renderer_.ready(); }

    void handleMotionEvent(const GameActivityMotionEvent& ev);
    void frame();

    static std::atomic<float> batteryTemp;   // pushed from the Java shim

private:
    // --- flow ---
    void startGame();
    void finishLoading();
    void beginWave(int idx);
    void doTick();
    void handleStroke(const Stroke& s);
    void enterError(const std::string& msg);
    void endRun(AppState finalState);
    void maybeWriteColdRows(bool force);
    void maybeWriteColdRowsForFailure();

    void flushAll();   // CSV + traces, only at the safe flush points

    // --- bench ---
    void startBench();
    void benchStep();

    // --- fixed-input replay (determinism study) ---
    void startReplay();
    bool setupPrevention();
    void logPreventionSummary();
    void replayStep();
    void renderReplay();

    // --- ui ---
    void renderMenu();
    void renderLoading();
    void renderPlayingOverlay();
    void renderEndScreen(const char* title);
    void renderBench();
    void renderErrors();
    void menuButton(int group, int index, float x, float y, float w, float h,
                    const char* label, bool selected);
    void handleMenuTap(Vec2 p);

    void sampleThermal();
    std::string policyAssetPath() const;
    std::string gestureAssetPath() const;
    std::string makeRunId(const char* kind) const;

    // --- members ---
    Renderer renderer_;
    InputState input_;
    Game game_;
    CsvLogger csv_;
    NnModel policy_, gesture_;
    AAssetManager* assets_ = nullptr;
    bool focused_ = false;

    AppState state_ = AppState::MENU;

    // start-screen selections
    Backend backend_ = Backend::CPU;
    int variant_ = 2;                 // default: medium
    Precision prec_ = Precision::F32;
    bool glow_ = true;
    bool autoReplay_ = false;         // LOOP toggle: restart run on game-over/victory
    bool traceOn_ = true;             // TRACE toggle: write run_<id>_trace.bin
    int replayCountdown_ = 0;         // frames until auto-restart (0 = inactive)

    // study 3 experiment C: on-device prevention (debug.npubench.prevent, see prevention.h)
    Prevention prevent_;
    PortablePolicy* portable_ = nullptr;
    std::string portableAsset_;
    long preventRowsTotal_ = 0, preventRowsSeen_ = 0;
    double preventMsTotal_ = 0;
    int lastPreventRows_ = 0;
    double lastPreventMs_ = 0;

    // run state
    int loadingStep_ = 0;
    int tickIndex_ = 0;
    int warmupLeft_ = 0;
    int64_t lastFrameNs_ = 0;
    int64_t accumulatorNs_ = 0;
    double waveSetupMs_ = 0;
    std::string batchHow_ = "resize";
    const char* pendingTickNote_ = "";
    float gestureCooldown_ = 0.f;
    bool coldPolicyPending_ = false, coldGesturePending_ = false;

    // per-frame render measurements
    double renderCpuMs_ = 0, renderGpuMs_ = -1;
    double fps_ = 0;
    int fpsFrames_ = 0;
    int64_t fpsWindowStartNs_ = 0;

    // rolling HUD stats
    double rollBuf_[120] = {0};
    int rollN_ = 0, rollPos_ = 0;
    double rollP95_ = 0;
    int missCount_ = 0;
    double sumPolicyMs_ = 0;
    int cntPolicy_ = 0;

    // thermal
    AThermalManager* thermal_ = nullptr;
    float thermalHeadroom_ = -1.f;
    int thermalStatus_ = -1;
    int thermalPollCounter_ = 0;

    // inference buffers (max wave 80, padded margin)
    float featBuf_[96 * kFeatureDim] = {0};
    float outBuf_[96 * kPolicyOut] = {0};

    // menu hit boxes
    struct Btn { float x, y, w, h; int group, index; };
    std::vector<Btn> btns_;

    // bench
    int benchIdx_ = 0;
    std::vector<std::string> benchLines_;

    // determinism traces
    std::string filesDir_;
    TraceWriter trace_;
    std::vector<uint8_t> stateBuf_;
    TraceReader replayIn_;
    TraceWriter replayOut_;
    size_t replayPos_ = 0;
    bool replayDone_ = false;
    std::string replaySrc_;

    std::string errorMsg_;
    std::string runId_;
};
