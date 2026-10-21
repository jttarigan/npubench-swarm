// Host (macOS/Linux) build of the game simulation + trace writer, used to
// check the determinism tooling without a device. NOT the measured
// instrument: no LiteRT, a scripted policy stands in for the model, and the
// host compiler is not the NDK. Mirrors Engine::doTick's order exactly:
// features -> policy -> applyPolicy -> simulate -> fingerprints -> wave step.
//
//   hostsim <out_prefix> [perturb_tick] [ulp|flip|eps:<v>]
//     ulp    : at perturb_tick, steer_x of row 0 moves by 1 ulp
//     flip   : at perturb_tick, row 0's fire decision is inverted
//     eps:<v>: at perturb_tick, <v> is added to steer_x and steer_y of every row
//   hostsim --sin-test
//   hostsim <out_prefix> --pipe
//   HOSTSIM_SEED=N hostsim ...   spawn seed N instead of the game's fixed one
//     co-simulation: the policy outputs come from the parent process. Per
//     tick hostsim writes to stdout  int32 tick, wave(1-based), batch, alive
//     then alive*kFeatureDim floats (little-endian), and reads alive*kPolicyOut
//     floats from stdin. A final header with tick = -1 ends the run. Lets a
//     Python driver run the real model in any numeric backend
//     (analysis/cosim.py) while the game itself is this exact C++ code.
// Writes <out_prefix>_trace.bin (NPBT) and <out_prefix>_ticks.csv.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "game.h"
#include "trace.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846   // MSVC's <cmath> defines it only with _USE_MATH_DEFINES
#endif

// game.cpp's render() references these; the host never draws.
void Renderer::sprite(SpriteId, Vec2, float, uint32_t, bool, float) {}
void Renderer::rect(float, float, float, float, uint32_t, bool) {}
void Renderer::line(Vec2, Vec2, float, uint32_t, bool) {}
void Renderer::text(int, int, const char*, uint32_t, int) {}
int Renderer::textWidth(const char*, int) const { return 0; }
void Renderer::textCentered(int, int, const char*, uint32_t, int) {}

namespace {

// Scripted stand-in policy: steer at the ship, fire when close.
void scriptedPolicy(const float* feat, int n, float* out) {
    for (int k = 0; k < n; k++) {
        const float* f = feat + k * kFeatureDim;
        Vec2 d = Vec2{f[4], f[5]}.normalized();
        out[k * kPolicyOut + 0] = d.x;
        out[k * kPolicyOut + 1] = d.y;
        out[k * kPolicyOut + 2] = f[6] < 0.3f ? 0.9f : -0.9f;
    }
}

// --pipe: exchange one tick with the driver; returns false on a broken pipe
bool pipePolicy(int tick, int wave, int batch, int alive, const float* feat, float* out) {
    int32_t hdr[4] = {tick, wave, batch, alive};
    if (fwrite(hdr, sizeof(hdr), 1, stdout) != 1) return false;
    if (alive > 0 && fwrite(feat, sizeof(float) * kFeatureDim, alive, stdout) != (size_t)alive)
        return false;
    fflush(stdout);
    if (alive > 0 && fread(out, sizeof(float) * kPolicyOut, alive, stdin) != (size_t)alive)
        return false;
    return true;
}

int sinTest() {
    double worst = 0;
    float worstP = 0;
    for (int i = 0; i <= 2000000; i++) {
        float p = (float)i / 100000.f;   // 0..20 periods
        double ref = std::sin(2.0 * M_PI * (double)(p - std::floor(p)));
        double err = std::fabs((double)detSin2Pi(p) - ref);
        if (err > worst) { worst = err; worstP = p; }
    }
    printf("detSin2Pi max abs error %.3g at p=%.6f\n", worst, worstP);
    return worst < 5e-7 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);   // the pipe protocol is binary
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (argc >= 2 && strcmp(argv[1], "--sin-test") == 0) return sinTest();
    if (argc < 2) {
        fprintf(stderr, "usage: hostsim <out_prefix> [perturb_tick] [ulp|flip]\n");
        return 2;
    }
    const std::string prefix = argv[1];
    const bool pipe = argc >= 3 && strcmp(argv[2], "--pipe") == 0;
    const int perturbTick = argc >= 3 && !pipe ? atoi(argv[2]) : -1;
    const bool flip = argc >= 4 && strcmp(argv[3], "flip") == 0;
    const float eps = argc >= 4 && strncmp(argv[3], "eps:", 4) == 0 ? (float)atof(argv[3] + 4) : 0.f;

    remove((prefix + "_trace.bin").c_str());
    TraceWriter trace;
    trace.start(prefix + "_trace.bin", "NPBT");
    FILE* csv = fopen((prefix + "_ticks.csv").c_str(), "w");
    fprintf(csv, "run_id,tick_index,wave,batch_size,alive,out_hash,state_hash,note\n");

    static float feat[96 * kFeatureDim], out[96 * kPolicyOut];
    std::vector<uint8_t> state;
    Game g;
    g.startRun(kInternalW, kInternalH);
    if (const char* seed = getenv("HOSTSIM_SEED")) g.setSeed((uint32_t)strtoul(seed, nullptr, 0));
    g.startWave(0);
    const char* endNote = "tick_limit";
    int tick = 0;
    for (; tick < 20000; tick++) {
        const int alive = g.buildFeatures(feat);
        const int batch = kWaveTable[g.waveIndex()];
        memset(feat + alive * kFeatureDim, 0, sizeof(float) * (size_t)(batch - alive) * kFeatureDim);
        if (pipe) {
            if (!pipePolicy(tick, g.waveIndex() + 1, batch, alive, feat, out)) {
                fprintf(stderr, "hostsim: pipe closed at tick %d\n", tick);
                return 3;
            }
        } else {
            scriptedPolicy(feat, alive, out);
        }
        if (tick == perturbTick && alive > 0) {
            if (flip) {
                out[2] = -out[2];
            } else if (eps != 0.f) {
                for (int k = 0; k < alive; k++) {
                    out[k * kPolicyOut + 0] += eps;
                    out[k * kPolicyOut + 1] += eps;
                }
            } else {
                out[0] = std::nextafter(out[0], 2.f);
            }
        }
        if (alive > 0) g.applyPolicy(out, alive);
        g.simulate(1.f / (float)kTickHz);

        state.clear();
        g.appendState(state);
        const uint64_t oh = fnv1a64(out, sizeof(float) * (size_t)alive * kPolicyOut);
        const uint64_t sh = fnv1a64(state.data(), state.size());
        trace.addTick(tick, g.waveIndex() + 1, batch, alive, feat, out, g.aliveIndices(), state);
        fprintf(csv, "host,%d,%d,%d,%d,%016llx,%016llx,\n", tick, g.waveIndex() + 1, batch,
                alive, (unsigned long long)oh, (unsigned long long)sh);

        if (g.playerDead()) { endNote = "game_over"; break; }
        if (g.waveCleared() || g.waveTimedOut()) {
            trace.flush();
            if (g.waveIndex() + 1 >= kNumWaves) { endNote = "victory"; break; }
            g.startWave(g.waveIndex() + 1);
        }
    }
    trace.flush();
    fclose(csv);
    if (pipe) {
        int32_t hdr[4] = {-1, g.waveIndex() + 1, g.score(), g.hp()};
        fwrite(hdr, sizeof(hdr), 1, stdout);
        fflush(stdout);
        fprintf(stderr, "%s: %d ticks, wave %d, score %d, hp %d, %s\n", prefix.c_str(), tick + 1,
                g.waveIndex() + 1, g.score(), g.hp(), endNote);
        return 0;
    }
    printf("%s: %d ticks, wave %d, score %d, hp %d, %s\n", prefix.c_str(), tick + 1,
           g.waveIndex() + 1, g.score(), g.hp(), endNote);
    return 0;
}
