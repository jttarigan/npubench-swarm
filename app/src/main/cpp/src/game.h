#pragma once
// World simulation. Deterministic where possible (fixed-seed RNG for spawns).
// Enemy behaviour comes exclusively from the policy net — buildFeatures() and
// applyPolicy() share a compaction order so batched rows line up.

#include <vector>

#include "common.h"
#include "renderer.h"

struct Enemy {
    Vec2 pos, vel;
    int sprite = 0;        // 0..2 -> SPR_ENEMY_A/B/C
    float hp = 1.f;
    float fireCd = 1.f;
    float hitFlash = 0.f;
    bool alive = true;
};

struct Bullet {
    Vec2 pos, vel;
    float zoneT = 0.f;   // cumulative time inside a damage zone (1 s dissolves it)
    bool alive = true;
};

// expanding damage field: full DPS at centre, 25% at the rim
struct ZoneFx {
    Vec2 pos;
    float radius = 40.f;   // final radius (from the drawn circle)
    float duration = 4.f;  // 4..6 s by drawn size
    float t = 0.f;
};
// persistent beam: damages intersecting enemies for its lifetime
struct BeamFx { Vec2 a, b; float t = 0.f; };

class Game {
public:
    void startRun(int w, int h);
    void startWave(int waveIdx);           // spawns kWaveTable[waveIdx] enemies
    int waveIndex() const { return wave_; }
    int waveSize() const { return kWaveTable[wave_]; }
    bool waveCleared() const;
    bool waveTimedOut() const { return waveT_ > kWaveTimeoutS; }
    bool playerDead() const { return hp_ <= 0; }
    int hp() const { return hp_; }
    int score() const { return score_; }
    float simTime() const { return simT_; }

    void setShipTarget(Vec2 t, bool active);

    // returns count of alive enemies; writes [n x kFeatureDim] into feat
    int buildFeatures(float* feat) const;
    void applyPolicy(const float* out, int n);   // consumes [n x kPolicyOut]
    void simulate(float dt);

    // gesture abilities (applied immediately; effect timestamp taken by caller)
    void castZone(Vec2 at, float radius);
    void castBeam(const std::vector<Vec2>& strokePts);
    void castShield();

    void render(Renderer& r) const;              // world phase only

private:
    uint32_t rnd();                              // xorshift32, seeded per run
    float rndf() { return (rnd() & 0xFFFFFF) / 16777216.f; }

    int W_ = 240, H_ = 480;
    int wave_ = 0;
    float waveT_ = 0.f, simT_ = 0.f;
    int hp_ = 3, score_ = 0;
    float invulnT_ = 0.f, shieldT_ = 0.f;

    Vec2 ship_{120, 400}, shipTarget_{120, 400};
    bool shipActive_ = false;

    void damageEnemy(Enemy& e, float amount);

    std::vector<Enemy> enemies_;
    std::vector<Bullet> bullets_;        // enemy shots
    std::vector<Bullet> playerBullets_;  // autofire
    float autoFireCd_ = 0.f;
    std::vector<ZoneFx> zones_;
    std::vector<BeamFx> beams_;
    mutable std::vector<int> aliveIdx_;          // buildFeatures -> applyPolicy mapping

    uint32_t rngState_ = 0x9E3779B9u;
    Vec2 stars_[48];
    float starPhase_[48] = {0};
};
