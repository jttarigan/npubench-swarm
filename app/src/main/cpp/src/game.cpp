#include "game.h"

#include "trace.h"

namespace {
constexpr float kDt = 1.f / (float)kTickHz;
constexpr float kEnemyMaxSpeed = 120.f;   // feature spec PF1 normalizes vel by this
constexpr float kEnemySpeedCap = 55.f;    // actual movement clamp: slow convergence
constexpr float kEnemyAccel = 260.f;
constexpr float kBulletSpeed = 85.f;   // slow: shield/bomb are the counterplay
constexpr float kPlayerBulletSpeed = 190.f;
constexpr float kAutoFirePeriod = 0.22f;
constexpr float kBeamHalfWidth = 10.f;
constexpr float kBeamDuration = 2.f;
constexpr float kBeamDps = 2.5f;          // enemy hp is 1.0 -> ~0.4 s in-beam kills
constexpr float kZoneDpsCenter = 3.0f;    // falls to 25% at the rim
constexpr float kZoneMinRadius = 30.f;
constexpr float kZoneMaxRadius = 90.f;
constexpr float kOuterMargin = 48.f;      // soft world bounds beyond the screen
constexpr int kPlayerMaxHp = 10;
}  // namespace

// sin(2*pi*p) from plain float arithmetic, so the policy's time feature does
// not depend on the device's libm (bionic versions differ across phones).
// p is reduced to [0,1) with floor (exact); odd Taylor series to x^11 on
// [-pi/2, pi/2]; max abs error 2.2e-7 vs double sin (float rounding).
float detSin2Pi(float p) {
    p -= std::floor(p);
    float q = p - 0.5f;                 // sin(2*pi*p) = -sin(2*pi*q)
    float x = 6.2831853f * q;           // [-pi, pi)
    if (x > 1.5707963f) x = 3.1415927f - x;
    else if (x < -1.5707963f) x = -3.1415927f - x;
    const float x2 = x * x;
    float s = -2.5052108e-08f;          // -1/11!
    s = s * x2 + 2.7557319e-06f;        //  1/9!
    s = s * x2 - 1.9841270e-04f;        // -1/7!
    s = s * x2 + 8.3333333e-03f;        //  1/5!
    s = s * x2 - 1.6666667e-01f;        // -1/3!
    s = s * x2 + 1.f;
    return -(x * s);
}

uint32_t Game::rnd() {
    uint32_t x = rngState_;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return rngState_ = x;
}

void Game::startRun(int w, int h) {
    W_ = w;
    H_ = h;
    hp_ = kPlayerMaxHp;   // stationary ship can't dodge; gestures are the defence
    score_ = 0;
    simT_ = 0.f;
    invulnT_ = shieldT_ = 0.f;
    rngState_ = 0x9E3779B9u;   // deterministic spawn patterns run-to-run
    ship_ = shipTarget_ = {W_ * 0.5f, H_ * 0.52f};   // fixed at screen centre
    autoFireCd_ = 0.f;
    enemies_.clear();
    bullets_.clear();
    playerBullets_.clear();
    zones_.clear();
    beams_.clear();
    float horizon = H_ * 0.40f;
    for (int i = 0; i < 48; i++) {
        stars_[i] = {rndf() * W_, rndf() * horizon};
        starPhase_[i] = rndf() * 6.283f;
    }
}

void Game::startWave(int waveIdx) {
    wave_ = waveIdx;
    waveT_ = 0.f;
    hp_ = kPlayerMaxHp;   // fresh ring every wave
    bullets_.clear();
    zones_.clear();
    beams_.clear();
    int n = kWaveTable[waveIdx];
    enemies_.clear();
    enemies_.reserve(n);
    for (int i = 0; i < n; i++) {
        Enemy e;
        for (int attempt = 0; attempt < 12; attempt++) {
            // spawn just outside a random screen edge; the policy walks them in
            float m = 12.f + rndf() * 30.f;
            switch (rnd() % 4) {
                case 0: e.pos = {rndf() * W_, -m}; break;
                case 1: e.pos = {rndf() * W_, H_ + m}; break;
                case 2: e.pos = {-m, rndf() * H_}; break;
                default: e.pos = {W_ + m, rndf() * H_}; break;
            }
            bool clear = true;
            for (const Enemy& o : enemies_)
                if ((o.pos - e.pos).len2() < 12.f * 12.f) { clear = false; break; }
            if (clear) break;
        }
        e.sprite = (int)(rnd() % 3);
        e.fireCd = 1.8f + rndf() * 2.2f;
        enemies_.push_back(e);
    }
}

bool Game::waveCleared() const {
    for (const Enemy& e : enemies_)
        if (e.alive) return false;
    return true;
}

void Game::setShipTarget(Vec2 t, bool active) {
    shipActive_ = active;
    if (active) shipTarget_ = t;
}

int Game::buildFeatures(float* feat) const {
    aliveIdx_.clear();
    const float W = (float)W_, H = (float)H_;
    const float tSin = detSin2Pi(simT_ / 3.f);   // PF1 feature 13: sin(2*pi*t/3s)
    int n = 0;
    for (int i = 0; i < (int)enemies_.size(); i++) {
        const Enemy& e = enemies_[i];
        if (!e.alive) continue;
        float* f = feat + n * kFeatureDim;
        f[0] = e.pos.x / W * 2.f - 1.f;
        f[1] = e.pos.y / H * 2.f - 1.f;
        f[2] = e.vel.x / kEnemyMaxSpeed;
        f[3] = e.vel.y / kEnemyMaxSpeed;
        f[4] = (ship_.x - e.pos.x) / W;
        f[5] = (ship_.y - e.pos.y) / H;
        f[6] = std::sqrt(f[4] * f[4] + f[5] * f[5]);
        // 3 nearest alive neighbours
        int b0 = -1, b1 = -1, b2 = -1;
        float d0 = 1e18f, d1 = 1e18f, d2 = 1e18f;
        for (int j = 0; j < (int)enemies_.size(); j++) {
            if (j == i || !enemies_[j].alive) continue;
            float d = (enemies_[j].pos - e.pos).len2();
            if (d < d0) { d2 = d1; b2 = b1; d1 = d0; b1 = b0; d0 = d; b0 = j; }
            else if (d < d1) { d2 = d1; b2 = b1; d1 = d; b1 = j; }
            else if (d < d2) { d2 = d; b2 = j; }
        }
        const int nb[3] = {b0, b1, b2};
        for (int k = 0; k < 3; k++) {
            if (nb[k] < 0) {
                f[7 + k * 2] = 0.f;
                f[8 + k * 2] = 0.f;
            } else {
                const Vec2 d = enemies_[nb[k]].pos - e.pos;
                f[7 + k * 2] = clampf((d.x / W) / 0.25f, -1.f, 1.f);
                f[8 + k * 2] = clampf((d.y / H) / 0.25f, -1.f, 1.f);
            }
        }
        f[13] = tSin;
        aliveIdx_.push_back(i);
        n++;
    }
    return n;
}

void Game::applyPolicy(const float* out, int n) {
    for (int k = 0; k < n; k++) {
        Enemy& e = enemies_[aliveIdx_[k]];
        const float sx = clampf(out[k * kPolicyOut + 0], -1.f, 1.f);
        const float sy = clampf(out[k * kPolicyOut + 1], -1.f, 1.f);
        const float fire = out[k * kPolicyOut + 2];
        e.vel += Vec2{sx, sy} * (kEnemyAccel * kDt);
        float sp = e.vel.len();
        if (sp > kEnemySpeedCap) e.vel = e.vel * (kEnemySpeedCap / sp);
        if (fire > kPolicyFireThreshold && e.fireCd <= 0.f) {
            Vec2 dir = (ship_ - e.pos).normalized();
            bullets_.push_back({e.pos, dir * kBulletSpeed});
            e.fireCd = 1.2f + rndf() * 1.4f;
        }
    }
}

void Game::simulate(float dt) {
    simT_ += dt;
    waveT_ += dt;
    if (invulnT_ > 0) invulnT_ -= dt;
    if (shieldT_ > 0) shieldT_ -= dt;

    // ship is fixed at screen centre; autofire at the nearest enemy
    autoFireCd_ -= dt;
    if (autoFireCd_ <= 0.f) {
        const Enemy* nearest = nullptr;
        float best = 1e18f;
        for (const Enemy& e : enemies_) {
            if (!e.alive) continue;
            float d = (e.pos - ship_).len2();
            if (d < best) { best = d; nearest = &e; }
        }
        if (nearest) {
            Vec2 dir = (nearest->pos - ship_).normalized();
            playerBullets_.push_back({ship_ + dir * 8.f, dir * kPlayerBulletSpeed});
            autoFireCd_ = kAutoFirePeriod;
        }
    }

    for (Enemy& e : enemies_) {
        if (!e.alive) continue;
        e.pos += e.vel * dt;
        if (e.pos.x < -kOuterMargin) { e.pos.x = -kOuterMargin; e.vel.x *= -0.5f; }
        if (e.pos.x > W_ + kOuterMargin) { e.pos.x = W_ + kOuterMargin; e.vel.x *= -0.5f; }
        if (e.pos.y < -kOuterMargin) { e.pos.y = -kOuterMargin; e.vel.y *= -0.5f; }
        if (e.pos.y > H_ + kOuterMargin) { e.pos.y = H_ + kOuterMargin; e.vel.y *= -0.5f; }
        if (e.fireCd > 0) e.fireCd -= dt;
        if (e.hitFlash > 0) e.hitFlash -= dt;

        // ram the ship
        if ((e.pos - ship_).len2() < 10.f * 10.f) {
            e.alive = false;
            score_++;
            if (shieldT_ <= 0.f && invulnT_ <= 0.f) {
                hp_--;
                invulnT_ = 1.2f;
            }
        }
    }

    for (Bullet& b : bullets_) {
        if (!b.alive) continue;
        b.pos += b.vel * dt;
        if (b.pos.x < -8 || b.pos.x > W_ + 8 || b.pos.y < -8 || b.pos.y > H_ + 8) {
            b.alive = false;
            continue;
        }
        if ((b.pos - ship_).len2() < 8.f * 8.f) {
            b.alive = false;
            if (shieldT_ <= 0.f && invulnT_ <= 0.f) {
                hp_--;
                invulnT_ = 1.2f;
            }
        }
    }

    // autofire bullets vs enemies
    for (Bullet& b : playerBullets_) {
        if (!b.alive) continue;
        b.pos += b.vel * dt;
        if (b.pos.x < -kOuterMargin || b.pos.x > W_ + kOuterMargin ||
            b.pos.y < -kOuterMargin || b.pos.y > H_ + kOuterMargin) {
            b.alive = false;
            continue;
        }
        for (Enemy& e : enemies_) {
            if (!e.alive) continue;
            if ((e.pos - b.pos).len2() < 7.f * 7.f) {
                damageEnemy(e, 1.f);
                b.alive = false;
                break;
            }
        }
    }

    // damage zones: expand quickly to drawn radius, DPS falls off toward the rim
    for (ZoneFx& z : zones_) {
        z.t += dt;
        float cur = z.radius * (0.4f + 0.6f * std::min(1.f, z.t / 0.6f));
        for (Enemy& e : enemies_) {
            if (!e.alive) continue;
            float d = (e.pos - z.pos).len();
            if (d < cur) {
                float f = 0.25f + 0.75f * (1.f - d / cur);   // 100% centre -> 25% rim
                damageEnemy(e, kZoneDpsCenter * f * dt);
            }
        }
    }

    // enemy projectiles caught in a zone dissolve after 1 s inside
    for (Bullet& b : bullets_) {
        if (!b.alive) continue;
        bool inside = false;
        for (const ZoneFx& z : zones_) {
            float cur = z.radius * (0.4f + 0.6f * std::min(1.f, z.t / 0.6f));
            if ((b.pos - z.pos).len2() < cur * cur) { inside = true; break; }
        }
        if (inside) {
            b.zoneT += dt;
            if (b.zoneT >= 1.f) b.alive = false;
        } else {
            b.zoneT = 0.f;
        }
    }

    // persistent beams: enemies take DPS, enemy projectiles die on contact
    for (BeamFx& fx : beams_) {
        fx.t += dt;
        Vec2 dir = (fx.b - fx.a).normalized();
        for (Enemy& e : enemies_) {
            if (!e.alive) continue;
            Vec2 d = e.pos - fx.a;
            float perp = std::fabs(d.x * dir.y - d.y * dir.x);
            if (perp < kBeamHalfWidth) damageEnemy(e, kBeamDps * dt);
        }
        for (Bullet& b : bullets_) {
            if (!b.alive) continue;
            Vec2 d = b.pos - fx.a;
            float perp = std::fabs(d.x * dir.y - d.y * dir.x);
            if (perp < kBeamHalfWidth) b.alive = false;
        }
    }

    std::erase_if(playerBullets_, [](const Bullet& b) { return !b.alive; });
    std::erase_if(bullets_, [](const Bullet& b) { return !b.alive; });
    std::erase_if(zones_, [](const ZoneFx& z) { return z.t > z.duration; });
    std::erase_if(beams_, [](const BeamFx& f) { return f.t > kBeamDuration; });
}

void Game::appendState(std::vector<uint8_t>& b) const {
    put<uint16_t>(b, (uint16_t)enemies_.size());
    for (const Enemy& e : enemies_) {
        put<float>(b, e.pos.x);
        put<float>(b, e.pos.y);
        put<uint8_t>(b, e.alive ? 1 : 0);
    }
    put<int16_t>(b, (int16_t)hp_);
    put<int32_t>(b, score_);
    put<uint16_t>(b, (uint16_t)bullets_.size());
    put<uint32_t>(b, rngState_);
}

void Game::damageEnemy(Enemy& e, float amount) {
    e.hp -= amount;
    e.hitFlash = 0.12f;
    if (e.hp <= 0.f && e.alive) {
        e.alive = false;
        score_++;
    }
}

void Game::castZone(Vec2 at, float radius) {
    ZoneFx z;
    z.pos = at;
    z.radius = clampf(radius, kZoneMinRadius, kZoneMaxRadius);
    // bigger drawn circle -> longer field (6..10 s)
    z.duration = 6.f + 4.f * (z.radius - kZoneMinRadius) / (kZoneMaxRadius - kZoneMinRadius);
    zones_.push_back(z);
}

void Game::castBeam(const std::vector<Vec2>& strokePts) {
    if (strokePts.size() < 2) return;
    Vec2 a = strokePts.front(), b = strokePts.back();
    Vec2 dir = (b - a).normalized();
    if (dir.len2() < 0.5f) return;
    Vec2 c = (a + b) * 0.5f;
    beams_.push_back({c - dir * 600.f, c + dir * 600.f, 0.f});
}

void Game::castShield() { shieldT_ = 4.f; }

void Game::render(Renderer& r) const {
    const float W = (float)W_, H = (float)H_;
    const float horizon = H * 0.40f;

    // perspective grid (dim — stays below the bloom bright-pass threshold)
    for (int i = 0; i <= 9; i++) {
        float f = i / 9.f + std::fmod(simT_ * 0.05f, 1.f / 9.f);
        if (f > 1.f) f -= 1.f;
        float y = horizon + f * f * (H - horizon);
        r.line({0, y}, {W, y}, 1.f, rgba(110, 55, 190, (uint8_t)(30 + 60 * f)), false);
    }
    for (int i = -6; i <= 6; i++) {
        float t = i / 6.f;
        r.line({W * 0.5f + t * W * 0.12f, horizon}, {W * 0.5f + t * W * 0.85f, H}, 1.f,
               rgba(110, 55, 190, 42), false);
    }
    // stars
    for (int i = 0; i < 48; i++) {
        float tw = 0.5f + 0.5f * std::sin(simT_ * 2.f + starPhase_[i]);
        r.sprite(SPR_STAR, stars_[i], 1.f, rgba(255, 255, 255, (uint8_t)(70 + 150 * tw)),
                 true);
    }

    // beams under everything else (pulse while active, fade at the end)
    for (const BeamFx& fx : beams_) {
        float fade = std::min(1.f, (kBeamDuration - fx.t) / 0.4f);
        float pulse = 0.75f + 0.25f * std::sin(simT_ * 18.f);
        r.line(fx.a, fx.b, 6.f, rgba(80, 255, 240, (uint8_t)(90 * fade * pulse)), true);
        r.line(fx.a, fx.b, 2.f, rgba(255, 255, 255, (uint8_t)(220 * fade * pulse)), true);
    }

    // damage zones: gradient field (75% centre -> 30% rim, baked in SPR_ZONE) + rim ring
    for (const ZoneFx& z : zones_) {
        float cur = z.radius * (0.4f + 0.6f * std::min(1.f, z.t / 0.6f));
        float fade = std::min(1.f, (z.duration - z.t) / 0.5f);
        float pulse = 0.92f + 0.08f * std::sin(simT_ * 10.f);
        r.sprite(SPR_ZONE, z.pos, cur * 2.f / 32.f,
                 rgba(255, 255, 255, (uint8_t)(255 * fade * pulse)), false);
        // rim: 20 short segments
        for (int i = 0; i < 20; i++) {
            float a0 = i * 6.2832f / 20.f, a1 = (i + 1) * 6.2832f / 20.f;
            r.line({z.pos.x + std::cos(a0) * cur, z.pos.y + std::sin(a0) * cur},
                   {z.pos.x + std::cos(a1) * cur, z.pos.y + std::sin(a1) * cur}, 1.f,
                   rgba(120, 255, 245, (uint8_t)(140 * fade)), true);
        }
    }

    for (const Enemy& e : enemies_) {
        if (!e.alive) continue;
        float rot = std::atan2(e.vel.y, e.vel.x) + 1.5708f;
        if (e.sprite == 2) rot = simT_ * 3.f;   // spinner spins
        uint32_t col = e.hitFlash > 0 ? rgba(255, 255, 255, 255) : rgba(255, 255, 255, 235);
        r.sprite((SpriteId)(SPR_ENEMY_A + e.sprite), e.pos, 1.f, col, true, rot);
    }

    for (const Bullet& b : bullets_)
        r.sprite(SPR_BULLET, b.pos, 1.f, rgba(255, 255, 255, 255), true);

    // autofire bullets tinted cyan to read differently from enemy shots
    for (const Bullet& b : playerBullets_)
        r.sprite(SPR_BULLET, b.pos, 1.f, rgba(120, 255, 240, 255), true);

    // ship (blinks while invulnerable)
    bool blink = invulnT_ > 0.f && std::fmod(simT_ * 10.f, 1.f) < 0.4f;
    if (!blink) r.sprite(SPR_SHIP, ship_, 1.f, rgba(255, 255, 255, 255), true);

    // circular health ring around the ship: arc length = hp fraction
    {
        const float ringR = 14.f;
        const int segs = 24;
        float frac = clampf((float)hp_ / (float)kPlayerMaxHp, 0.f, 1.f);
        int lit = (int)(frac * segs + 0.5f);
        for (int i = 0; i < lit; i++) {
            // start at 12 o'clock, clockwise
            float a0 = -1.5708f + i * 6.2832f / segs;
            float a1 = -1.5708f + (i + 1) * 6.2832f / segs;
            uint32_t col = frac > 0.5f ? rgba(80, 255, 240, 190)
                          : frac > 0.25f ? rgba(255, 210, 80, 200)
                                         : rgba(255, 60, 120, 220);
            r.line({ship_.x + std::cos(a0) * ringR, ship_.y + std::sin(a0) * ringR},
                   {ship_.x + std::cos(a1) * ringR, ship_.y + std::sin(a1) * ringR}, 1.f,
                   col, true);
        }
    }
    if (shieldT_ > 0.f) {
        float pulse = 0.6f + 0.4f * std::sin(simT_ * 8.f);
        float fade = shieldT_ < 1.f ? shieldT_ : 1.f;
        r.sprite(SPR_ORB, ship_, 2.6f, rgba(80, 255, 240, (uint8_t)(120 * pulse * fade)),
                 true);
    }
}
