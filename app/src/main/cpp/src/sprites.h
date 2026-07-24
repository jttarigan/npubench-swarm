#pragma once
// Procedural pixel-art atlas: every sprite + the 5x7 font baked into one
// 128x128 RGBA texture at startup. No art assets in the APK, fully reproducible.

#include <cstdint>
#include <vector>

constexpr int kAtlasSize = 128;

enum SpriteId {
    SPR_WHITE = 0,   // 4x4 solid white (rects, lines, trails)
    SPR_SHIP,        // 16x16 player ship, nose up
    SPR_ENEMY_A,     // 12x12 magenta bug
    SPR_ENEMY_B,     // 12x12 violet manta
    SPR_ENEMY_C,     // 12x12 pink spinner
    SPR_BULLET,      // 6x6 glowing dot
    SPR_ORB,         // 16x16 radial glow (shield)
    SPR_STAR,        // 3x3 background star
    SPR_ZONE,        // 32x32 damage-zone field: 75% alpha centre -> 30% at rim
    SPR_COUNT
};

struct AtlasRegion {
    int x = 0, y = 0, w = 0, h = 0;
};

// Fills a kAtlasSize^2 RGBA buffer. Font glyphs (ASCII 32..95) are packed in
// 8x8 cells, 16 per row, starting at (0, kFontAtlasY).
constexpr int kFontAtlasY = 64;
std::vector<uint8_t> buildAtlas(AtlasRegion regions[SPR_COUNT]);
