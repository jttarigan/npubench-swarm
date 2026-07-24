#include "sprites.h"

#include <cmath>
#include <cstring>

#include "font5x7.h"

namespace {

struct Painter {
    std::vector<uint8_t>& buf;
    void set(int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
        if (x < 0 || y < 0 || x >= kAtlasSize || y >= kAtlasSize) return;
        uint8_t* p = buf.data() + (y * kAtlasSize + x) * 4;
        p[0] = r; p[1] = g; p[2] = b; p[3] = a;
    }
};

// Neon synthwave palette (emissive-leaning, bright-pass friendly)
struct Rgb { uint8_t r, g, b; };
constexpr Rgb kCyan     = {80, 255, 240};
constexpr Rgb kCyanHot  = {210, 255, 252};
constexpr Rgb kMagenta  = {255, 60, 220};
constexpr Rgb kViolet   = {160, 90, 255};
constexpr Rgb kPink     = {255, 105, 160};
constexpr Rgb kBlue     = {70, 130, 255};
constexpr Rgb kWhite    = {255, 255, 255};

// Paint an ASCII-art sprite. Map: '.'=transparent, everything else looked up.
void paintAscii(Painter& p, int ox, int oy, const char* const* rows, int h,
                const char* keys, const Rgb* colors, int nkeys) {
    for (int y = 0; y < h; y++) {
        const char* row = rows[y];
        for (int x = 0; row[x]; x++) {
            char c = row[x];
            if (c == '.') continue;
            for (int k = 0; k < nkeys; k++) {
                if (keys[k] == c) {
                    p.set(ox + x, oy + y, colors[k].r, colors[k].g, colors[k].b, 255);
                    break;
                }
            }
        }
    }
}

}  // namespace

std::vector<uint8_t> buildAtlas(AtlasRegion regions[SPR_COUNT]) {
    std::vector<uint8_t> buf(kAtlasSize * kAtlasSize * 4, 0);
    Painter p{buf};

    // --- WHITE 4x4 at (0,0) ---
    regions[SPR_WHITE] = {1, 1, 2, 2};  // sample interior to avoid edge bleed
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) p.set(x, y, 255, 255, 255, 255);

    // --- SHIP 16x16 at (8,0) ---
    {
        const char* rows[16] = {
            ".......ww.......",
            ".......CC.......",
            "......cCCc......",
            "......cCCc......",
            ".....ccCCcc.....",
            ".....cCbbCc.....",
            "....ccCbbCcc....",
            "....cCCbbCCc....",
            "...ccCCCCCCcc...",
            "...cCCcCCcCCc...",
            "..ccCccCCccCcc..",
            "..cCc.cCCc.cCc..",
            ".ccc..cCCc..ccc.",
            ".cc...mmmm...cc.",
            "......m..m......",
            "......w..w......",
        };
        const char keys[] = {'c', 'C', 'b', 'm', 'w'};
        const Rgb cols[] = {kCyan, kCyanHot, kBlue, kMagenta, kWhite};
        paintAscii(p, 8, 0, rows, 16, keys, cols, 5);
        regions[SPR_SHIP] = {8, 0, 16, 16};
    }

    // --- ENEMY_A 12x12 magenta bug at (28,0) ---
    {
        const char* rows[12] = {
            "..m......m..",
            "..mm....mm..",
            "...mmmmmm...",
            "..mmMMMMmm..",
            ".mmMwMMwMmm.",
            ".mMMMMMMMMm.",
            ".mMMMMMMMMm.",
            ".mmMMMMMMmm.",
            "..mmMMMMmm..",
            "...mmmmmm...",
            "..mm....mm..",
            "..m......m..",
        };
        const char keys[] = {'m', 'M', 'w'};
        const Rgb cols[] = {{170, 20, 150}, kMagenta, kWhite};
        paintAscii(p, 28, 0, rows, 12, keys, cols, 3);
        regions[SPR_ENEMY_A] = {28, 0, 12, 12};
    }

    // --- ENEMY_B 12x12 violet manta at (44,0) ---
    {
        const char* rows[12] = {
            "............",
            "v..........v",
            "vv........vv",
            "vVv......vVv",
            "vVVv....vVVv",
            "vVVVvvvvVVVv",
            ".vVVVwwVVVv.",
            "..vVVVVVVv..",
            "...vVVVVv...",
            "....vVVv....",
            ".....vv.....",
            "............",
        };
        const char keys[] = {'v', 'V', 'w'};
        const Rgb cols[] = {{100, 50, 180}, kViolet, kWhite};
        paintAscii(p, 44, 0, rows, 12, keys, cols, 3);
        regions[SPR_ENEMY_B] = {44, 0, 12, 12};
    }

    // --- ENEMY_C 12x12 pink spinner at (60,0) ---
    {
        const char* rows[12] = {
            ".....pp.....",
            "....pPPp....",
            "...pPPPPp...",
            "..pPPwwPPp..",
            ".pPPwWWwPPp.",
            "pPPwWWWWwPPp",
            "pPPwWWWWwPPp",
            ".pPPwWWwPPp.",
            "..pPPwwPPp..",
            "...pPPPPp...",
            "....pPPp....",
            ".....pp.....",
        };
        const char keys[] = {'p', 'P', 'w', 'W'};
        const Rgb cols[] = {{190, 60, 110}, kPink, {255, 190, 215}, kWhite};
        paintAscii(p, 60, 0, rows, 12, keys, cols, 4);
        regions[SPR_ENEMY_C] = {60, 0, 12, 12};
    }

    // --- BULLET 6x6 at (76,0): white core, magenta rim ---
    {
        for (int y = 0; y < 6; y++)
            for (int x = 0; x < 6; x++) {
                float dx = x - 2.5f, dy = y - 2.5f;
                float d = std::sqrt(dx * dx + dy * dy);
                if (d < 1.3f) p.set(76 + x, y, 255, 255, 255, 255);
                else if (d < 2.6f) p.set(76 + x, y, kMagenta.r, kMagenta.g, kMagenta.b, 230);
            }
        regions[SPR_BULLET] = {76, 0, 6, 6};
    }

    // --- ORB 16x16 at (84,0): cyan radial falloff ---
    {
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++) {
                float dx = x - 7.5f, dy = y - 7.5f;
                float d = std::sqrt(dx * dx + dy * dy) / 7.5f;
                if (d > 1.f) continue;
                float t = 1.f - d;
                uint8_t a = (uint8_t)(t * t * 255.f);
                p.set(84 + x, y, (uint8_t)(80 + 175 * t), 255, (uint8_t)(240 * (0.6f + 0.4f * t)), a);
            }
        regions[SPR_ORB] = {84, 0, 16, 16};
    }

    // --- STAR 3x3 at (104,0) ---
    {
        p.set(105, 1, 255, 255, 255, 255);
        p.set(104, 1, 180, 200, 255, 150);
        p.set(106, 1, 180, 200, 255, 150);
        p.set(105, 0, 180, 200, 255, 150);
        p.set(105, 2, 180, 200, 255, 150);
        regions[SPR_STAR] = {104, 0, 3, 3};
    }

    // --- ZONE 32x32 at (0,24): alpha gradient 75% centre -> 30% rim ---
    {
        for (int y = 0; y < 32; y++)
            for (int x = 0; x < 32; x++) {
                float dx = x - 15.5f, dy = y - 15.5f;
                float d = std::sqrt(dx * dx + dy * dy) / 15.5f;
                if (d > 1.f) continue;
                float t = 1.f - d;
                uint8_t a = (uint8_t)(255.f * (0.30f + 0.45f * t * std::sqrt(t)));
                p.set(x, 24 + y, (uint8_t)(80 + 130 * t), 255,
                      (uint8_t)(240 + 15 * t), a);
            }
        regions[SPR_ZONE] = {0, 24, 32, 32};
    }

    // --- FONT: ASCII 32..95 (+ '|' at index 64) in 8x8 cells, 16/row,
    //     starting at (0, kFontAtlasY) ---
    for (int i = 0; i < kFontGlyphCount; i++) {
        int cx = (i % 16) * 8;
        int cy = kFontAtlasY + (i / 16) * 8;
        for (int r = 0; r < kFontGlyphH; r++) {
            uint8_t bits = kFont5x7[i][r];
            for (int c = 0; c < kFontGlyphW; c++)
                if (bits & (1 << (kFontGlyphW - 1 - c)))
                    p.set(cx + c, cy + r, 255, 255, 255, 255);
        }
    }

    return buf;
}
