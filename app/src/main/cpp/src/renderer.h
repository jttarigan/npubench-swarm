#pragma once
// GLES 3.0 low-res neon pipeline (PROJECT_BRIEF "Tech stack"):
//   emissive sprites -> internal FBO (240 x H) -> bright pass -> separable blur
//   at half res -> additive composite back at internal res -> overlay (HUD) ->
//   ONE nearest-neighbour integer upscale to screen.
// Bloom cost is resolution-fixed: render time stays flat as swarm size grows.

#include <EGL/egl.h>
#include <GLES3/gl3.h>

#include <cstdint>
#include <vector>

#include "common.h"
#include "sprites.h"

struct ANativeWindow;

inline uint32_t rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    return (uint32_t(r) << 24) | (uint32_t(g) << 16) | (uint32_t(b) << 8) | a;
}

class Renderer {
public:
    bool init(ANativeWindow* window);   // full EGL + GL setup (idempotent per surface)
    void destroy();                     // release surface + context
    bool ready() const { return display_ != EGL_NO_DISPLAY && surface_ != EGL_NO_SURFACE; }

    int internalW() const { return kInternalW; }
    int internalH() const { return internalH_; }

    // Maps a raw touch coordinate into internal pixels. Returns false outside view.
    void screenToInternal(float sx, float sy, float* ix, float* iy) const;

    void beginFrame();

    // Draw calls append to the current phase batch:
    //   world phase (before applyBloom): part of the bloomed scene
    //   overlay phase (after applyBloom): crisp, un-bloomed (HUD/UI)
    void sprite(SpriteId id, Vec2 center, float scale, uint32_t color, bool additive,
                float rotRad = 0.f);
    void rect(float x, float y, float w, float h, uint32_t color, bool additive = false);
    void line(Vec2 a, Vec2 b, float thickness, uint32_t color, bool additive);
    void text(int x, int y, const char* s, uint32_t color, int scale = 1);
    int textWidth(const char* s, int scale = 1) const;
    void textCentered(int cx, int y, const char* s, uint32_t color, int scale = 1);

    void applyBloom(bool glowOn);   // flushes world batches, runs bloom chain, enters overlay phase

    // Flushes overlay, upscales to screen, swaps. Returns CPU encode ms of this
    // frame and the most recent available GPU elapsed ms (-1 if unsupported).
    void endFrame(double* cpuMs, double* gpuMs);

private:
    struct Vtx { float x, y, u, v, r, g, b, a; };

    void flushBatch(std::vector<Vtx>& batch, bool additive);
    void fullscreenPass(GLuint prog, GLuint tex);
    GLuint compile(const char* vs, const char* fs);
    void makeFbo(GLuint* fbo, GLuint* tex, int w, int h, GLenum filter);
    void pushQuad(std::vector<Vtx>& b, float x0, float y0, float x1, float y1,
                  float u0, float v0, float u1, float v1, uint32_t color, float rot,
                  float cx, float cy);

    EGLDisplay display_ = EGL_NO_DISPLAY;
    EGLSurface surface_ = EGL_NO_SURFACE;
    EGLContext context_ = EGL_NO_CONTEXT;

    int screenW_ = 0, screenH_ = 0;
    int internalH_ = 480;
    int viewScale_ = 1, viewOffX_ = 0, viewOffY_ = 0;

    GLuint atlasTex_ = 0;
    AtlasRegion regions_[SPR_COUNT];

    GLuint quadProg_ = 0, brightProg_ = 0, blurProg_ = 0, copyProg_ = 0;
    GLint quadViewportLoc_ = -1, blurDirLoc_ = -1;
    GLuint vbo_ = 0, vao_ = 0, fsVbo_ = 0, fsVao_ = 0;

    GLuint sceneFbo_ = 0, sceneTex_ = 0;
    GLuint halfFboA_ = 0, halfTexA_ = 0, halfFboB_ = 0, halfTexB_ = 0;
    int halfW_ = 0, halfH_ = 0;

    std::vector<Vtx> worldAlpha_, worldAdd_, overlayAlpha_, overlayAdd_;
    bool overlayPhase_ = false;

    // GL_EXT_disjoint_timer_query ring (optional)
    bool timerQuerySupported_ = false;
    GLuint queries_[4] = {0, 0, 0, 0};
    int queryWrite_ = 0, queryRead_ = 0, queriesInFlight_ = 0;
    double lastGpuMs_ = -1.0;

    int64_t frameCpuStart_ = 0;
};
