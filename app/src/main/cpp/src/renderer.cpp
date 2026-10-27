#include "renderer.h"

#include <GLES2/gl2ext.h>
#include <android/native_window.h>

#include <cstring>

#include "font5x7.h"

namespace {

// --- GL_EXT_disjoint_timer_query entry points (optional GPU render timing) ---
PFNGLGENQUERIESEXTPROC pGenQueriesEXT = nullptr;
PFNGLDELETEQUERIESEXTPROC pDeleteQueriesEXT = nullptr;
PFNGLBEGINQUERYEXTPROC pBeginQueryEXT = nullptr;
PFNGLENDQUERYEXTPROC pEndQueryEXT = nullptr;
PFNGLGETQUERYOBJECTUIVEXTPROC pGetQueryObjectuivEXT = nullptr;
PFNGLGETQUERYOBJECTUI64VEXTPROC pGetQueryObjectui64vEXT = nullptr;

const char* kQuadVS = R"(#version 300 es
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
layout(location=2) in vec4 aCol;
uniform vec2 uViewport;
out vec2 vUV; out vec4 vCol;
void main() {
  gl_Position = vec4(aPos.x / uViewport.x * 2.0 - 1.0,
                     1.0 - aPos.y / uViewport.y * 2.0, 0.0, 1.0);
  vUV = aUV; vCol = aCol;
})";

const char* kQuadFS = R"(#version 300 es
precision mediump float;
uniform sampler2D uTex;
in vec2 vUV; in vec4 vCol;
out vec4 frag;
void main() { frag = texture(uTex, vUV) * vCol; })";

const char* kFsVS = R"(#version 300 es
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); vUV = aUV; })";

const char* kBrightFS = R"(#version 300 es
precision mediump float;
uniform sampler2D uTex;
in vec2 vUV;
out vec4 frag;
void main() {
  vec3 c = texture(uTex, vUV).rgb;
  vec3 b = max(c - vec3(0.55), vec3(0.0)) * 1.6;
  frag = vec4(b, 1.0);
})";

const char* kBlurFS = R"(#version 300 es
precision mediump float;
uniform sampler2D uTex;
uniform vec2 uDir;   // texel step * direction
in vec2 vUV;
out vec4 frag;
void main() {
  vec3 c = texture(uTex, vUV).rgb * 0.2270270270;
  c += texture(uTex, vUV + uDir * 1.3846153846).rgb * 0.3162162162;
  c += texture(uTex, vUV - uDir * 1.3846153846).rgb * 0.3162162162;
  c += texture(uTex, vUV + uDir * 3.2307692308).rgb * 0.0702702703;
  c += texture(uTex, vUV - uDir * 3.2307692308).rgb * 0.0702702703;
  frag = vec4(c, 1.0);
})";

const char* kCopyFS = R"(#version 300 es
precision mediump float;
uniform sampler2D uTex;
in vec2 vUV;
out vec4 frag;
void main() { frag = texture(uTex, vUV); })";

}  // namespace

GLuint Renderer::compile(const char* vs, const char* fs) {
    auto sh = [](GLenum type, const char* src) -> GLuint {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[512];
            glGetShaderInfoLog(s, sizeof(log), nullptr, log);
            LOGE("shader compile failed: %s", log);
        }
        return s;
    };
    GLuint v = sh(GL_VERTEX_SHADER, vs), f = sh(GL_FRAGMENT_SHADER, fs);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, v);
    glAttachShader(prog, f);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) LOGE("program link failed");
    glDeleteShader(v);
    glDeleteShader(f);
    return prog;
}

void Renderer::makeFbo(GLuint* fbo, GLuint* tex, int w, int h, GLenum filter) {
    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_2D, *tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, *fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        LOGE("FBO incomplete (%dx%d)", w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

bool Renderer::init(ANativeWindow* window) {
    destroy();

    display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    eglInitialize(display_, nullptr, nullptr);

    const EGLint cfgAttribs[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
                                 EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
                                 EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8,
                                 EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
                                 EGL_NONE};
    EGLConfig config;
    EGLint numCfg = 0;
    eglChooseConfig(display_, cfgAttribs, &config, 1, &numCfg);
    if (numCfg < 1) {
        LOGE("no EGL config");
        return false;
    }

    surface_ = eglCreateWindowSurface(display_, config, window, nullptr);
    const EGLint ctxAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    context_ = eglCreateContext(display_, config, EGL_NO_CONTEXT, ctxAttribs);
    if (surface_ == EGL_NO_SURFACE || context_ == EGL_NO_CONTEXT) {
        LOGE("EGL surface/context creation failed");
        return false;
    }
    eglMakeCurrent(display_, surface_, surface_, context_);
    eglSwapInterval(display_, 1);

    eglQuerySurface(display_, surface_, EGL_WIDTH, &screenW_);
    eglQuerySurface(display_, surface_, EGL_HEIGHT, &screenH_);

    // Internal portrait resolution: fixed 240 x kInternalH on every device (the
    // world size is part of the determinism contract); letterboxed to fit.
    internalH_ = kInternalH;
    viewScale_ = screenW_ / kInternalW;
    int vs2 = screenH_ / internalH_;
    if (vs2 < viewScale_) viewScale_ = vs2;
    if (viewScale_ < 1) viewScale_ = 1;
    viewOffX_ = (screenW_ - kInternalW * viewScale_) / 2;
    viewOffY_ = (screenH_ - internalH_ * viewScale_) / 2;
    LOGI("screen %dx%d internal %dx%d scale %d GL_RENDERER=%s", screenW_, screenH_,
         kInternalW, internalH_, viewScale_, (const char*)glGetString(GL_RENDERER));

    // Programs
    quadProg_ = compile(kQuadVS, kQuadFS);
    brightProg_ = compile(kFsVS, kBrightFS);
    blurProg_ = compile(kFsVS, kBlurFS);
    copyProg_ = compile(kFsVS, kCopyFS);
    quadViewportLoc_ = glGetUniformLocation(quadProg_, "uViewport");
    blurDirLoc_ = glGetUniformLocation(blurProg_, "uDir");

    // Atlas
    AtlasRegion regions[SPR_COUNT];
    auto pixels = buildAtlas(regions);
    memcpy(regions_, regions, sizeof(regions_));
    glGenTextures(1, &atlasTex_);
    glBindTexture(GL_TEXTURE_2D, atlasTex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, kAtlasSize, kAtlasSize, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, pixels.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // FBOs: scene at internal res (NEAREST for the final crisp upscale),
    // half-res ping-pong pair for the bloom blur (LINEAR).
    halfW_ = kInternalW / 2;
    halfH_ = internalH_ / 2;
    makeFbo(&sceneFbo_, &sceneTex_, kInternalW, internalH_, GL_NEAREST);
    makeFbo(&halfFboA_, &halfTexA_, halfW_, halfH_, GL_LINEAR);
    makeFbo(&halfFboB_, &halfTexB_, halfW_, halfH_, GL_LINEAR);

    // Batch VBO/VAO
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)(4 * sizeof(float)));

    // Fullscreen triangle
    const float fsTri[] = {-1.f, -1.f, 0.f, 0.f, 3.f, -1.f, 2.f, 0.f, -1.f, 3.f, 0.f, 2.f};
    glGenVertexArrays(1, &fsVao_);
    glGenBuffers(1, &fsVbo_);
    glBindVertexArray(fsVao_);
    glBindBuffer(GL_ARRAY_BUFFER, fsVbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(fsTri), fsTri, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);

    // Optional GPU timer queries
    const char* ext = (const char*)glGetString(GL_EXTENSIONS);
    timerQuerySupported_ = ext && strstr(ext, "GL_EXT_disjoint_timer_query");
    if (timerQuerySupported_) {
        pGenQueriesEXT = (PFNGLGENQUERIESEXTPROC)eglGetProcAddress("glGenQueriesEXT");
        pDeleteQueriesEXT = (PFNGLDELETEQUERIESEXTPROC)eglGetProcAddress("glDeleteQueriesEXT");
        pBeginQueryEXT = (PFNGLBEGINQUERYEXTPROC)eglGetProcAddress("glBeginQueryEXT");
        pEndQueryEXT = (PFNGLENDQUERYEXTPROC)eglGetProcAddress("glEndQueryEXT");
        pGetQueryObjectuivEXT =
            (PFNGLGETQUERYOBJECTUIVEXTPROC)eglGetProcAddress("glGetQueryObjectuivEXT");
        pGetQueryObjectui64vEXT =
            (PFNGLGETQUERYOBJECTUI64VEXTPROC)eglGetProcAddress("glGetQueryObjectui64vEXT");
        if (pGenQueriesEXT && pBeginQueryEXT && pEndQueryEXT && pGetQueryObjectuivEXT &&
            pGetQueryObjectui64vEXT) {
            pGenQueriesEXT(4, queries_);
        } else {
            timerQuerySupported_ = false;
        }
    }
    LOGI("GPU timer queries: %s", timerQuerySupported_ ? "available" : "unavailable");

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);

    worldAlpha_.reserve(4096);
    worldAdd_.reserve(4096);
    overlayAlpha_.reserve(4096);
    overlayAdd_.reserve(512);
    return true;
}

void Renderer::destroy() {
    if (display_ != EGL_NO_DISPLAY) {
        eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (context_ != EGL_NO_CONTEXT) eglDestroyContext(display_, context_);
        if (surface_ != EGL_NO_SURFACE) eglDestroySurface(display_, surface_);
        eglTerminate(display_);
    }
    display_ = EGL_NO_DISPLAY;
    surface_ = EGL_NO_SURFACE;
    context_ = EGL_NO_CONTEXT;
    queriesInFlight_ = 0;
    queryWrite_ = queryRead_ = 0;
    lastGpuMs_ = -1.0;
}

void Renderer::screenToInternal(float sx, float sy, float* ix, float* iy) const {
    *ix = (sx - (float)viewOffX_) / (float)viewScale_;
    *iy = (sy - (float)viewOffY_) / (float)viewScale_;
}

void Renderer::beginFrame() {
    frameCpuStart_ = now_ns();
    if (timerQuerySupported_ && queriesInFlight_ < 4) {
        pBeginQueryEXT(GL_TIME_ELAPSED_EXT, queries_[queryWrite_]);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, sceneFbo_);
    glViewport(0, 0, kInternalW, internalH_);
    glClearColor(0.043f, 0.031f, 0.102f, 1.f);  // deep navy-purple
    glClear(GL_COLOR_BUFFER_BIT);
    overlayPhase_ = false;
    worldAlpha_.clear();
    worldAdd_.clear();
    overlayAlpha_.clear();
    overlayAdd_.clear();
}

void Renderer::pushQuad(std::vector<Vtx>& b, float x0, float y0, float x1, float y1,
                        float u0, float v0, float u1, float v1, uint32_t color, float rot,
                        float cx, float cy) {
    float r = ((color >> 24) & 0xFF) / 255.f;
    float g = ((color >> 16) & 0xFF) / 255.f;
    float bl = ((color >> 8) & 0xFF) / 255.f;
    float a = (color & 0xFF) / 255.f;
    float pts[4][2] = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
    if (rot != 0.f) {
        float c = std::cos(rot), s = std::sin(rot);
        for (auto& pt : pts) {
            float dx = pt[0] - cx, dy = pt[1] - cy;
            pt[0] = cx + dx * c - dy * s;
            pt[1] = cy + dx * s + dy * c;
        }
    }
    float uvs[4][2] = {{u0, v0}, {u1, v0}, {u1, v1}, {u0, v1}};
    int idx[6] = {0, 1, 2, 0, 2, 3};
    for (int i : idx)
        b.push_back({pts[i][0], pts[i][1], uvs[i][0], uvs[i][1], r, g, bl, a});
}

void Renderer::sprite(SpriteId id, Vec2 c, float scale, uint32_t color, bool additive,
                      float rotRad) {
    const AtlasRegion& rg = regions_[id];
    float w = rg.w * scale, h = rg.h * scale;
    auto& batch = overlayPhase_ ? (additive ? overlayAdd_ : overlayAlpha_)
                                : (additive ? worldAdd_ : worldAlpha_);
    pushQuad(batch, c.x - w / 2, c.y - h / 2, c.x + w / 2, c.y + h / 2,
             rg.x / (float)kAtlasSize, rg.y / (float)kAtlasSize,
             (rg.x + rg.w) / (float)kAtlasSize, (rg.y + rg.h) / (float)kAtlasSize, color,
             rotRad, c.x, c.y);
}

void Renderer::rect(float x, float y, float w, float h, uint32_t color, bool additive) {
    const AtlasRegion& rg = regions_[SPR_WHITE];
    auto& batch = overlayPhase_ ? (additive ? overlayAdd_ : overlayAlpha_)
                                : (additive ? worldAdd_ : worldAlpha_);
    float u = (rg.x + 0.5f) / kAtlasSize, v = (rg.y + 0.5f) / kAtlasSize;
    pushQuad(batch, x, y, x + w, y + h, u, v, u, v, color, 0.f, 0.f, 0.f);
}

void Renderer::line(Vec2 a, Vec2 b, float thickness, uint32_t color, bool additive) {
    Vec2 d = b - a;
    float len = d.len();
    if (len < 1e-4f) return;
    float rot = std::atan2(d.y, d.x);
    Vec2 mid = (a + b) * 0.5f;
    const AtlasRegion& rg = regions_[SPR_WHITE];
    float u = (rg.x + 0.5f) / kAtlasSize, v = (rg.y + 0.5f) / kAtlasSize;
    auto& batch = overlayPhase_ ? (additive ? overlayAdd_ : overlayAlpha_)
                                : (additive ? worldAdd_ : worldAlpha_);
    pushQuad(batch, mid.x - len / 2, mid.y - thickness / 2, mid.x + len / 2,
             mid.y + thickness / 2, u, v, u, v, color, rot, mid.x, mid.y);
}

void Renderer::text(int x, int y, const char* s, uint32_t color, int scale) {
    auto& batch = overlayPhase_ ? overlayAlpha_ : worldAlpha_;
    int cx = x;
    for (const char* p = s; *p; p++) {
        char c = *p;
        if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
        int i;
        if (c == '|') {
            i = kFontBarIndex;
        } else {
            if (c < kFontFirstChar || c > kFontLastChar) c = '?';
            i = c - kFontFirstChar;
        }
        float gx = (float)((i % 16) * 8), gy = (float)(kFontAtlasY + (i / 16) * 8);
        pushQuad(batch, (float)cx, (float)y, (float)(cx + kFontGlyphW * scale),
                 (float)(y + kFontGlyphH * scale), gx / kAtlasSize, gy / kAtlasSize,
                 (gx + kFontGlyphW) / kAtlasSize, (gy + kFontGlyphH) / kAtlasSize, color,
                 0.f, 0.f, 0.f);
        cx += (kFontGlyphW + 1) * scale;
    }
}

int Renderer::textWidth(const char* s, int scale) const {
    return (int)strlen(s) * (kFontGlyphW + 1) * scale - scale;
}

void Renderer::textCentered(int cx, int y, const char* s, uint32_t color, int scale) {
    text(cx - textWidth(s, scale) / 2, y, s, color, scale);
}

void Renderer::flushBatch(std::vector<Vtx>& batch, bool additive) {
    if (batch.empty()) return;
    glUseProgram(quadProg_);
    glUniform2f(quadViewportLoc_, (float)kInternalW, (float)internalH_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, atlasTex_);
    glBlendFunc(GL_SRC_ALPHA, additive ? GL_ONE : GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(batch.size() * sizeof(Vtx)), batch.data(),
                 GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)batch.size());
    batch.clear();
}

void Renderer::fullscreenPass(GLuint prog, GLuint tex) {
    glUseProgram(prog);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glBindVertexArray(fsVao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

void Renderer::applyBloom(bool glowOn) {
    // finish the world (bloom-participating) content
    glBindFramebuffer(GL_FRAMEBUFFER, sceneFbo_);
    glViewport(0, 0, kInternalW, internalH_);
    flushBatch(worldAlpha_, false);
    flushBatch(worldAdd_, true);

    if (glowOn) {
        glDisable(GL_BLEND);
        // bright pass -> half A
        glBindFramebuffer(GL_FRAMEBUFFER, halfFboA_);
        glViewport(0, 0, halfW_, halfH_);
        fullscreenPass(brightProg_, sceneTex_);
        // horizontal blur -> half B
        glBindFramebuffer(GL_FRAMEBUFFER, halfFboB_);
        glUseProgram(blurProg_);
        glUniform2f(blurDirLoc_, 1.f / (float)halfW_, 0.f);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, halfTexA_);
        glBindVertexArray(fsVao_);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        // vertical blur -> half A
        glBindFramebuffer(GL_FRAMEBUFFER, halfFboA_);
        glUniform2f(blurDirLoc_, 0.f, 1.f / (float)halfH_);
        glBindTexture(GL_TEXTURE_2D, halfTexB_);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        // additive composite back into the internal-res scene
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        glBindFramebuffer(GL_FRAMEBUFFER, sceneFbo_);
        glViewport(0, 0, kInternalW, internalH_);
        fullscreenPass(copyProg_, halfTexA_);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }
    overlayPhase_ = true;
}

void Renderer::endFrame(double* cpuMs, double* gpuMs) {
    // overlay (HUD/UI) — crisp, after bloom
    glBindFramebuffer(GL_FRAMEBUFFER, sceneFbo_);
    glViewport(0, 0, kInternalW, internalH_);
    flushBatch(overlayAlpha_, false);
    flushBatch(overlayAdd_, true);

    // single nearest-neighbour integer upscale to screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, screenW_, screenH_);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_BLEND);
    glViewport(viewOffX_, viewOffY_, kInternalW * viewScale_, internalH_ * viewScale_);
    fullscreenPass(copyProg_, sceneTex_);
    glEnable(GL_BLEND);

    if (timerQuerySupported_ && queriesInFlight_ < 4) {
        pEndQueryEXT(GL_TIME_ELAPSED_EXT);
        queryWrite_ = (queryWrite_ + 1) % 4;
        queriesInFlight_++;
    }
    // harvest oldest finished query (non-blocking)
    while (timerQuerySupported_ && queriesInFlight_ > 0) {
        GLuint avail = 0;
        pGetQueryObjectuivEXT(queries_[queryRead_], GL_QUERY_RESULT_AVAILABLE_EXT, &avail);
        if (!avail) break;
        GLuint64 ns = 0;
        pGetQueryObjectui64vEXT(queries_[queryRead_], GL_QUERY_RESULT_EXT, &ns);
        lastGpuMs_ = (double)ns * 1e-6;
        queryRead_ = (queryRead_ + 1) % 4;
        queriesInFlight_--;
    }

    *cpuMs = ns_to_ms(now_ns() - frameCpuStart_);
    *gpuMs = lastGpuMs_;
    eglSwapBuffers(display_, surface_);
}
