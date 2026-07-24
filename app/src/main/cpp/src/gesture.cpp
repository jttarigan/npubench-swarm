#include "gesture.h"

#include <algorithm>

bool resampleStroke(const std::vector<Vec2>& raw, float* out) {
    if (raw.size() < 8) return false;
    float total = 0.f;
    for (size_t i = 1; i < raw.size(); i++) total += (raw[i] - raw[i - 1]).len();
    if (total < 15.f) return false;   // taps / jitters aren't gestures

    // equidistant resample
    const float step = total / (kGesturePts - 1);
    Vec2 pts[kGesturePts];
    pts[0] = raw[0];
    size_t seg = 1;
    float segStart = 0.f;
    float segLen = (raw[1] - raw[0]).len();
    for (int k = 1; k < kGesturePts - 1; k++) {
        float target = step * k;
        while (segStart + segLen < target && seg + 1 < raw.size()) {
            segStart += segLen;
            seg++;
            segLen = (raw[seg] - raw[seg - 1]).len();
        }
        float f = segLen > 1e-6f ? (target - segStart) / segLen : 0.f;
        pts[k] = raw[seg - 1] + (raw[seg] - raw[seg - 1]) * f;
    }
    pts[kGesturePts - 1] = raw.back();

    // normalize: centroid to origin, scale by half max bbox dim
    Vec2 c{0, 0}, lo = pts[0], hi = pts[0];
    for (auto& p : pts) {
        c += p;
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y);
    }
    c = c * (1.f / kGesturePts);
    float scale = std::max(hi.x - lo.x, hi.y - lo.y) * 0.5f;
    if (scale < 1e-3f) return false;
    for (int k = 0; k < kGesturePts; k++) {
        out[k * 2 + 0] = (pts[k].x - c.x) / scale;
        out[k * 2 + 1] = (pts[k].y - c.y) / scale;
    }
    return true;
}

GestureResult classifyStroke(NnModel& model, const std::vector<Vec2>& raw) {
    GestureResult r;
    float in[kGesturePts * 2];
    int64_t t0 = now_ns();
    if (!resampleStroke(raw, in)) return r;
    r.resampleMs = ns_to_ms(now_ns() - t0);

    float probs[kGestureClasses] = {0};
    if (!model.invoke(in, kGesturePts * 2, probs, kGestureClasses, &r.inferMs)) return r;
    r.ok = true;
    r.cls = 0;
    for (int i = 1; i < kGestureClasses; i++)
        if (probs[i] > probs[r.cls]) r.cls = i;
    r.conf = probs[r.cls];
    return r;
}
