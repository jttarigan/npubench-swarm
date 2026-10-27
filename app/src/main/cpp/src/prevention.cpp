#include "prevention.h"

#include <sys/system_properties.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "common.h"
#include "policy_portable.h"

Prevention preventionFromProperty() {
    Prevention p;
    char v[PROP_VALUE_MAX] = {0};
    if (__system_property_get("debug.npubench.prevent", v) <= 0) return p;
    std::string s(v);
    if (s == "portable") {
        p.all = true;
        p.spec = s;
        return p;
    }
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t end = s.find('+', pos);
        std::string part = s.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        if (part.rfind("grid:", 0) == 0) p.k = atoi(part.c_str() + 5);
        else if (part.rfind("fallback:", 0) == 0) p.delta = atof(part.c_str() + 9);
        if (end == std::string::npos) break;
        pos = end + 1;
    }
    if (p.k > 12) p.k = -1;
    if (p.active()) p.spec = s;
    return p;
}

// Same test as cosim.Prevent.near_boundary: fire distance in float32, edge distance in
// double.
static bool nearBoundary(const Prevention& p, const float* o) {
    if (p.delta < 0) return false;
    if (std::fabs(o[2] - 0.35f) < (float)p.delta) return true;
    if (p.k < 0) return false;
    const double scale = std::ldexp(1.0, p.k);
    for (int c = 0; c < kPolicyOut; c++) {
        const double scaled = (double)o[c] * scale;
        const double frac = scaled - std::floor(scaled);
        if (std::fabs(frac - 0.5) / scale < p.delta) return true;
    }
    return false;
}

int applyPrevention(const Prevention& p, const PortablePolicy* pp, const float* feat,
                    float* out, int rows, double* ms) {
    int n = 0;
    const int64_t t0 = now_ns();
    for (int r = 0; r < rows; r++) {
        float* o = out + r * kPolicyOut;
        if (pp && (p.all || nearBoundary(p, o))) {
            pp_forward(pp, feat + r * kFeatureDim, o);
            n++;
        }
        if (p.k >= 0) {
            const float step = std::ldexp(1.0f, -p.k);
            for (int c = 0; c < kPolicyOut; c++) o[c] = std::nearbyint(o[c] / step) * step;
        }
    }
    *ms = ns_to_ms(now_ns() - t0);
    return n;
}
