// Study 3 on-device prevention (experiment C): grid rounding with a guarded recompute by
// the portable policy (tools/portable_policy), the same rule as analysis/cosim.py
// Prevent. Off unless the system property debug.npubench.prevent holds a spec:
//   grid:K            round all outputs to multiples of 2^-K (half to even)
//   fallback:D        rows with the fire value within D of 0.35 or, with a grid, any
//                     output within D of a rounding edge are recomputed portably
//   grid:K+fallback:D both;   portable: every row recomputed (bit check, no grid)
// Float32 models only.
#pragma once
#include <string>

struct PortablePolicy;

struct Prevention {
    std::string spec;     // "" = off
    int k = -1;           // grid 2^-k; -1 = no grid
    double delta = -1.0;  // guard width; < 0 = no guard
    bool all = false;     // "portable"
    bool active() const { return all || k >= 0 || delta >= 0; }
    bool needsPortable() const { return all || delta >= 0; }
};

Prevention preventionFromProperty();
// Applies p in place to rows [0, rows) of out (3 per row), recomputing with pp where the
// rule says so. Returns the number of rows recomputed; *ms gets the recompute time.
int applyPrevention(const Prevention& p, const PortablePolicy* pp, const float* feat,
                    float* out, int rows, double* ms);
