#include "policy_portable.h"
#include <stdlib.h>
#include <string.h>

#define MAXL 8
#define MAXW 1024
struct PortablePolicy {
    int n_layers, n_in[MAXL], n_out[MAXL];
    float *w[MAXL], *b[MAXL];   // w: [n_in][n_out] (transposed at load: output index fastest)
    void* mem;
    float* wt;
};

static unsigned rd32(const unsigned char* p) { return p[0] | p[1] << 8 | p[2] << 16 | (unsigned)p[3] << 24; }

PortablePolicy* pp_load(const void* data, unsigned long size) {
    const unsigned char* p = (const unsigned char*)data;
    if (size < 8 || memcmp(p, "NPPW", 4) != 0) return 0;
    PortablePolicy* pp = (PortablePolicy*)calloc(1, sizeof *pp);
    pp->n_layers = (int)rd32(p + 4);
    if (pp->n_layers < 1 || pp->n_layers > MAXL) { free(pp); return 0; }
    pp->mem = malloc(size);
    memcpy(pp->mem, data, size);
    unsigned char* q = (unsigned char*)pp->mem + 8;
    for (int l = 0; l < pp->n_layers; l++) {
        pp->n_in[l] = (int)rd32(q); pp->n_out[l] = (int)rd32(q + 4); q += 8;
        pp->w[l] = (float*)q; q += 4ul * pp->n_in[l] * pp->n_out[l];
        pp->b[l] = (float*)q; q += 4ul * pp->n_out[l];
        if (pp->n_in[l] > MAXW || pp->n_out[l] > MAXW || q > (unsigned char*)pp->mem + size) { pp_free(pp); return 0; }
    }
    unsigned long tot = 0;
    for (int l = 0; l < pp->n_layers; l++) tot += (unsigned long)pp->n_in[l] * pp->n_out[l];
    pp->wt = (float*)malloc(4ul * tot);
    float* t = pp->wt;
    for (int l = 0; l < pp->n_layers; l++) {          // file order W[out][in] -> wt[in][out]
        for (int k = 0; k < pp->n_in[l]; k++)
            for (int j = 0; j < pp->n_out[l]; j++) t[(unsigned long)k * pp->n_out[l] + j] = pp->w[l][(unsigned long)j * pp->n_in[l] + k];
        pp->w[l] = t; t += (unsigned long)pp->n_in[l] * pp->n_out[l];
    }
    return pp;
}

void pp_free(PortablePolicy* p) { if (p) { free(p->wt); free(p->mem); free(p); } }
int pp_n_in(const PortablePolicy* p) { return p->n_in[0]; }
int pp_n_out(const PortablePolicy* p) { return p->n_out[p->n_layers - 1]; }

// exp(x) for |x| <= 20: x = n ln2 + r, |r| <= ln2/2, degree-7 Taylor polynomial in r.
static float pexp(float x) {
    const float ln2_hi = 0.693145751953125f, ln2_lo = 1.428606765330187e-06f;
    float t = x * 1.4426950408889634f;
    int n = (int)(t < 0 ? t - 0.5f : t + 0.5f);
    float r = (x - (float)n * ln2_hi) - (float)n * ln2_lo;
    float s = 1.0f / 5040.0f;
    s = s * r + 1.0f / 720.0f;
    s = s * r + 1.0f / 120.0f;
    s = s * r + 1.0f / 24.0f;
    s = s * r + 1.0f / 6.0f;
    s = s * r + 0.5f;
    s = s * r + 1.0f;
    s = s * r + 1.0f;
    union { float f; unsigned u; } two;                       // 2^n by exponent bits
    two.u = (unsigned)(n + 127) << 23;
    return s * two.f;
}

float pp_tanh(float z) {
    if (z > 9.0f) return 1.0f;
    if (z < -9.0f) return -1.0f;
    float e = pexp(2.0f * z);
    return (e - 1.0f) / (e + 1.0f);
}

void pp_forward(const PortablePolicy* p, const float* in, float* out) {
    float a[MAXW], c[MAXW];
    memcpy(a, in, sizeof(float) * p->n_in[0]);
    for (int l = 0; l < p->n_layers; l++) {
        const int ni = p->n_in[l], no = p->n_out[l];
        const float* w = p->w[l];
        // Every output j sums bias, then a[0]*w, a[1]*w, ... in that order; the loop over j
        // is innermost so it vectorizes across outputs without reordering any sum.
        for (int j = 0; j < no; j++) c[j] = p->b[l][j];
        for (int k = 0; k < ni; k++) {
            const float ak = a[k];
            const float* wk = w + (unsigned long)k * no;
            for (int j = 0; j < no; j++) {
                float m = ak * wk[j];
                c[j] = c[j] + m;
            }
        }
        for (int j = 0; j < no; j++)
            c[j] = (l + 1 < p->n_layers) ? (c[j] > 0.0f ? c[j] : 0.0f) : pp_tanh(c[j]);
        memcpy(a, c, sizeof(float) * no);
    }
    memcpy(out, a, sizeof(float) * p->n_out[p->n_layers - 1]);
}
