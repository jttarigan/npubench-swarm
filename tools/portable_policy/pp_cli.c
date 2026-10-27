// Mac/host driver: pp_cli WEIGHTS.bin ROWS.f32 OUT.f32  (rows of n_in floats -> n_out floats)
#include "policy_portable.h"
#include <stdio.h>
#include <stdlib.h>
static void* slurp(const char* path, long* n) {
    FILE* f = fopen(path, "rb"); if (!f) return 0;
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    void* b = malloc(*n); fread(b, 1, *n, f); fclose(f); return b;
}
int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "usage: pp_cli W.bin ROWS.f32 OUT.f32\n"); return 2; }
    long nw, nr; void* w = slurp(argv[1], &nw); float* rows = (float*)slurp(argv[2], &nr);
    PortablePolicy* p = pp_load(w, (unsigned long)nw);
    if (!p || !rows) { fprintf(stderr, "load failed\n"); return 1; }
    int ni = pp_n_in(p), no = pp_n_out(p); long n = nr / (4L * ni);
    float* out = (float*)malloc(4L * no * n);
    for (long i = 0; i < n; i++) pp_forward(p, rows + i * ni, out + i * no);
    FILE* f = fopen(argv[3], "wb"); fwrite(out, 4, (size_t)(no * n), f); fclose(f);
    fprintf(stderr, "%ld rows, %d -> %d\n", n, ni, no);
    return 0;
}
