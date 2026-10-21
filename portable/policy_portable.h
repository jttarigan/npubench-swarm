// Portable policy network for study 3: the same float32 result on every IEEE-754
// platform. Fixed summation order (bias first, then inputs in order), separate
// multiply and add (build with -ffp-contract=off, no fast-math), and a tanh made of
// basic operations only (library tanhf differs between platforms).
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef struct PortablePolicy PortablePolicy;
PortablePolicy* pp_load(const void* data, unsigned long size);   // NPPW blob
void pp_free(PortablePolicy* p);
int pp_n_in(const PortablePolicy* p);
int pp_n_out(const PortablePolicy* p);
// One row: in[n_in] -> out[n_out].
void pp_forward(const PortablePolicy* p, const float* in, float* out);
float pp_tanh(float z);
#ifdef __cplusplus
}
#endif
