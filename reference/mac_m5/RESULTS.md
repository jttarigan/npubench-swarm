# x86 kit results: Apple M5

macOS-26.6.2-arm64-arm-64bit, Python 3.9.6, TensorFlow 2.20.0, numpy 2.0.2, 10 CPUs; 2026-10-21T15:10:12+0700 to 2026-10-21T15:11:33+0700.

## Replays over the recorded inputs (values differing of all; max abs diff; median ULP; fire flips)

| replay | reference | identical | differing | max abs diff | median ULP | flips |
|---|---|---|---|---|---|---|
| replay_xlarge_f32_xnnpack | device:R1_android_cpu_xlarge_f32 | no | 230,645 of 370,062 | 1.31e-06 | 2 | 0 |
| replay_xlarge_f32_xnnpack | device:R11_iphone_cpu_xlarge_f32_litert220 | no | 234,458 of 370,062 | 1.16e-06 | 2 | 0 |
| replay_xlarge_f32_xnnpack | device:R19_iphone_cpu_xlarge_f32_litert217 | no | 242,963 of 370,062 | 1.13e-06 | 2 | 0 |
| replay_xlarge_f32_builtin | device:R1_android_cpu_xlarge_f32 | no | 201,908 of 370,062 | 1.19e-06 | 2 | 0 |
| replay_xlarge_f32_builtin | device:R11_iphone_cpu_xlarge_f32_litert220 | YES | 0 of 370,062 | 0 | 0 | 0 |
| replay_xlarge_f32_builtin | device:R19_iphone_cpu_xlarge_f32_litert217 | no | 187,913 of 370,062 | 2.98e-07 | 1 | 0 |
| replay_xlarge_f32_ref | device:R1_android_cpu_xlarge_f32 | no | 232,540 of 370,062 | 1.27e-06 | 2 | 0 |
| replay_xlarge_f32_ref | device:R11_iphone_cpu_xlarge_f32_litert220 | no | 219,796 of 370,062 | 8.34e-07 | 1 | 0 |
| replay_xlarge_f32_ref | device:R19_iphone_cpu_xlarge_f32_litert217 | no | 225,204 of 370,062 | 8.34e-07 | 1 | 0 |
| replay_xlarge_int8_xnnpack | device:R12_iphone_cpu_xlarge_int8 | YES | 0 of 370,062 | 0 | 0 | 0 |
| replay_xlarge_int8_xnnpack | device:R2_android_cpu_xlarge_int8 | no | 4,658 of 370,062 | 0.109 | 786432 | 5 |
| replay_xlarge_int8_builtin | device:R12_iphone_cpu_xlarge_int8 | YES | 0 of 370,062 | 0 | 0 | 0 |
| replay_xlarge_int8_builtin | device:R2_android_cpu_xlarge_int8 | no | 4,658 of 370,062 | 0.109 | 786432 | 5 |
| replay_xlarge_int8_ref | device:R12_iphone_cpu_xlarge_int8 | no | 4,658 of 370,062 | 0.109 | 786432 | 5 |
| replay_xlarge_int8_ref | device:R2_android_cpu_xlarge_int8 | YES | 0 of 370,062 | 0 | 0 | 0 |
| replay_tiny_f32_xnnpack | device:T1_android_cpu_tiny_f32 | no | 290,127 of 389,232 | 5.29e-07 | 2 | 0 |
| replay_tiny_f32_builtin | device:T1_android_cpu_tiny_f32 | no | 171,142 of 389,232 | 3.87e-07 | 2 | 0 |
| replay_tiny_f32_ref | device:T1_android_cpu_tiny_f32 | no | 250,731 of 389,232 | 3.87e-07 | 1 | 0 |
| replay_tiny_int8_xnnpack | device:T2_android_cpu_tiny_int8 | no | 3,061 of 389,232 | 0.0234 | 262144 | 3 |
| replay_tiny_int8_builtin | device:T2_android_cpu_tiny_int8 | no | 3,052 of 389,232 | 0.0234 | 262144 | 3 |
| replay_tiny_int8_ref | device:T2_android_cpu_tiny_int8 | no | 1 of 389,232 | 0.0156 | 262144 | 0 |

## Re-simulation of the recorded games (host build of the game, recorded outputs fed in)

- hostsim_strict/xlarge: differs: {'ok': False, 'ticks': 3724, 'state_identical_ticks': 50, 'first_state_diff_tick': 17, 'feature_mismatches': 3656, 'file_equal': False}
- hostsim_strict/tiny: differs: {'ok': False, 'ticks': 4407, 'state_identical_ticks': 29, 'first_state_diff_tick': 13, 'feature_mismatches': 4349, 'file_equal': False}
- hostsim_contract/xlarge: byte-identical trace
- hostsim_contract/tiny: byte-identical trace

## Closed loop over seeds (80 games)

