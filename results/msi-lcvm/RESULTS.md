# x86 kit results: 12th Gen Intel(R) Core(TM) i7-12700F

Windows-11-10.0.26200-SP0, Python 3.13.14, TensorFlow 2.20.0, numpy 2.5.4, 20 CPUs; 2026-10-21T15:49:08+0700 to 2026-10-21T15:51:39+0700.

## Replays over the recorded inputs (values differing of all; max abs diff; median ULP; fire flips)

| replay | reference | identical | differing | max abs diff | median ULP | flips |
|---|---|---|---|---|---|---|
| replay_xlarge_f32_xnnpack | device:R1_android_cpu_xlarge_f32 | no | 235,897 of 370,062 | 1.25e-06 | 2 | 0 |
| replay_xlarge_f32_xnnpack | device:R11_iphone_cpu_xlarge_f32_litert220 | no | 230,469 of 370,062 | 1.07e-06 | 2 | 0 |
| replay_xlarge_f32_xnnpack | device:R19_iphone_cpu_xlarge_f32_litert217 | no | 239,390 of 370,062 | 1.07e-06 | 2 | 0 |
| replay_xlarge_f32_xnnpack | mac_m5:replay_xlarge_f32_xnnpack | no | 199,282 of 370,062 | 1.07e-06 | 2 | 0 |
| replay_xlarge_f32_builtin | device:R1_android_cpu_xlarge_f32 | no | 195,754 of 370,062 | 8.94e-07 | 2 | 0 |
| replay_xlarge_f32_builtin | device:R11_iphone_cpu_xlarge_f32_litert220 | no | 216,180 of 370,062 | 1.14e-06 | 2 | 0 |
| replay_xlarge_f32_builtin | device:R19_iphone_cpu_xlarge_f32_litert217 | no | 244,429 of 370,062 | 1.15e-06 | 2 | 0 |
| replay_xlarge_f32_builtin | mac_m5:replay_xlarge_f32_builtin | no | 216,180 of 370,062 | 1.14e-06 | 2 | 0 |
| replay_xlarge_f32_ref | device:R1_android_cpu_xlarge_f32 | no | 232,280 of 370,062 | 1.45e-06 | 2 | 0 |
| replay_xlarge_f32_ref | device:R11_iphone_cpu_xlarge_f32_litert220 | no | 220,426 of 370,062 | 8.34e-07 | 1 | 0 |
| replay_xlarge_f32_ref | device:R19_iphone_cpu_xlarge_f32_litert217 | no | 226,116 of 370,062 | 8.64e-07 | 1 | 0 |
| replay_xlarge_f32_ref | mac_m5:replay_xlarge_f32_ref | no | 138,001 of 370,062 | 8.05e-07 | 2 | 0 |
| replay_xlarge_int8_xnnpack | device:R12_iphone_cpu_xlarge_int8 | YES | 0 of 370,062 | 0 | 0 | 0 |
| replay_xlarge_int8_xnnpack | device:R2_android_cpu_xlarge_int8 | no | 4,658 of 370,062 | 0.109 | 786432 | 5 |
| replay_xlarge_int8_xnnpack | mac_m5:replay_xlarge_int8_xnnpack | YES | 0 of 370,062 | 0 | 0 | 0 |
| replay_xlarge_int8_builtin | device:R12_iphone_cpu_xlarge_int8 | YES | 0 of 370,062 | 0 | 0 | 0 |
| replay_xlarge_int8_builtin | device:R2_android_cpu_xlarge_int8 | no | 4,658 of 370,062 | 0.109 | 786432 | 5 |
| replay_xlarge_int8_builtin | mac_m5:replay_xlarge_int8_builtin | YES | 0 of 370,062 | 0 | 0 | 0 |
| replay_xlarge_int8_ref | device:R12_iphone_cpu_xlarge_int8 | no | 4,658 of 370,062 | 0.109 | 786432 | 5 |
| replay_xlarge_int8_ref | device:R2_android_cpu_xlarge_int8 | YES | 0 of 370,062 | 0 | 0 | 0 |
| replay_xlarge_int8_ref | mac_m5:replay_xlarge_int8_ref | YES | 0 of 370,062 | 0 | 0 | 0 |
| replay_tiny_f32_xnnpack | device:T1_android_cpu_tiny_f32 | no | 291,236 of 389,232 | 5.36e-07 | 2 | 0 |
| replay_tiny_f32_xnnpack | mac_m5:replay_tiny_f32_xnnpack | no | 189,875 of 389,232 | 2.98e-07 | 2 | 0 |
| replay_tiny_f32_builtin | device:T1_android_cpu_tiny_f32 | no | 218,116 of 389,232 | 4.47e-07 | 2 | 0 |
| replay_tiny_f32_builtin | mac_m5:replay_tiny_f32_builtin | no | 232,152 of 389,232 | 4.47e-07 | 2 | 0 |
| replay_tiny_f32_ref | device:T1_android_cpu_tiny_f32 | no | 260,037 of 389,232 | 4.47e-07 | 1 | 0 |
| replay_tiny_f32_ref | mac_m5:replay_tiny_f32_ref | no | 141,023 of 389,232 | 3.87e-07 | 1 | 0 |
| replay_tiny_int8_xnnpack | device:T2_android_cpu_tiny_int8 | no | 3,061 of 389,232 | 0.0234 | 262144 | 3 |
| replay_tiny_int8_xnnpack | mac_m5:replay_tiny_int8_xnnpack | YES | 0 of 389,232 | 0 | 0 | 0 |
| replay_tiny_int8_builtin | device:T2_android_cpu_tiny_int8 | no | 3,052 of 389,232 | 0.0234 | 262144 | 3 |
| replay_tiny_int8_builtin | mac_m5:replay_tiny_int8_builtin | YES | 0 of 389,232 | 0 | 0 | 0 |
| replay_tiny_int8_ref | device:T2_android_cpu_tiny_int8 | no | 1 of 389,232 | 0.0156 | 262144 | 0 |
| replay_tiny_int8_ref | mac_m5:replay_tiny_int8_ref | YES | 0 of 389,232 | 0 | 0 | 0 |
| pp_xlarge | mac_m5 | YES | 0 of 370,062 | 0 | 0 | 0 |
| pp_tiny | mac_m5 | YES | 0 of 389,232 | 0 | 0 | 0 |

## Re-simulation of the recorded games (host build of the game, recorded outputs fed in)

- hostsim_strict/xlarge: differs: {'ok': False, 'ticks': 3724, 'state_identical_ticks': 50, 'first_state_diff_tick': 17, 'feature_mismatches': 3656, 'file_equal': False}
- hostsim_strict/tiny: differs: {'ok': False, 'ticks': 4407, 'state_identical_ticks': 29, 'first_state_diff_tick': 13, 'feature_mismatches': 4349, 'file_equal': False}
- hostsim_precise/xlarge: differs: {'ok': False, 'ticks': 3724, 'state_identical_ticks': 50, 'first_state_diff_tick': 17, 'feature_mismatches': 3656, 'file_equal': False}
- hostsim_precise/tiny: differs: {'ok': False, 'ticks': 4407, 'state_identical_ticks': 29, 'first_state_diff_tick': 13, 'feature_mismatches': 4349, 'file_equal': False}
- hostsim_contract/xlarge: differs: {'ok': False, 'ticks': 3724, 'state_identical_ticks': 42, 'first_state_diff_tick': 4, 'feature_mismatches': 3697, 'file_equal': False}
- hostsim_contract/tiny: differs: {'ok': False, 'ticks': 4407, 'state_identical_ticks': 37, 'first_state_diff_tick': 4, 'feature_mismatches': 4386, 'file_equal': False}

## Closed loop over seeds (80 games)

- against mac_m5: 0 of 80 games identical on every tick; outcome equal in 80
