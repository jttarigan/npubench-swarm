# NPU Bench Swarm — run summary

Generated from `E:\OneDrive\MyProject\NPUBenchSwarm\runs` on 2026-07-24 18:59. 53 tick runs, 912 gesture rows, 135 coldstart rows. Warmup ticks and -1 sentinels excluded; tick budget 16.667 ms.

## Steady-state policy inference (game runs, all waves pooled)

| backend | model | prec | runs | ticks | inf mean | p50 | p95 | p99 | miss | miss% | render mean | tick p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu | medium | f32 | 20 | 24031 | 0.101 | 0.081 | 0.219 | 0.347 | 0 | 0.00% | 5.667 | 0.293 |
| cpu | xlarge | f32 | 15 | 37425 | 5.539 | 5.381 | 9.499 | 11.485 | 23 | 0.06% | 5.176 | 9.578 |
| cpu | xlarge | int8 | 2 | 5641 | 5.736 | 5.969 | 8.530 | 9.801 | 1 | 0.02% | 5.127 | 8.582 |
| gpu | xlarge | f32 | 6 | 17048 | 7.380 | 7.408 | 9.767 | 11.277 | 11 | 0.06% | 4.708 | 9.796 |
| npu | xlarge | f32 | 8 | 26561 | 0.952 | 0.919 | 1.220 | 1.508 | 0 | 0.00% | 5.950 | 1.235 |
| npu | xlarge | int8 | 2 | 5588 | 0.927 | 0.891 | 1.249 | 1.543 | 0 | 0.00% | 5.854 | 1.263 |

## Cold start (phase=cold rows; ms, mean over runs, -1 = failed excluded)

| backend | model | prec | n | model load | delegate create | interp create | first infer | second infer |
|---|---|---|---|---|---|---|---|---|
| cpu | medium | f32 | 20 | 0.308 | 0.000 | 1.029 | 1.622 | 0.061 |
| cpu | xlarge | f32 | 15 | 5.185 | 0.000 | 0.817 | 9.420 | 2.075 |
| cpu | xlarge | int8 | 2 | 1.014 | 0.000 | 1.591 | 4.654 | 4.492 |
| cpu | gesture | f32 | 37 | 0.229 | 0.000 | 0.897 | 1.271 | 0.129 |
| cpu | gesture | int8 | 2 | 0.247 | 0.000 | 1.690 | 0.153 | 0.147 |
| gpu | xlarge | f32 | 7 | 5.990 | 0.007 | 35.743 | 8.725 | 5.833 |
| npu | xlarge | f32 | 8 | 4.129 | 2.805 | 113.971 | 2.994 | 1.048 |
| npu | xlarge | int8 | 4 | 1.752 | 1.726 | 105.524 | 3.732 | 2.117 |
| npu | gesture | f32 | 8 | 0.090 | 0.003 | 94.759 | 2.530 | 1.079 |
| npu | gesture | int8 | 2 | 0.082 | 0.003 | 109.409 | 2.197 | 0.891 |

## Micro-bench steady state (batch 32, 200 iters; ms, mean over runs)

| backend | model | prec | n | steady mean | p50 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| cpu | medium | f32 | 3 | 0.0449 | 0.0420 | 0.0542 | 0.1176 |
| cpu | large | f32 | 1 | 0.1310 | 0.1120 | 0.2452 | 0.2791 |
| cpu | l192 | f32 | 1 | 0.2145 | 0.1702 | 0.3995 | 0.5552 |
| cpu | l192 | int8 | 1 | 0.1214 | 0.0880 | 0.1983 | 0.4170 |
| cpu | l256 | f32 | 1 | 1.0159 | 0.8765 | 2.2979 | 3.1513 |
| cpu | l256 | int8 | 1 | 0.1445 | 0.1163 | 0.2249 | 0.3452 |
| cpu | xlarge | f32 | 1 | 2.1458 | 1.8151 | 3.9821 | 5.8311 |
| cpu | xlarge | int8 | 1 | 0.3507 | 0.3086 | 0.7050 | 0.7832 |
| gpu | medium | f32 | 1 | 0.7950 | 0.6991 | 1.2686 | 1.4922 |
| gpu | large | f32 | 1 | 1.2719 | 1.2483 | 1.7532 | 2.2910 |
| gpu | l192 | f32 | 1 | 1.3123 | 1.2902 | 1.8169 | 2.4523 |
| gpu | l192 | int8 | 1 | 1.2570 | 1.2005 | 1.7337 | 2.8258 |
| gpu | l256 | f32 | 1 | 1.7338 | 1.6760 | 2.1895 | 2.6701 |
| gpu | l256 | int8 | 1 | 1.7289 | 1.6473 | 2.5000 | 3.6371 |
| gpu | xlarge | f32 | 1 | 2.1895 | 1.9978 | 2.8491 | 4.5891 |
| gpu | xlarge | int8 | 1 | 3.1223 | 3.0134 | 4.4531 | 5.0509 |
| npu | medium | f32 | 3 | 0.4729 | 0.4626 | 0.5694 | 0.6899 |
| npu | large | f32 | 1 | 0.5215 | 0.5079 | 0.6339 | 0.7769 |
| npu | l192 | f32 | 1 | 0.5452 | 0.5490 | 0.6247 | 0.6642 |
| npu | l192 | int8 | 1 | 0.5215 | 0.5038 | 0.6314 | 0.8535 |
| npu | l256 | f32 | 1 | 0.5138 | 0.4987 | 0.6218 | 0.8880 |
| npu | l256 | int8 | 1 | 0.5353 | 0.5160 | 0.8232 | 0.9063 |
| npu | xlarge | f32 | 1 | 0.5165 | 0.5051 | 0.6087 | 0.6417 |
| npu | xlarge | int8 | 1 | 0.5730 | 0.5571 | 0.6873 | 1.0041 |

## Gestures (per backend x precision; -1 up->effect = rejected, excluded)

| backend | prec | strokes | accepted | infer mean | infer p95 | up->effect mean | up->effect p95 |
|---|---|---|---|---|---|---|---|
| cpu | f32 | 682 | 549 | 0.187 | 0.654 | 6.91 | 11.61 |
| cpu | int8 | 62 | 60 | 0.103 | 0.142 | 6.99 | 11.49 |
| npu | f32 | 105 | 96 | 0.971 | 1.208 | 7.25 | 10.98 |
| npu | int8 | 63 | 60 | 0.943 | 1.156 | 6.68 | 10.64 |
