# NPU Bench Swarm — run summary

Generated from `E:\OneDrive\MyProject\NPUBenchSwarm\runs` on 2026-07-25 14:04. 66 tick runs, 1306 gesture rows, 161 coldstart rows. Warmup ticks and -1 sentinels excluded; tick budget 16.667 ms.

## Steady-state policy inference (game runs, all waves pooled)

| backend | model | prec | runs | ticks | inf mean | p50 | p95 | p99 | miss | miss% | render mean | tick p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu | medium | f32 | 20 | 24031 | 0.101 | 0.081 | 0.219 | 0.347 | 0 | 0.00% | 5.667 | 0.293 |
| cpu | xlarge | f32 | 19 | 48553 | 5.380 | 5.116 | 9.402 | 11.493 | 40 | 0.08% | 5.259 | 9.486 |
| cpu | xlarge | int8 | 5 | 14812 | 6.180 | 6.371 | 8.561 | 9.666 | 3 | 0.02% | 5.104 | 8.608 |
| gpu | xlarge | f32 | 6 | 17048 | 7.380 | 7.408 | 9.767 | 11.277 | 11 | 0.06% | 4.708 | 9.796 |
| npu | xlarge | f32 | 8 | 26561 | 0.952 | 0.919 | 1.220 | 1.508 | 0 | 0.00% | 5.950 | 1.235 |
| npu | xlarge | int8 | 8 | 23730 | 0.894 | 0.860 | 1.185 | 1.493 | 0 | 0.00% | 5.600 | 1.201 |

## Cold start (phase=cold rows; ms, mean over runs, -1 = failed excluded)

| backend | model | prec | n | model load | delegate create | interp create | first infer | second infer |
|---|---|---|---|---|---|---|---|---|
| cpu | medium | f32 | 20 | 0.308 | 0.000 | 1.029 | 1.622 | 0.061 |
| cpu | xlarge | f32 | 19 | 5.597 | 0.000 | 0.782 | 9.880 | 1.935 |
| cpu | xlarge | int8 | 5 | 1.128 | 0.000 | 2.761 | 6.548 | 4.862 |
| cpu | gesture | f32 | 41 | 0.230 | 0.000 | 0.885 | 1.318 | 0.133 |
| cpu | gesture | int8 | 5 | 0.250 | 0.000 | 1.545 | 0.154 | 0.130 |
| gpu | xlarge | f32 | 7 | 5.990 | 0.007 | 35.743 | 8.725 | 5.833 |
| npu | xlarge | f32 | 8 | 4.129 | 2.805 | 113.971 | 2.994 | 1.048 |
| npu | xlarge | int8 | 10 | 1.283 | 1.686 | 101.059 | 3.430 | 1.230 |
| npu | gesture | f32 | 8 | 0.090 | 0.003 | 94.759 | 2.530 | 1.079 |
| npu | gesture | int8 | 8 | 0.080 | 0.003 | 101.760 | 2.214 | 0.863 |

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
| cpu | f32 | 835 | 683 | 0.183 | 0.598 | 6.94 | 11.70 |
| cpu | int8 | 146 | 133 | 0.106 | 0.143 | 6.99 | 11.34 |
| npu | f32 | 105 | 96 | 0.971 | 1.208 | 7.25 | 10.98 |
| npu | int8 | 220 | 197 | 0.930 | 1.294 | 6.95 | 10.84 |
