# NPU Bench Swarm — run summary

Generated from `E:\OneDrive\MyProject\NPUBenchSwarm\runs` on 2026-07-24 17:14. 32 tick runs, 603 gesture rows, 77 coldstart rows. Warmup ticks and -1 sentinels excluded; tick budget 16.667 ms.

## Steady-state policy inference (game runs, all waves pooled)

| backend | model | prec | runs | ticks | inf mean | p50 | p95 | p99 | miss | miss% | render mean | tick p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu | medium | f32 | 19 | 21272 | 0.096 | 0.078 | 0.205 | 0.332 | 0 | 0.00% | 5.586 | 0.264 |
| cpu | xlarge | f32 | 8 | 12070 | 4.512 | 4.162 | 8.362 | 9.931 | 1 | 0.01% | 5.699 | 8.432 |
| cpu | xlarge | int8 | 1 | 2883 | 6.304 | 6.356 | 8.602 | 9.663 | 1 | 0.03% | 5.147 | 8.655 |
| npu | xlarge | f32 | 3 | 8441 | 0.984 | 0.951 | 1.313 | 1.598 | 0 | 0.00% | 6.004 | 1.327 |
| npu | xlarge | int8 | 1 | 2882 | 0.972 | 0.939 | 1.308 | 1.517 | 0 | 0.00% | 6.095 | 1.325 |

## Cold start (phase=cold rows; ms, mean over runs, -1 = failed excluded)

| backend | model | prec | n | model load | delegate create | interp create | first infer | second infer |
|---|---|---|---|---|---|---|---|---|
| cpu | medium | f32 | 19 | 0.293 | 0.000 | 0.912 | 1.608 | 0.061 |
| cpu | xlarge | f32 | 8 | 5.531 | 0.000 | 0.878 | 9.789 | 2.688 |
| cpu | xlarge | int8 | 1 | 1.016 | 0.000 | 2.232 | 6.356 | 4.317 |
| cpu | gesture | f32 | 23 | 0.249 | 0.000 | 1.055 | 1.328 | 0.131 |
| cpu | gesture | int8 | 1 | 0.260 | 0.000 | 2.765 | 0.177 | 0.171 |
| npu | xlarge | f32 | 3 | 5.211 | 7.443 | 152.997 | 3.427 | 1.364 |
| npu | xlarge | int8 | 3 | 2.155 | 2.298 | 116.771 | 3.691 | 0.798 |
| npu | gesture | f32 | 3 | 0.098 | 0.003 | 96.953 | 2.530 | 1.079 |
| npu | gesture | int8 | 1 | 0.084 | 0.003 | 111.593 | 2.438 | 0.987 |

## Micro-bench steady state (batch 32, 200 iters; ms, mean over runs)

| backend | model | prec | n | steady mean | p50 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| cpu | medium | f32 | 3 | 0.0449 | 0.0420 | 0.0542 | 0.1176 |
| cpu | large | f32 | 1 | 0.1310 | 0.1120 | 0.2452 | 0.2791 |
| cpu | xlarge | f32 | 1 | 2.1458 | 1.8151 | 3.9821 | 5.8311 |
| gpu | medium | f32 | 1 | 0.7950 | 0.6991 | 1.2686 | 1.4922 |
| gpu | large | f32 | 1 | 1.2719 | 1.2483 | 1.7532 | 2.2910 |
| gpu | xlarge | f32 | 1 | 2.1895 | 1.9978 | 2.8491 | 4.5891 |
| npu | medium | f32 | 3 | 0.4729 | 0.4626 | 0.5694 | 0.6899 |
| npu | large | f32 | 1 | 0.5215 | 0.5079 | 0.6339 | 0.7769 |
| npu | xlarge | f32 | 1 | 0.5165 | 0.5051 | 0.6087 | 0.6417 |

## Gestures (per backend x precision; -1 up->effect = rejected, excluded)

| backend | prec | strokes | accepted | infer mean | infer p95 | up->effect mean | up->effect p95 |
|---|---|---|---|---|---|---|---|
| cpu | f32 | 439 | 348 | 0.201 | 0.714 | 7.00 | 11.97 |
| cpu | int8 | 28 | 28 | 0.101 | 0.156 | 7.60 | 11.39 |
| npu | f32 | 105 | 96 | 0.971 | 1.208 | 7.25 | 10.98 |
| npu | int8 | 31 | 29 | 0.997 | 1.185 | 6.87 | 10.99 |
