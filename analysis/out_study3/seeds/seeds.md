# Study 3: closed-loop comparison over many seeds (`analysis/study3_seeds.py`)

Host co-simulation, 100 seeds per model (index 0 = the study's game). Each peer is compared with the reference of its group (float32: LiteRT XNNPACK; int8: TFLite reference kernels) by per-tick state hashes. Outcome = length, kills and final hp. 95 % Wilson intervals.

## xlarge

Reference games: 100 seeds, 3634 to 3781 ticks, final hp 6 to 7, waves reached 10 to 10.

| group | peer | seeds | games differing | outcome changed (95 % CI) | hp changed | first differing tick, median | guarded rows |
|---|---|---|---|---|---|---|---|
| f32 | np_f32 | 100 | 100 | 0 (0 %, 0 to 4 %) | 0 | 9 |  |
| f32 | np_f16 | 100 | 100 | 10 (10 %, 6 to 17 %) | 3 | 0 |  |
| f32 | np_f16acc | 100 | 100 | 13 (13 %, 8 to 21 %) | 4 | 0 |  |
| f32_guard | lite_f32 | 100 | 0 | 0 (0 %, 0 to 4 %) | 0 |  | 22.2 % |
| f32_guard | np_f32 | 100 | 0 | 0 (0 %, 0 to 4 %) | 0 |  | 22.2 % |
| f32_guard | np_f16 | 100 | 0 | 0 (0 %, 0 to 4 %) | 0 |  | 22.1 % |
| f32_guard | np_f16acc | 100 | 0 | 0 (0 %, 0 to 4 %) | 0 |  | 22.1 % |
| int8 | lite_int8 | 100 | 100 | 64 (64 %, 54 to 73 %) | 14 | 3 |  |

## tiny

Reference games: 100 seeds, 4145 to 4623 ticks, final hp 5 to 6, waves reached 10 to 10.

| group | peer | seeds | games differing | outcome changed (95 % CI) | hp changed | first differing tick, median | guarded rows |
|---|---|---|---|---|---|---|---|
| f32 | np_f32 | 100 | 100 | 0 (0 %, 0 to 4 %) | 0 | 12 |  |
| f32 | np_f16 | 100 | 100 | 38 (38 %, 29 to 48 %) | 19 | 0 |  |
| f32 | np_f16acc | 100 | 100 | 34 (34 %, 25 to 44 %) | 18 | 0 |  |
| f32_guard | lite_f32 | 100 | 0 | 0 (0 %, 0 to 4 %) | 0 |  | 38.6 % |
| f32_guard | np_f32 | 100 | 0 | 0 (0 %, 0 to 4 %) | 0 |  | 38.6 % |
| f32_guard | np_f16 | 100 | 0 | 0 (0 %, 0 to 4 %) | 0 |  | 38.4 % |
| f32_guard | np_f16acc | 100 | 0 | 0 (0 %, 0 to 4 %) | 0 |  | 38.4 % |
| int8 | lite_int8 | 100 | 100 | 68 (68 %, 58 to 76 %) | 20 | 9 |  |

