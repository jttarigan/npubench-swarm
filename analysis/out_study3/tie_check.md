# Int8 claims traced to the emulation (`analysis/study3_tie_check.py`)

## C8: the one MediaTek NPU int8 value that the two-rounding emulation misses

Emulation (gemmlowp requantization, round-to-nearest-even tanh) against the Android CPU int8 codes: 100.0000 % equal; against the MediaTek NPU: 99.9997 % equal, 1 value(s) differ.
- row 14191, output 1: emulation and CPU code 121, MediaTek code 120 (CPU code 121).
  - layer 1 (512 units): exact ties in the first rounding: 0 []; in the second rounding: 8 [35, 85, 118, 155, 250, 372, 402, 448].
  - layer 2 (512 units): exact ties in the first rounding: 0 []; in the second rounding: 0 [].
  - layer 3 (512 units): exact ties in the first rounding: 0 []; in the second rounding: 2 [80, 504].
  - layer 4 (3 units): exact ties in the first rounding: 0 []; in the second rounding: 1 [1].
  - tanh input code 39, tanh output before rounding 121.430256 (fraction 0.430256, distance from the .5 tie 0.069744); round-to-nearest gives 121, the MediaTek code is 120.

Last layer (shift -12: left 0, right 12), all 123,354 rows x 3 units: 0 exact ties in the first rounding, 85 in the second (RoundingDivideByPOT). Rounding the second-rounding ties the other way changes the output code at 51 of them; there the MediaTek codes equal the emulation (ties away from zero) at 50 and the other side at 1; round half to even would give the MediaTek code at 29 of the 51.

## C2: distance of the fused-tanh emulation from the Qualcomm NNAPI int8 codes

| hidden requantization | equal | within 1 step | within 2 steps | largest distance (steps) |
|---|---|---|---|---|
| gemmlowp | 91.01 % | 99.92 % | 99.998 % | 3 |
| fp32_rne | 92.33 % | 99.98 % | 100.000 % | 2 |
