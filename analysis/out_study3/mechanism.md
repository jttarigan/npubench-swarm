# Study 3: reproducing the device outputs on the Mac (`analysis/study3_mechanism.py`)

Inputs: the 123,354 enemy rows of the shared XL replay; 370,062 output values per run.

## int8: share of output codes reproduced exactly

| requantization | tanh rounding | CPU INT8 (all five Android devices) | MediaTek NPU INT8 | Qualcomm NPU INT8 (qti-default) | Mac LiteRT int8 |
|---|---|---|---|---|---|
| fp32_rne | rne | 98.7413 % | 98.7416 % | 59.1482 % | 100.0000 % |
| fp32_rne | away | 98.7413 % | 98.7416 % | 59.1482 % | 100.0000 % |
| fp32_away | rne | 98.7413 % | 98.7416 % | 59.1482 % | 100.0000 % |
| fp32_away | away | 98.7413 % | 98.7416 % | 59.1482 % | 100.0000 % |
| gemmlowp | rne | 100.0000 % | 99.9997 % | 59.1377 % | 98.7413 % |
| gemmlowp | away | 100.0000 % | 99.9997 % | 59.1377 % | 98.7413 % |
| rndnu | rne | 98.7413 % | 98.7416 % | 59.1482 % | 100.0000 % |
| rndnu | away | 98.7413 % | 98.7416 % | 59.1482 % | 100.0000 % |
| f64_rne | rne | 98.7413 % | 98.7416 % | 59.1482 % | 100.0000 % |
| f64_rne | away | 98.7413 % | 98.7416 % | 59.1482 % | 100.0000 % |

Fused last layer (tanh applied to the dequantized accumulator, no int8 step before it), hidden layers requantized as named:

| hidden requantization | CPU INT8 (all five Android devices) | MediaTek NPU INT8 | Qualcomm NPU INT8 (qti-default) | Mac LiteRT int8 |
|---|---|---|---|---|
| gemmlowp | 59.1093 % | 59.1090 % | 91.0107 % | 59.1193 % |
| fp32_rne | 59.1060 % | 59.1058 % | 92.3280 % | 59.1166 % |

## float32 model: error against exact (float64) arithmetic

20,000 rows sampled from the replay (seed 0). An emulation explains a device when its error has the same size; summation order inside a GPU or NPU kernel is unknown, so bit-exact matches are not expected except for the MediaTek NPU. 'on fp16 grid' is the share of output values exactly representable in fp16 (1.000 = final tensor stored in fp16).

| run | mean abs err | p99 abs err | max abs err | on fp16 grid | bit-exact with |
|---|---|---|---|---|---|
| CPU F32 (all five Android devices) | 4.95e-08 | 2.51e-07 | 6.07e-07 | 0.208 |  |
| Mali-G72 GPU | 9.94e-05 | 8.07e-04 | 4.77e-03 | 0.189 |  |
| Mali-G615 GPU | 9.94e-05 | 8.07e-04 | 4.77e-03 | 0.189 |  |
| MediaTek NPU F32 | 1.29e-04 | 6.81e-04 | 4.05e-03 | 1.000 |  |
| Adreno 650 GPU | 2.49e-04 | 1.57e-03 | 1.03e-02 | 0.169 |  |
| Qualcomm NPU F32 (qti-default) | 6.54e-04 | 4.48e-03 | 1.45e-02 | 1.000 |  |
| Mac LiteRT f32 | 7.24e-08 | 4.39e-07 | 1.03e-06 | 0.209 |  |
| Xclipse 540 GPU | 1.64e-04 | 7.17e-04 | 2.63e-03 | 0.199 |  |
| Metal GPU (A14) | 3.69e-04 | 2.64e-03 | 6.66e-03 | 1.000 |  |
| Metal GPU (A19 Pro) | 1.43e-04 | 8.16e-04 | 4.14e-03 | 1.000 |  |
| Core ML F32 | 3.31e-04 | 1.64e-03 | 5.17e-03 | 1.000 |  |
| Adreno 840 GPU | 2.30e-04 | 1.27e-03 | 8.63e-03 | 0.169 |  |
| emulation: f16 weights and activations, f32 accumulation, f32 output | 8.56e-05 | 6.48e-04 | 4.08e-03 | 0.132 | 17.04 % (Mali-G72 GPU) |
| emulation: f16 weights and activations, f32 accumulation, f16 output | 1.29e-04 | 6.83e-04 | 4.05e-03 | 1.000 | 95.58 % (MediaTek NPU F32) |
| emulation: f16 weights and activations, f16 accumulation every 64 inputs, f32 output | 1.37e-04 | 1.06e-03 | 4.93e-03 | 0.132 | 16.76 % (Mali-G72 GPU) |
| emulation: f16 weights and activations, f16 accumulation every 16 inputs, f32 output | 1.99e-04 | 1.54e-03 | 5.16e-03 | 0.132 | 16.40 % (Mali-G72 GPU) |
| emulation: f16 weights and activations, f16 accumulation every 4 inputs, f32 output | 3.47e-04 | 2.62e-03 | 6.52e-03 | 0.132 | 16.07 % (Mali-G72 GPU) |
| emulation: f16 weights and activations, f16 accumulation every 1 inputs, f16 output | 5.14e-04 | 3.69e-03 | 9.07e-03 | 1.000 | 55.47 % (Metal GPU (A19 Pro)) |
