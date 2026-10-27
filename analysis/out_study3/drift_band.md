# Closed-loop drift per path (`analysis/study3_drift_band.py`)

First accelerator game against the first CPU game of the same device, XL model, per-second mean distance between matching live enemies (the Figure 2 measure).

| path | seconds compared | min | max | first second above 0.03 px | seconds above 0.03 px (value) |
|---|---|---|---|---|---|
| Mali-G72 GPU F32 (Tab S6 Lite) | 62 | 0.0010 | 0.007 | never | none |
| Mali-G615 GPU F32 (POCO) | 62 | 0.0010 | 0.007 | never | none |
| MediaTek NPU F32 (POCO) | 62 | 0.0011 | 0.008 | never | none |
| Xclipse 540 GPU F32 (A56) | 62 | 0.0039 | 0.017 | never | none |
| Adreno 650 GPU F32 (Tab S7+) | 62 | 0.0046 | 0.026 | never | none |
| Adreno 840 GPU F32 (Fold 8) | 62 | 0.0046 | 0.025 | never | none |
| Qualcomm NNAPI F32 (Tab S7+) | 62 | 0.0072 | 274.905 | 42 s | 42 to 61 s (275) |
| Qualcomm NNAPI INT8 (Tab S7+) | 61 | 0.0487 | 326.045 | 0 s | 0 to 60 s (326) |
| Metal GPU F32 (iPhone 12) | 62 | 0.0036 | 0.014 | never | none |
| Core ML F32 (iPhone 12) | 62 | 0.0082 | 274.904 | 3 s | 3 s (0.0302); 9 to 10 s (0.0411); 35 s (0.0316); 42 to 61 s (275) |

Wave starts of the CPU game (s): 4.7, 10.8, 17.1, 23.5, 29.9, 36.2, 42.5, 49.0, 55.5.
