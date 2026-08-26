# Posit and takum GPU storage-conversion results

This report answers the four questions fixed before implementation. The run used one NVIDIA H200 NVL GPU, scalar x1 access, 30 timing samples after 10 warm-ups, DOT with N = 2^27, and GEMV with M = 1024 and N = 65,536. Ratios below are alternative time divided by reference time. Lower is faster. The brackets give the 95% bootstrap confidence interval. `Equivalent` means the full interval lies inside [0.97, 1.03].

## Case tables

### Question 1: low-bit strategy winner

Every cell reports the case winner, its median kernel time, and its paired ratio to the direct decoder.

| Format | Arithmetic | Field DOT | Field GEMV | Log-uniform DOT | Log-uniform GEMV |
| --- | --- | --- | --- | --- | --- |
| posit<8,0> | FP32 | shared LUT, 0.489 ms; 0.362 [0.362, 0.363], faster vs direct | shared LUT, 0.116 ms; 0.208 [0.208, 0.209], faster vs direct | shared LUT, 0.487 ms; 0.361 [0.360, 0.362], faster vs direct | shared LUT, 0.117 ms; 0.210 [0.210, 0.211], faster vs direct |
| posit<8,0> | FP64 | shared LUT, 0.489 ms; 0.361 [0.360, 0.361], faster vs direct | shared LUT, 0.117 ms; 0.208 [0.208, 0.209], faster vs direct | shared LUT, 0.498 ms; 0.367 [0.366, 0.367], faster vs direct | shared LUT, 0.134 ms; 0.239 [0.237, 0.240], faster vs direct |
| takum<8> | FP32 | shared LUT, 0.489 ms; 0.393 [0.392, 0.393], faster vs direct | shared LUT, 0.116 ms; 0.230 [0.229, 0.231], faster vs direct | shared LUT, 0.487 ms; 0.343 [0.342, 0.343], faster vs direct | shared LUT, 0.117 ms; 0.198 [0.197, 0.199], faster vs direct |
| takum<8> | FP64 | shared LUT, 0.489 ms; 0.392 [0.392, 0.393], faster vs direct | shared LUT, 0.117 ms; 0.230 [0.230, 0.231], faster vs direct | shared LUT, 0.497 ms; 0.399 [0.398, 0.399], faster vs direct | shared LUT, 0.131 ms; 0.260 [0.259, 0.261], faster vs direct |
| takum_log<8> | FP32 | shared LUT, 0.489 ms; 0.248 [0.248, 0.249], faster vs direct | shared LUT, 0.116 ms; 0.159 [0.158, 0.159], faster vs direct | shared LUT, 0.487 ms; 0.199 [0.198, 0.199], faster vs direct | shared LUT, 0.117 ms; 0.141 [0.141, 0.141], faster vs direct |
| takum_log<8> | FP64 | shared LUT, 0.489 ms; 0.241 [0.241, 0.241], faster vs direct | shared LUT, 0.117 ms; 0.154 [0.153, 0.154], faster vs direct | shared LUT, 0.497 ms; 0.209 [0.209, 0.210], faster vs direct | shared LUT, 0.131 ms; 0.161 [0.160, 0.162], faster vs direct |
| posit<14,1> | FP32 | global LUT, 0.525 ms; 0.374 [0.373, 0.374], faster vs direct | global LUT, 0.189 ms; 0.297 [0.296, 0.298], faster vs direct | global LUT, 0.562 ms; 0.400 [0.400, 0.401], faster vs direct | global LUT, 0.216 ms; 0.339 [0.339, 0.340], faster vs direct |
| posit<14,1> | FP64 | global LUT, 0.534 ms; 0.381 [0.380, 0.381], faster vs direct | global LUT, 0.197 ms; 0.302 [0.301, 0.304], faster vs direct | global LUT, 0.584 ms; 0.416 [0.416, 0.418], faster vs direct | global LUT, 0.244 ms; 0.373 [0.372, 0.374], faster vs direct |
| takum<14> | FP32 | global LUT, 0.525 ms; 0.423 [0.422, 0.423], faster vs direct | global LUT, 0.184 ms; 0.347 [0.343, 0.349], faster vs direct | global LUT, 0.590 ms; 0.416 [0.415, 0.416], faster vs direct | global LUT, 0.270 ms; 0.430 [0.429, 0.430], faster vs direct |
| takum<14> | FP64 | global LUT, 0.533 ms; 0.433 [0.432, 0.433], faster vs direct | global LUT, 0.205 ms; 0.380 [0.363, 0.382], faster vs direct | global LUT, 0.609 ms; 0.494 [0.492, 0.495], faster vs direct | global LUT, 0.282 ms; 0.522 [0.521, 0.523], faster vs direct |
| takum_log<14> | FP32 | global LUT, 0.526 ms; 0.239 [0.237, 0.240], faster vs direct | global LUT, 0.185 ms; 0.206 [0.203, 0.207], faster vs direct | global LUT, 0.588 ms; 0.239 [0.239, 0.239], faster vs direct | global LUT, 0.268 ms; 0.238 [0.237, 0.238], faster vs direct |
| takum_log<14> | FP64 | global LUT, 0.534 ms; 0.235 [0.234, 0.235], faster vs direct | global LUT, 0.203 ms; 0.215 [0.206, 0.215], faster vs direct | global LUT, 0.610 ms; 0.244 [0.243, 0.244], faster vs direct | global LUT, 0.281 ms; 0.238 [0.237, 0.238], faster vs direct |

### Question 2: alternative LUT divided by IEEE LUT

These controls reuse the same raw index stream and compiled table-only kernel. Only table contents change. Each row covers one alternative family, width, arithmetic type, trace, and table placement.

| Bits | Arithmetic | Trace | Placement | Family | DOT | GEMV |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | FP32 | scattered | shared LUT | posit<8,0> | 1.001 [1.000, 1.001], equivalent | 1.001 [0.996, 1.005], equivalent |
| 8 | FP32 | scattered | shared LUT | takum<8> | 1.001 [1.000, 1.002], equivalent | 1.002 [0.999, 1.004], equivalent |
| 8 | FP32 | scattered | shared LUT | takum_log<8> | 1.001 [1.000, 1.002], equivalent | 0.998 [0.994, 1.002], equivalent |
| 8 | FP32 | scattered | global LUT | posit<8,0> | 1.000 [0.999, 1.001], equivalent | 0.999 [0.997, 1.005], equivalent |
| 8 | FP32 | scattered | global LUT | takum<8> | 1.000 [0.999, 1.002], equivalent | 1.003 [0.999, 1.004], equivalent |
| 8 | FP32 | scattered | global LUT | takum_log<8> | 0.999 [0.998, 1.001], equivalent | 1.000 [0.995, 1.004], equivalent |
| 8 | FP32 | concentrated | shared LUT | posit<8,0> | 1.000 [0.999, 1.002], equivalent | 1.000 [0.998, 1.002], equivalent |
| 8 | FP32 | concentrated | shared LUT | takum<8> | 1.000 [0.999, 1.002], equivalent | 0.999 [0.996, 1.003], equivalent |
| 8 | FP32 | concentrated | shared LUT | takum_log<8> | 1.000 [0.998, 1.001], equivalent | 1.000 [0.997, 1.002], equivalent |
| 8 | FP32 | concentrated | global LUT | posit<8,0> | 1.001 [0.999, 1.003], equivalent | 0.999 [0.997, 1.003], equivalent |
| 8 | FP32 | concentrated | global LUT | takum<8> | 1.000 [0.999, 1.002], equivalent | 1.001 [0.996, 1.003], equivalent |
| 8 | FP32 | concentrated | global LUT | takum_log<8> | 1.001 [0.999, 1.003], equivalent | 0.998 [0.996, 1.002], equivalent |
| 8 | FP64 | scattered | shared LUT | posit<8,0> | 0.999 [0.998, 1.001], equivalent | 1.001 [0.998, 1.003], equivalent |
| 8 | FP64 | scattered | shared LUT | takum<8> | 1.000 [0.998, 1.000], equivalent | 1.000 [0.997, 1.002], equivalent |
| 8 | FP64 | scattered | shared LUT | takum_log<8> | 0.999 [0.998, 1.000], equivalent | 0.998 [0.994, 1.002], equivalent |
| 8 | FP64 | scattered | global LUT | posit<8,0> | 0.999 [0.998, 1.001], equivalent | 0.998 [0.996, 1.004], equivalent |
| 8 | FP64 | scattered | global LUT | takum<8> | 0.999 [0.997, 1.001], equivalent | 1.001 [0.999, 1.002], equivalent |
| 8 | FP64 | scattered | global LUT | takum_log<8> | 1.000 [0.999, 1.001], equivalent | 0.998 [0.994, 1.003], equivalent |
| 8 | FP64 | concentrated | shared LUT | posit<8,0> | 1.000 [0.999, 1.001], equivalent | 0.997 [0.995, 1.005], equivalent |
| 8 | FP64 | concentrated | shared LUT | takum<8> | 1.001 [0.999, 1.003], equivalent | 1.000 [0.997, 1.002], equivalent |
| 8 | FP64 | concentrated | shared LUT | takum_log<8> | 1.000 [0.997, 1.001], equivalent | 0.997 [0.995, 1.000], equivalent |
| 8 | FP64 | concentrated | global LUT | posit<8,0> | 1.002 [1.000, 1.003], equivalent | 1.001 [0.998, 1.002], equivalent |
| 8 | FP64 | concentrated | global LUT | takum<8> | 1.001 [0.999, 1.003], equivalent | 1.001 [1.000, 1.005], equivalent |
| 8 | FP64 | concentrated | global LUT | takum_log<8> | 1.001 [1.000, 1.002], equivalent | 1.001 [1.000, 1.004], equivalent |
| 14 | FP32 | scattered | shared LUT | posit<14,1> | 1.000 [0.999, 1.001], equivalent | 0.999 [0.998, 1.001], equivalent |
| 14 | FP32 | scattered | shared LUT | takum<14> | 0.999 [0.998, 1.000], equivalent | 0.998 [0.996, 1.002], equivalent |
| 14 | FP32 | scattered | shared LUT | takum_log<14> | 0.999 [0.998, 1.001], equivalent | 1.003 [1.000, 1.004], equivalent |
| 14 | FP32 | scattered | global LUT | posit<14,1> | 1.001 [1.000, 1.004], equivalent | 0.989 [0.985, 0.996], equivalent |
| 14 | FP32 | scattered | global LUT | takum<14> | 1.000 [0.998, 1.002], equivalent | 0.999 [0.996, 1.004], equivalent |
| 14 | FP32 | scattered | global LUT | takum_log<14> | 0.998 [0.995, 1.001], equivalent | 1.002 [0.996, 1.003], equivalent |
| 14 | FP32 | concentrated | shared LUT | posit<14,1> | 1.000 [0.998, 1.001], equivalent | 1.000 [0.998, 1.002], equivalent |
| 14 | FP32 | concentrated | shared LUT | takum<14> | 1.000 [0.997, 1.001], equivalent | 0.998 [0.997, 1.002], equivalent |
| 14 | FP32 | concentrated | shared LUT | takum_log<14> | 1.000 [0.999, 1.002], equivalent | 1.000 [0.998, 1.004], equivalent |
| 14 | FP32 | concentrated | global LUT | posit<14,1> | 1.000 [0.999, 1.002], equivalent | 1.000 [0.994, 1.003], equivalent |
| 14 | FP32 | concentrated | global LUT | takum<14> | 1.000 [0.999, 1.002], equivalent | 1.000 [0.996, 1.002], equivalent |
| 14 | FP32 | concentrated | global LUT | takum_log<14> | 0.999 [0.997, 1.001], equivalent | 0.999 [0.994, 1.002], equivalent |
| 14 | FP64 | scattered | shared LUT | posit<14,1> | 1.000 [0.999, 1.002], equivalent | 0.999 [0.998, 1.002], equivalent |
| 14 | FP64 | scattered | shared LUT | takum<14> | 1.001 [1.000, 1.001], equivalent | 1.000 [0.998, 1.001], equivalent |
| 14 | FP64 | scattered | shared LUT | takum_log<14> | 1.000 [0.999, 1.001], equivalent | 1.000 [0.999, 1.001], equivalent |
| 14 | FP64 | scattered | global LUT | posit<14,1> | 1.001 [0.999, 1.003], equivalent | 1.009 [1.003, 1.012], equivalent |
| 14 | FP64 | scattered | global LUT | takum<14> | 1.001 [0.999, 1.002], equivalent | 1.001 [0.993, 1.007], equivalent |
| 14 | FP64 | scattered | global LUT | takum_log<14> | 0.998 [0.995, 1.000], equivalent | 1.000 [0.995, 1.005], equivalent |
| 14 | FP64 | concentrated | shared LUT | posit<14,1> | 1.000 [0.999, 1.001], equivalent | 1.000 [0.999, 1.001], equivalent |
| 14 | FP64 | concentrated | shared LUT | takum<14> | 0.999 [0.999, 1.000], equivalent | 1.000 [0.999, 1.001], equivalent |
| 14 | FP64 | concentrated | shared LUT | takum_log<14> | 1.000 [0.999, 1.001], equivalent | 1.000 [1.000, 1.001], equivalent |
| 14 | FP64 | concentrated | global LUT | posit<14,1> | 1.000 [0.999, 1.001], equivalent | 1.000 [0.998, 1.003], equivalent |
| 14 | FP64 | concentrated | global LUT | takum<14> | 1.000 [0.998, 1.002], equivalent | 0.999 [0.996, 1.001], equivalent |
| 14 | FP64 | concentrated | global LUT | takum_log<14> | 1.002 [1.001, 1.003], equivalent | 0.999 [0.997, 1.003], equivalent |

### Question 3: strategy after the shared LUT limit

At 16 bits, the table reports global-LUT time divided by direct-decoder time. At 32 bits, a complete LUT would require 16 GiB for FP32 or 32 GiB for FP64, so direct was the only full decoder tested. The 32-bit cells report its median time.

| 16-bit format | Arithmetic | Field DOT | Field GEMV | Log-uniform DOT | Log-uniform GEMV |
| --- | --- | --- | --- | --- | --- |
| posit<16,1> | FP32 | 0.381 [0.380, 0.381], faster | 0.270 [0.270, 0.271], faster | 0.455 [0.452, 0.458], faster | 0.354 [0.353, 0.355], faster |
| posit<16,1> | FP64 | 0.377 [0.377, 0.378], faster | 0.287 [0.286, 0.287], faster | 0.488 [0.488, 0.489], faster | 0.404 [0.404, 0.406], faster |
| takum<16> | FP32 | 0.425 [0.424, 0.426], faster | 0.384 [0.378, 0.391], faster | 0.455 [0.439, 0.467], faster | 0.415 [0.415, 0.416], faster |
| takum<16> | FP64 | 0.419 [0.418, 0.419], faster | 0.414 [0.407, 0.424], faster | 0.578 [0.577, 0.578], faster | 0.623 [0.621, 0.625], faster |
| takum_log<16> | FP32 | 0.230 [0.230, 0.230], faster | 0.238 [0.233, 0.242], faster | 0.266 [0.265, 0.266], faster | 0.249 [0.249, 0.250], faster |
| takum_log<16> | FP64 | 0.225 [0.225, 0.225], faster | 0.244 [0.239, 0.249], faster | 0.283 [0.282, 0.283], faster | 0.306 [0.305, 0.307], faster |

| 32-bit format | Arithmetic | Field DOT | Field GEMV | Log-uniform DOT | Log-uniform GEMV |
| --- | --- | --- | --- | --- | --- |
| posit<32,2> | FP32 | 1.498 ms | 0.597 ms | 1.694 ms | 0.740 ms |
| posit<32,2> | FP64 | 1.377 ms | 0.577 ms | 1.377 ms | 0.579 ms |
| takum<32> | FP32 | 1.451 ms | 0.572 ms | 2.058 ms | 0.851 ms |
| takum<32> | FP64 | 1.192 ms | 0.475 ms | 1.194 ms | 0.473 ms |
| takum_log<32> | FP32 | 1.574 ms | 0.673 ms | 1.576 ms | 0.674 ms |
| takum_log<32> | FP64 | 1.595 ms | 0.692 ms | 1.595 ms | 0.692 ms |

### Question 4: best alternative divided by fastest retained same-width IEEE

The alternative winner and IEEE reference are selected independently inside each format, arithmetic, distribution, and kernel case. The IEEE reference shown in each cell is the fastest retained format and scalar strategy for that case.

| Format | Arithmetic | Distribution | DOT | GEMV |
| --- | --- | --- | --- | --- |
| posit<8,0> | FP32 | field-balanced | 1.028 [1.026, 1.029], equivalent vs fp8_e5m2 native | 1.027 [1.024, 1.031], inconclusive vs fp8_e5m2 native |
| posit<8,0> | FP32 | paired log-uniform | 1.023 [1.020, 1.025], equivalent vs fp8_e4m3 native | 1.036 [1.034, 1.043], slower vs fp8_e5m2 native |
| posit<8,0> | FP64 | field-balanced | 1.003 [1.001, 1.005], equivalent vs fp8_e5m2 shared LUT | 1.004 [1.000, 1.008], equivalent vs fp8_e5m2 native |
| posit<8,0> | FP64 | paired log-uniform | 1.018 [1.016, 1.019], equivalent vs fp8_e5m2 native | 1.144 [1.139, 1.148], slower vs fp8_e4m3 native |
| takum<8> | FP32 | field-balanced | 1.027 [1.025, 1.028], equivalent vs fp8_e5m2 native | 1.027 [1.022, 1.033], inconclusive vs fp8_e5m2 native |
| takum<8> | FP32 | paired log-uniform | 1.022 [1.020, 1.025], equivalent vs fp8_e4m3 native | 1.039 [1.036, 1.043], slower vs fp8_e5m2 native |
| takum<8> | FP64 | field-balanced | 1.003 [1.000, 1.004], equivalent vs fp8_e5m2 shared LUT | 0.998 [0.995, 1.003], equivalent vs fp8_e5m2 native |
| takum<8> | FP64 | paired log-uniform | 1.016 [1.014, 1.017], equivalent vs fp8_e5m2 native | 1.120 [1.115, 1.125], slower vs fp8_e4m3 native |
| takum_log<8> | FP32 | field-balanced | 1.027 [1.026, 1.029], equivalent vs fp8_e5m2 native | 1.025 [1.022, 1.029], equivalent vs fp8_e5m2 native |
| takum_log<8> | FP32 | paired log-uniform | 1.022 [1.021, 1.025], equivalent vs fp8_e4m3 native | 1.038 [1.034, 1.042], slower vs fp8_e5m2 native |
| takum_log<8> | FP64 | field-balanced | 1.002 [1.000, 1.003], equivalent vs fp8_e5m2 shared LUT | 0.998 [0.993, 1.001], equivalent vs fp8_e5m2 native |
| takum_log<8> | FP64 | paired log-uniform | 1.016 [1.014, 1.018], equivalent vs fp8_e5m2 native | 1.120 [1.116, 1.130], slower vs fp8_e4m3 native |
| posit<14,1> | FP32 | field-balanced | 1.078 [1.076, 1.080], slower vs e8m5 direct branchy | 1.216 [1.208, 1.220], slower vs e8m5 direct branchy |
| posit<14,1> | FP32 | paired log-uniform | 1.155 [1.153, 1.158], slower vs e8m5 direct branchy | 1.375 [1.371, 1.380], slower vs e8m5 direct branchy |
| posit<14,1> | FP64 | field-balanced | 1.089 [1.086, 1.091], slower vs e11m2 direct masked | 1.215 [1.204, 1.221], slower vs e11m2 direct masked |
| posit<14,1> | FP64 | paired log-uniform | 1.193 [1.190, 1.194], slower vs e11m2 direct masked | 1.492 [1.488, 1.495], slower vs e11m2 direct masked |
| takum<14> | FP32 | field-balanced | 1.079 [1.077, 1.081], slower vs e8m5 direct branchy | 1.186 [1.167, 1.190], slower vs e8m5 direct branchy |
| takum<14> | FP32 | paired log-uniform | 1.212 [1.210, 1.214], slower vs e8m5 direct branchy | 1.717 [1.713, 1.724], slower vs e8m5 direct branchy |
| takum<14> | FP64 | field-balanced | 1.088 [1.086, 1.091], slower vs e11m2 direct masked | 1.267 [1.209, 1.271], slower vs e11m2 direct masked |
| takum<14> | FP64 | paired log-uniform | 1.244 [1.243, 1.247], slower vs e11m2 direct masked | 1.728 [1.724, 1.733], slower vs e11m2 direct masked |
| takum_log<14> | FP32 | field-balanced | 1.079 [1.078, 1.081], slower vs e8m5 direct branchy | 1.189 [1.171, 1.194], slower vs e8m5 direct branchy |
| takum_log<14> | FP32 | paired log-uniform | 1.209 [1.207, 1.212], slower vs e8m5 direct branchy | 1.708 [1.704, 1.714], slower vs e8m5 direct branchy |
| takum_log<14> | FP64 | field-balanced | 1.089 [1.086, 1.092], slower vs e11m2 direct masked | 1.257 [1.209, 1.265], slower vs e11m2 direct masked |
| takum_log<14> | FP64 | paired log-uniform | 1.245 [1.243, 1.246], slower vs e11m2 direct masked | 1.723 [1.719, 1.726], slower vs e11m2 direct masked |
| posit<16,1> | FP32 | field-balanced | 1.057 [1.055, 1.059], slower vs bf16_e8m7 native | 1.343 [1.340, 1.345], slower vs bf16_e8m7 native |
| posit<16,1> | FP32 | paired log-uniform | 1.265 [1.255, 1.272], slower vs fp16_e5m10 native | 1.764 [1.759, 1.768], slower vs fp16_e5m10 native |
| posit<16,1> | FP64 | field-balanced | 1.048 [1.047, 1.051], slower vs e11m4 direct masked | 1.458 [1.453, 1.461], slower vs e11m4 direct masked |
| posit<16,1> | FP64 | paired log-uniform | 1.359 [1.357, 1.361], slower vs e11m4 direct masked | 2.058 [2.054, 2.062], slower vs e11m4 direct masked |
| takum<16> | FP32 | field-balanced | 1.055 [1.053, 1.057], slower vs bf16_e8m7 native | 1.644 [1.619, 1.676], slower vs bf16_e8m7 native |
| takum<16> | FP32 | paired log-uniform | 1.298 [1.252, 1.335], slower vs fp16_e5m10 native | 2.122 [2.116, 2.127], slower vs fp16_e5m10 native |
| takum<16> | FP64 | field-balanced | 1.037 [1.035, 1.038], slower vs e11m4 direct masked | 1.761 [1.728, 1.807], slower vs e11m4 direct masked |
| takum<16> | FP64 | paired log-uniform | 1.429 [1.427, 1.433], slower vs e11m4 direct masked | 2.658 [2.650, 2.668], slower vs e11m4 direct masked |
| takum_log<16> | FP32 | field-balanced | 1.055 [1.053, 1.057], slower vs bf16_e8m7 native | 1.675 [1.643, 1.707], slower vs bf16_e8m7 native |
| takum_log<16> | FP32 | paired log-uniform | 1.327 [1.323, 1.331], slower vs fp16_e5m10 native | 2.109 [2.103, 2.115], slower vs fp16_e5m10 native |
| takum_log<16> | FP64 | field-balanced | 1.037 [1.036, 1.039], slower vs e11m4 direct masked | 1.774 [1.731, 1.809], slower vs e11m4 direct masked |
| takum_log<16> | FP64 | paired log-uniform | 1.438 [1.435, 1.441], slower vs e11m4 direct masked | 2.698 [2.687, 2.707], slower vs e11m4 direct masked |
| posit<32,2> | FP32 | field-balanced | 2.880 [2.876, 2.885], slower vs fp32_e8m23 native | 4.890 [4.882, 4.904], slower vs fp32_e8m23 native |
| posit<32,2> | FP32 | paired log-uniform | 3.257 [3.252, 3.261], slower vs fp32_e8m23 native | 6.057 [6.043, 6.071], slower vs fp32_e8m23 native |
| posit<32,2> | FP64 | field-balanced | 2.654 [2.651, 2.660], slower vs e11m20 direct masked | 4.735 [4.716, 4.745], slower vs e11m20 direct masked |
| posit<32,2> | FP64 | paired log-uniform | 2.654 [2.650, 2.657], slower vs e11m20 direct masked | 4.751 [4.736, 4.766], slower vs e11m20 direct masked |
| takum<32> | FP32 | field-balanced | 2.790 [2.786, 2.794], slower vs fp32_e8m23 native | 4.679 [4.671, 4.694], slower vs fp32_e8m23 native |
| takum<32> | FP32 | paired log-uniform | 3.956 [3.949, 3.961], slower vs fp32_e8m23 native | 6.965 [6.948, 6.981], slower vs fp32_e8m23 native |
| takum<32> | FP64 | field-balanced | 2.298 [2.295, 2.303], slower vs e11m20 direct masked | 3.896 [3.879, 3.905], slower vs e11m20 direct masked |
| takum<32> | FP64 | paired log-uniform | 2.301 [2.297, 2.304], slower vs e11m20 direct masked | 3.882 [3.869, 3.890], slower vs e11m20 direct masked |
| takum_log<32> | FP32 | field-balanced | 3.026 [3.022, 3.029], slower vs fp32_e8m23 native | 5.510 [5.500, 5.528], slower vs fp32_e8m23 native |
| takum_log<32> | FP32 | paired log-uniform | 3.030 [3.025, 3.033], slower vs fp32_e8m23 native | 5.517 [5.505, 5.534], slower vs fp32_e8m23 native |
| takum_log<32> | FP64 | field-balanced | 3.074 [3.069, 3.080], slower vs e11m20 direct masked | 5.678 [5.657, 5.690], slower vs e11m20 direct masked |
| takum_log<32> | FP64 | paired log-uniform | 3.074 [3.069, 3.078], slower vs e11m20 direct masked | 5.676 [5.658, 5.688], slower vs e11m20 direct masked |

## Answers

### 1. Is a full LUT best at low bit counts?

Yes for every measured 8-bit and 14-bit main case. Shared LUT won all 24 8-bit cases. Global LUT won all 24 14-bit cases. Across the 96 individual LUT-versus-direct comparisons, 87 LUT cases were faster. The nine slower cases were 14-bit FP64 shared-LUT cases, where each block stages a 128 KiB table. That staging cost does not change the winner because the global LUT remained faster than direct.

### 2. Does LUT speed depend on whether the entries contain IEEE, posit, or takum values?

No measurable dependence appeared. All 96 paired controls were equivalent under the predeclared 3% band. This held for both widths, arithmetic types, traces, placements, and kernels. The observed alternative-to-IEEE ratios ranged from 0.989 to 1.009. Once the raw index stream and kernel are fixed, the LUT contents do not affect conversion throughput.

### 3. What wins after a shared full LUT becomes too large?

At 16 bits, the global full LUT beat direct decoding in all 24 cases. Its paired ratio to direct ranged from 0.225 to 0.623. At 32 bits, direct decoding is the only complete strategy measured because a full table is impractical. The experiment therefore establishes a 32-bit direct baseline, not that direct is better than every possible segmented or approximate decoder.

### 4. How do the best alternatives compare with retained IEEE formats of the same width?

No alternative case was faster. Of 96 cases, 16 were equivalent, 2 were inconclusive, and 78 were slower. All equivalent and inconclusive results occurred at 8 bits. The ratio ranges were 0.998 to 1.144 at 8 bits, 1.078 to 1.728 at 14 bits, 1.037 to 2.698 at 16 bits, and 2.298 to 6.965 at 32 bits.

The main-format comparisons do not reuse identical values because each format has its own safe interval and field-balanced generator. Question 2 is the clean conversion-only comparison. It shows equal LUT throughput. Question 4 measures the specified end-to-end kernels with each format's own inputs, so cache locality and code distribution can also affect the result.

## Validation and limits

The full run contains 604 timed cases and 18,120 successful samples. All 119 decoder validation rows passed. The run recorded 23,668 histogram rows and no infeasible case. The build gate compiled all 44 targets, the smoke run covered the same 604-case matrix, and the full job completed in 421 seconds. A separate context-free CUDA review ended with no remaining finding.

The benchmark measures storage conversion followed by ordinary FP32 or FP64 arithmetic. It does not test posit quires, native posit or takum arithmetic, segmented 32-bit decoders, or an accuracy-matched application. These timings alone cannot show that any family has a better accuracy-performance trade-off.

## Reproducibility files

- [Raw timing samples](run_20260826T183425Z/full/timing_samples.csv)
- [Decoder validation](run_20260826T183425Z/full/decoder_validation.csv)
- [Input histograms](run_20260826T183425Z/full/histograms.csv)
- [All confidence intervals](run_20260826T183425Z/analysis/case_comparisons.csv)
- [All strategy winners](run_20260826T183425Z/analysis/strategy_winners.csv)
- [Timing medians](run_20260826T183425Z/analysis/timing_summary.csv)
- [Run manifest](run_20260826T183425Z/run_manifest.txt)
- [GPU and compiler environment](run_20260826T183425Z/environment.txt)
- [Independent CUDA review](code_review.md)
- [Predeclared experiment specification](../../docs/posit_takum_strategy_benchmark.md)
