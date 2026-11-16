Table 1. Systems and software. Values read from the env block of the benchmark JSON files (neo_bench/results).

| System | Host CPU | CPU cores (node / job) | GPU and memory | Python / numpy / numba / numba-cuda | Role |
|---|---|---|---|---|---|
| GH200 (NEO) | NVIDIA Grace, aarch64 | 72 / 72 | GH200, 144 GB HBM3e | 3.11.18 / 2.4.6 / 0.68.1 / 0.30.4 | GPU; also Grace CPU baseline (72 threads) |
| Rivanna A100 | x86_64 | 128 / 16 | A100-SXM4 80 GB | 3.11.18 / 2.4.6 / 0.68.1 / 0.30.4 | GPU |
| Rivanna B200 | x86_64 | 120 / 16 | B200, 183 GiB | 3.11.18 / 2.4.6 / 0.68.1 / 0.30.4 | GPU |
| Afton CPU | AMD EPYC 9454, x86_64 | 96 / 32 threads | none | 3.11.18 / 2.4.6 / 0.68.1 / none | Like-for-like CPU baseline |
