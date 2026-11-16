Table 2. Throughput for the 120-scan run (56 cm and 70 cm rings, rest and stress; 20 scans at 200 ps and 10 at 320 ps). Per-scan stage times are computed from the JSON files (matches). Wall time, queue and submission-to-results are from the project handoff notes, not recomputed here. Rivanna jobs used 16 CPU cores; GH200 used 72. Queue and submission times are single samples.

| System | Wall time for 120 scans | Scans per node-hour | Queue wait | Submission to results | Per scan, open ring 200 ps: simulate / OSEM / fit+other / total (s) |
|---|---|---|---|---|---|
| GH200 | 5 h 49 min | 20.6 | 0 | 5 h 49 min | 76 / 26 / 20 / 124 |
| B200 | 5 h 23 min | 22.3 | 4 h 55 min | 10 h 18 min | 67 / 17 / 28 / 112 |
| A100 | 9 h 38 min | 12.4 | 1 min 31 s | 9 h 40 min | 126 / 34 / 41 / 202 |
