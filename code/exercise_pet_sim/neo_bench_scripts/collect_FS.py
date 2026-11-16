"""Summarize run FS: python collect_FS.py <folder with dynamic_gpu_c200/ and dynamic_gpu_c320/> """
import sys, json, glob, os, numpy as np
from collections import defaultdict
root = sys.argv[1]
d = defaultdict(dict)
for f in glob.glob(os.path.join(root, "dynamic_gpu_c*", "gpu_*_r*.json")):
    r = json.load(open(f)); ctr = int(os.path.basename(os.path.dirname(f))[len("dynamic_gpu_c"):])
    d[(r["config"], ctr)][(r["state"], r["real"])] = r
def ms(x): x = np.array(x); return f"{x.mean():.3f} +/- {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"
lines = ["| Ring | CTR ps | scans | rest normal | stress normal | rest defect | stress defect | MFR normal | MFR defect | SD stress normal across scans % | trues rest M | trues stress M | total s rest | total s stress |", "|" + "---|" * 14]
for (cfg, ctr), v in sorted(d.items()):
    reals = sorted({k[1] for k in v if ("rest", k[1]) in v and ("stress", k[1]) in v})
    if not reals: continue
    g = lambda s, key: [v[(s, i)][key] for i in reals]
    sec = lambda s: [v[(s, i)]["seconds"]["total"] for i in reals]
    mfr_n = [v[("stress", i)]["mbf_normal_mean"] / v[("rest", i)]["mbf_normal_mean"] for i in reals]
    mfr_d = [v[("stress", i)]["mbf_defect"] / v[("rest", i)]["mbf_defect"] for i in reals]
    sdp = 100 * np.std(g("stress", "mbf_normal_mean"), ddof=1) / np.mean(g("stress", "mbf_normal_mean")) if len(reals) > 1 else float("nan")
    lines.append(f"| {cfg} | {ctr} | {len(reals)} | {ms(g('rest','mbf_normal_mean'))} | {ms(g('stress','mbf_normal_mean'))} | {ms(g('rest','mbf_defect'))} | {ms(g('stress','mbf_defect'))} | {ms(mfr_n)} | {ms(mfr_d)} | {sdp:.2f} | {np.mean(g('rest','n_trues'))/1e6:.1f} | {np.mean(g('stress','n_trues'))/1e6:.1f} | {np.mean(sec('rest')):.0f} | {np.mean(sec('stress')):.0f} |")
txt = "\n".join(lines); print(txt); open(os.path.join(root, "SUMMARY_FS.md"), "w").write(txt + "\n")
