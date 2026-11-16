"""
Stage 5c: 15O-water MBF precision with randoms, scatter and dead time (NEC-equivalent thinning), for injected
doses of 400 and 800 MBq, without and with 60 mm end shields.

    python mbf_nec.py --recon /scratch/$USER/exercise_pet_recon --task N     one (geometry, state, scan)
    python mbf_nec.py --recon /scratch/$USER/exercise_pet_recon --collect    figure and tables

Each task simulates one stage-5 scan at 400 MBq (trues only, same seeds as stage 5), then for every scenario
keeps each true with probability f_k = NEC_k / T_k in its frame k, where NEC_k is the frame NEC of the scenario
from the stage-7 count-rate model (randoms, scatter fraction, block dead time, end shields) and T_k the loss-free
trues of the simulated 400 MBq scan. The kept counts are reconstructed exactly as in stage 5 and each frame image
is divided by f_k, so activity stays unbiased while the noise matches a randoms-, scatter- and dead-time-corrected
scan. Approximations: randoms and scatter noise are given the spatial distribution of the trues, and the
time-of-flight reduction of randoms noise is not credited (conservative).
Needs countrate/tables_<config>.npz from stage 7.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import dynamic as D
import countrate as CR
from run_recon import log

SIM_MBQ = 400.0
SCENARIOS = [("400 MBq", 400.0, None), ("400 MBq, 60 mm shields", 400.0, 60.0),
             ("800 MBq", 800.0, None), ("800 MBq, 60 mm shields", 800.0, 60.0)]


def frame_factors(recon_dir, config, state, dose, shields, par=None):
    tab = dict(np.load(Path(recon_dir) / "countrate" / f"tables_{config}.npz"))
    tg, act = CR.class_activity(state)
    p = dict(par or CR.BASE)
    if shields:
        p["shields"] = shields
    r = CR.rates(tab, tg, act, dose, p, config)
    fi = np.searchsorted(D.EDGES, tg, side="right") - 1
    nk = len(D.FRAMES)
    nec = np.bincount(fi, weights=r["nec"], minlength=nk)[:nk]
    t_sim = np.bincount(fi, weights=r["trues_ideal"] * SIM_MBQ / dose, minlength=nk)[:nk]
    f = nec / np.maximum(t_sim, 1e-12)
    # frames with almost no activity (before the bolus arrives): NEC/T is dominated by the Lu-176 background and
    # is meaningless; keep at least 5% there so that the few counts are not amplified without bound
    rate = t_sim / np.asarray(D.FRAMES, float)
    low = rate < 0.01 * rate.max()
    f[low] = np.maximum(f[low], 0.05)
    if f.max() > 1.0:
        log(f"  warning: NEC exceeds simulated trues in some frames ({f.max():.2f}); clipped to 1")
    return np.clip(f, 1e-6, 1.0)


def run_task(recon_dir, out, config, state, real, ctr, iters, subsets, seed):
    grid, sens, ev_all, n_tot = D.simulate(recon_dir, config, state, real, SIM_MBQ, seed)
    fi = np.searchsorted(D.EDGES, ev_all["t"], side="right") - 1
    mn, md = D.TRUE_MBF[state]
    rows, meta = [], {"config": config, "state": state, "real": real, "sim_mbq": SIM_MBQ, "ctr_ps": ctr,
                      "iters": iters, "subsets": subsets, "n_trues_sim": int(len(fi)), "scenarios": {}}
    npz = {}
    for si, (name, dose, sh) in enumerate(SCENARIOS):
        f = frame_factors(recon_dir, config, state, dose, sh)
        rng = np.random.default_rng(seed + 7000 + 1000 * real + 17 * D.STATES.index(state) + 101 * si)
        keep = rng.random(len(fi)) < f[fi]
        ev = {k: v[keep] for k, v in ev_all.items()}
        rng_r = np.random.default_rng(seed + 9000 + 1000 * real + 17 * D.STATES.index(state) + 101 * si)
        blood, segs, fits, counts, _ = D.recon_fit(config, grid, sens, ev, ctr, iters, subsets, rng_r, scale=f)
        for g, ft in enumerate(fits):
            rows.append({"config": config, "state": state, "real": real, "scenario": name, "dose_mbq": dose,
                         "shields_mm": sh or 0, "segment": g, "defect": g == D.DEF_SEG,
                         "true_mbf": md if g == D.DEF_SEG else mn, "mbf": ft["mbf"], "ptf": ft["ptf"],
                         "fbv": ft["fbv"], "cost": ft["cost"]})
        meta["scenarios"][name] = {"frame_factor": f.tolist(), "counts_kept": counts,
                                   "nec_total_M": float(keep.sum()) / 1e6}
        npz[f"s{si}_blood"], npz[f"s{si}_segs"] = blood, segs
        nm = [r["mbf"] for r in rows if r["scenario"] == name and not r["defect"]]
        log(f"  {name}: kept {keep.mean():.3f} of trues; MBF normal {np.mean(nm):.2f}, defect "
            f"{[r['mbf'] for r in rows if r['scenario'] == name and r['defect']][0]:.2f}")
    tag = f"task_{config}_{state}_r{real}"
    (out / f"{tag}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    np.savez_compressed(out / f"{tag}.npz", **npz)


def collect(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, metas = [], []
    for f in sorted(out.glob("task_*.json")):
        d = json.loads(f.read_text())
        rows += d["rows"]
        metas.append(d["meta"])
    if not rows:
        raise SystemExit("no task results")
    with open(out / "mbf_nec_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    configs = [c for c in D.CONFIGS if any(r["config"] == c for r in rows)]
    names = [s[0] for s in SCENARIOS]
    get = {(r["config"], r["state"], r["real"], r["scenario"], r["segment"]): r["mbf"] for r in rows}
    reals = sorted({r["real"] for r in rows})
    L = ["Stage 5c: 15O-water MBF with randoms, scatter and dead time (NEC-equivalent thinning of 400 MBq trues)",
         "config        state   scenario                 n  normal MBF    defect MBF    seg SD%  MFR normal     MFR defect"]
    summ = {}
    for c in configs:
        for sc in names:
            mfr_n, mfr_d = [], []
            for r in reals:
                if all((c, s, r, sc, 0) in get for s in D.STATES):
                    mfr_n.append(np.mean([get[(c, "stress", r, sc, g)] / get[(c, "rest", r, sc, g)]
                                          for g in range(16) if g != D.DEF_SEG]))
                    mfr_d.append(get[(c, "stress", r, sc, D.DEF_SEG)] / get[(c, "rest", r, sc, D.DEF_SEG)])
            for s in D.STATES:
                rr = [r for r in reals if (c, s, r, sc, 0) in get]
                if not rr:
                    continue
                nm = [np.mean([get[(c, s, r, sc, g)] for g in range(16) if g != D.DEF_SEG]) for r in rr]
                dm = [get[(c, s, r, sc, D.DEF_SEG)] for r in rr]
                segsd = np.mean([np.std([get[(c, s, r, sc, g)] for r in rr], ddof=1) / D.TRUE_MBF[s][0] * 100
                                 for g in range(16) if g != D.DEF_SEG]) if len(rr) > 1 else np.nan
                summ[(c, s, sc)] = dict(n=len(rr), nm=np.mean(nm), nsd=np.std(nm), dm=np.mean(dm), dsd=np.std(dm),
                                        segsd=segsd, mfrn=(np.mean(mfr_n), np.std(mfr_n)) if mfr_n else (np.nan, np.nan),
                                        mfrd=(np.mean(mfr_d), np.std(mfr_d)) if mfr_d else (np.nan, np.nan))
                q = summ[(c, s, sc)]
                L.append(f"{c:12s}  {s:6s}  {sc:23s} {q['n']:2d}  {q['nm']:.2f} ± {q['nsd']:.2f}   {q['dm']:.2f} ± {q['dsd']:.2f}"
                         f"   {q['segsd']:5.1f}   {q['mfrn'][0]:.2f} ± {q['mfrn'][1]:.2f}   {q['mfrd'][0]:.2f} ± {q['mfrd'][1]:.2f}")
    L.append("Stage 5 (trues only, 400 MBq) for comparison: segmental SD 1.5 to 2.0%; MFR defect 1.51 (open), 1.54 (closed).")
    (out / "SUMMARY_mbf_nec.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    col = {"R280_slot": "#2a78d6", "R350_closed": "#4a3aa7"}
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.arange(len(names))
    w = 0.38
    for n, c in enumerate(configs):
        for k, s in enumerate(D.STATES):
            v = [summ.get((c, s, sc), {}).get("segsd", np.nan) for sc in names]
            axs[0].bar(x + (n - 0.5) * w + (k - 0.5) * w / 2, v, w / 2 * 0.9, color=col[c], alpha=0.5 if s == "rest" else 1.0)
        m = [summ.get((c, "stress", sc), {}).get("mfrd", (np.nan, np.nan)) for sc in names]
        axs[1].bar(x + (n - 0.5) * w, [a for a, _ in m], w * 0.9, yerr=[b for _, b in m], color=col[c], capsize=2)
    axs[0].set_ylabel("Segmental MBF SD over scans (% of truth)")
    axs[0].set_title("A.  MBF precision (light: rest, dark: stress)", loc="left", fontweight="bold", fontsize=9)
    axs[1].axhline(1.5, color="k", lw=1)
    axs[1].axhline(2.0, color="#e34948", lw=0.8, ls=(0, (4, 2)))
    axs[1].set_ylabel("Flow reserve, ischemic segment")
    axs[1].set_title("B.  Ischemic flow reserve (true 1.5)", loc="left", fontweight="bold", fontsize=9)
    for ax in axs:
        ax.set_xticks(x, [s.replace(", ", "\n") for s in names], fontsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Fig_mbf_nec.{ext}", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--ntasks", action="store_true")
    ap.add_argument("--n-real", type=int, default=5)
    ap.add_argument("--real-from", type=int, default=0, help="run only scans with index >= this (adds repeats)")
    ap.add_argument("--ctr", type=int, default=200)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--subsets", type=int, default=10)
    ap.add_argument("--seed", type=int, default=51)                  # same as stage 5: same simulated scans
    ap.add_argument("--sim-mbq", type=float, default=400.0, help="smoke tests only")
    ap.add_argument("--outdir", default="mbf_nec")
    args = ap.parse_args()
    tasks = [t for t in D.tasks(args.n_real) if t[2] >= args.real_from]
    if args.ntasks:
        print(len(tasks))
        return
    global SIM_MBQ
    SIM_MBQ = args.sim_mbq
    out = Path(args.recon) / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    if args.collect:
        collect(out)
    elif args.task is not None:
        c, s, r = tasks[args.task]
        run_task(Path(args.recon), out, c, s, r, args.ctr, args.iters, args.subsets, args.seed)
    else:
        raise SystemExit("give --task N or --collect")


if __name__ == "__main__":
    main()
