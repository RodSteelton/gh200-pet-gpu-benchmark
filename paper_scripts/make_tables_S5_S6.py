"""Tables S5 (run C, 10 geometries) and S6 (run E with the original inputs) for the GH200 methods paper.
Inputs: neo_bench/results/runC8c_files_648559.tgz, runEorig_files_648559.tgz, runCDE_641162.out, runEC_648559.out,
results/exercise_pet_recon (old Rivanna results), rivanna_scratch_backup/logs (old Rivanna logs)."""
import json, re, glob, os, tarfile, tempfile, csv
import numpy as np
A = os.path.expanduser('~/mnt/Claude-analysis'); R = A + '/neo_bench/results'; O = A + '/results/exercise_pet_recon'
L = A + '/rivanna_scratch_backup/logs'; OUT = A + '/paper_figs'
tmp = tempfile.mkdtemp()
for f in ('runC8c_files_648559.tgz', 'runEorig_files_648559.tgz'):
    tarfile.open(R + '/' + f).extractall(tmp)
N = glob.glob(tmp + '/**/ivy-hip-monte', recursive=True)[0]
LABEL = {"R280_slot": "56 cm ring, 290°", "R350_closed": "70 cm closed ring", "P2_AP40": "2 panels AP, 40 cm", "P4_eq": "4 panels, 34 cm",
         "P2_T20A30": "2 panels, 20 cm wide x 30 cm axial", "P2_T30A20": "2 panels, 30 cm wide x 20 cm axial",
         "P4_T20A30": "4 panels, 20 cm wide x 30 cm axial", "P4_T30A20": "4 panels, 30 cm wide x 20 cm axial",
         "P4_T30A24": "4 panels, 30 cm wide x 24 cm axial", "P4_T30A30": "4 panels, 30 cm wide x 30 cm axial"}
pat = re.compile(r'\[(\S+)\] (\S+)\s+[\d.e+]+ decays:.*?\((\d+) s\)')
def last_timer(txt):
    d = {}
    for c, _, s in pat.findall(txt): d[c] = int(s)
    return d
grace = {}
for f in ('runCDE_641162.out', 'runEC_648559.out'): grace.update(last_timer(open(R + '/' + f).read()))
old = {}
for f in sorted(glob.glob(L + '/panels_cr_2*_*.log')):
    for c, s in last_timer(open(f, errors='ignore').read()).items(): old.setdefault(c, []).append(s)
new = {(r['config'], r['state']): r for r in json.load(open(N + '/recon_C_8c/panels_cr/countrate_8b.json'))}
ref = {(r['config'], r['state']): r for r in json.load(open(O + '/panels_cr/countrate_8b.json'))}
mx = max(abs(v - ref[k][f]) / max(abs(ref[k][f]), 1e-30) for k in new for f, v in new[k].items() if isinstance(v, (int, float)))
rng = lambda v: str(v[0]) if min(v) == max(v) else f'{min(v)} to {max(v)}'
md = ['Table S5. Count-rate results for run C on the Grace CPU (15O-water, 400 MBq, 240 s, non-TOF NEC, scatter fraction 0.35, block dead time 200 ns). One task per geometry covers rest and stress. Task time is the timer in the code (tables for all activity classes, 30 worker processes); the Rivanna times are from the original logs (32 core tasks).', '',
      '| Geometry | State | Window (ns) | Trues (M) | Randoms (M) | NEC (M) | NEC / ideal trues | Minimum live fraction | Task time, Grace (s) | Task time, Rivanna (s) |', '|---|---|---|---|---|---|---|---|---|---|']
rows = [['geometry', 'label', 'state', 'window_ns', 'trues_M', 'randoms_M', 'nec_M', 'nec_over_ideal_trues', 'live_min', 'grace_task_s', 'rivanna_task_s']]
for c in LABEL:
    for s in ('rest', 'stress'):
        r = new[(c, s)]; first = s == 'rest'
        md.append(f"| {LABEL[c]} | {s} | {r['window_ns']:.2f} | {r['trues_M']:.1f} | {r['randoms_M']:.1f} | {r['nec_M']:.1f} | {r['nec_over_trues_ideal']:.3f} | {r['live_min']:.3f} | {grace[c] if first else ''} | {rng(old[c]) if first else ''} |")
        rows.append([c, LABEL[c], s, f"{r['window_ns']:.2f}", f"{r['trues_M']:.4f}", f"{r['randoms_M']:.4f}", f"{r['nec_M']:.4f}", f"{r['nec_over_trues_ideal']:.5f}", f"{r['live_min']:.5f}", grace[c], rng(old[c])])
open(OUT + '/TableS5_countrate.md', 'w', encoding='utf-8').write('\n'.join(md) + '\n')
csv.writer(open(OUT + '/TableS5_countrate.csv', 'w', newline='', encoding='utf-8')).writerows(rows)
ratio = [(grace[c] / max(old[c]), grace[c] / min(old[c])) for c in LABEL]
print('S5: max relative difference to old, 20 rows: %.2e; Grace %d to %d s, Rivanna %d to %d s; Grace/Rivanna %.2f to %.2f' % (
    mx, min(grace.values()), max(grace.values()), min(min(v) for v in old.values()), max(max(v) for v in old.values()), min(a for a, _ in ratio), max(b for _, b in ratio)))
# ---- S6
VAR = [('none', 'No correction'), ('gated', 'Respiratory gate'), ('mc_data', 'Data-driven correction'), ('cam_good', 'Camera, good'),
       ('cam_poor', 'Camera, poor'), ('hybrid', 'Camera plus data'), ('mc_true', 'Ideal correction')]
def load(base, scen):
    j = json.load(open(f'{base}/task_R280_slot_{scen}_r0.json')); rr = j if isinstance(j, list) else j['rows']
    return {r['variant']: r for r in rr}
md6 = ['Table S6. Run E on the Grace CPU (original events file and sensitivity image) against the old results (56 cm ring, scan 0, 150 ps, true defect contrast 0.20).', '',
       '| Variant | Raw defect contrast, Grace / old | Calibrated contrast, Grace / old | Recovery, Grace / old | Remote CV (%), Grace / old |', '|---|---|---|---|---|']
dmax = {}
def row(name, a, b):
    for f in ('defect_contrast', 'defect_contrast_pc', 'recovery', 'remote_cv_pct', 'remote_cv_pct_pc', 'false_defect', 'false_defect_pc'):
        dmax[f] = max(dmax.get(f, 0), abs(a[f] - b[f]))
    md6.append(f"| {name} | {a['defect_contrast']:.3f} / {b['defect_contrast']:.3f} | {a['defect_contrast_pc']:.3f} / {b['defect_contrast_pc']:.3f} | {a['recovery']:.3f} / {b['recovery']:.3f} | {a['remote_cv_pct']:.1f} / {b['remote_cv_pct']:.1f} |")
gs, os_ = load(N + '/recon_E_orig/motion9', 'static'), load(O + '/motion9', 'static')
row('Static, no motion', gs['none'], os_['none'])
ge, oe = load(N + '/recon_E_orig/motion9', 'exercise'), load(O + '/motion9', 'exercise')
for v, name in VAR: row(name, ge[v], oe[v])
open(OUT + '/TableS6_runE.md', 'w', encoding='utf-8').write('\n'.join(md6) + '\n')
imx = 0
for scen in ('static', 'exercise'):
    a = np.load(f'{N}/recon_E_orig/motion9/task_R280_slot_{scen}_r0.npz'); b = np.load(f'{O}/motion9/task_R280_slot_{scen}_r0.npz')
    for k in a.files: imx = max(imx, float(np.max(np.abs(a[k] - b[k]))))
ca, cb = np.load(N + '/recon_E_orig/motion9/cal_R280_slot.npz'), np.load(O + '/motion9/cal_R280_slot.npz')
cmx = max(float(np.max(np.abs(ca[k] - cb[k]) / np.abs(cb[k]))) for k in ca.files)
print('S6: n_trues static %d / %d, exercise %d / %d; image slices max abs difference %g; calibration max relative difference %.2e' % (
    gs['none']['n_trues'], os_['none']['n_trues'], ge['none']['n_trues'], oe['none']['n_trues'], imx, cmx))
print('S6 max abs differences:', {k: float('%.3g' % v) for k, v in dmax.items()})
