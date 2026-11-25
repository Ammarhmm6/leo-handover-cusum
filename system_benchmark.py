"""System-level benchmark on the constellation simulator (ele6701_project_graphs.py by G. Dandil).
Runs Event A3, NP conventional HO, NP CHO, CUSUM and A3 with 10-sample averaging over 10 fading seeds.

Usage:  python system_benchmark.py --sim-path /path/to/ele6701_project   (folder containing ele6701_project_graphs.py)
Output: results/system.json  (mean and 95% CI over seeds for every detector setting)
"""
import sys, io, os, contextlib, json, time, argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--sim-path", default="ele6701_project"); ap.add_argument("--seeds", type=int, default=10)
ARGS = ap.parse_args(); sys.path.insert(0, ARGS.sim_path); os.makedirs("results", exist_ok=True)
import matplotlib; matplotlib.use('Agg')
import ele6701_project_graphs as M
M.plot_results = lambda a: None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sr_theory import SRModel
_SR = SRModel()
_thr = M.calculate_np_threshold; _gen = M.shadowed_rician_generator; _conv = M.conventional_handover
SEEDS = list(range(1, ARGS.seeds + 1))
# ---- CUSUM detector: S_j <- max(0, S_j + ln Lambda(T_j)), hand over to argmax S_j when it exceeds h
S = {}; H = [1.25]
def cusum(sat_beam_pairs, frame):
    global S
    powers, mean_power = {}, {}
    for sid_s, beams in sat_beam_pairs.items():
        sid = int(sid_s[3:]); d = beams[f"b{sid}_0"]
        powers[sid] = float(d["|h|"][0]); mean_power[sid] = float(d["mean_power"][0])
    old = M.serving_satellite
    rec = dict(frame=frame, time_s=float(M.t[frame]), serving_before=old, serving_after=old, hypothesis="N/A", handover=False)
    if not powers or old is None or old not in powers:
        M.serving_satellite = max(powers, key=powers.get) if powers else None
        rec["serving_after"] = M.serving_satellite; S = {}; return rec
    nb = {j: p for j, p in powers.items() if j != old}
    if not nb: return rec
    S = {j: max(0.0, S.get(j, 0.0) + float(_SR.llr_fast(np.array([nb[j]/powers[old]]))[0])) for j in nb}
    best = max(nb, key=mean_power.get); h1 = mean_power[best]/mean_power[old] > M.eta
    tgt = max(S, key=S.get); ho = S[tgt] > H[0]
    rec.update(hypothesis="H1" if h1 else "H0", handover=ho); c = M.ho_counts
    if ho:
        if h1: c["P_11"] += 1; c["D" if tgt == best else "WT"] += 1
        else: c["P_10"] += 1
        M.serving_satellite = tgt; rec["serving_after"] = tgt; S = {}
    else: c["P_01" if h1 else "P_00"] += 1
    return rec
Kv = [1]
def genK(m, omega, b):
    return np.sqrt(np.mean([abs(_gen(m, omega, b))**2 for _ in range(Kv[0])])) + 0j
def one(alg, seed):
    global S
    S = {}
    with contextlib.redirect_stdout(io.StringIO()): M.run_simulation(alg, seed=seed)
    d = M.decisions_per_frame; c = M.ho_counts
    nH0 = c["P_00"]+c["P_10"]; nH1 = c["P_01"]+c["P_11"]
    pp = sum(1 for i in range(1, len(d)) if d[i]["handover"] and d[i-1]["handover"] and d[i-1]["serving_before"] == d[i]["serving_after"] and d[i-1]["serving_after"] == d[i]["serving_before"])
    return dict(HO=sum(x["handover"] for x in d), UNN=c["P_10"], PP=pp, LATE=100*nH1/(nH0+nH1), PFA=c["P_10"]/nH0, PD=c["D"]/max(nH1, 1))
def config(name):
    M.shadowed_rician_generator = _gen; Kv[0] = 1; M.conventional_handover = _conv
    M.calculate_np_threshold = _thr; M.np_alpha = 0.05; _thr.cache_clear()
    if name.startswith("A3"):
        off = float(name.split()[1]); M.a3_trehsold_db = off; alg = "a3"
        if "K=10" in name: Kv[0] = 10; M.shadowed_rician_generator = genK
    elif name.startswith("NP-conv"):
        M.np_alpha = float(name.split()[1]); _thr.cache_clear()
        M.calculate_np_threshold = lambda N: ("upper_tail", np.nan, _thr(N)[2], np.inf) if N >= 1 else ("none",)*4
        alg = "conventional_handover"
    elif name.startswith("NP-CHO"):
        M.np_alpha = float(name.split()[1]); _thr.cache_clear()
        M.cho_gama_lower = _thr(3)[2]; M.cho_gama_upper = np.inf; alg = "conditional_handover"
    elif name.startswith("CUSUM"):
        H[0] = float(name.split()[1]); M.conventional_handover = cusum; alg = "conventional_handover"
    return alg
OUTF = 'results/system.json'
names = ["A3 1", "A3 3", "A3 6", "NP-conv 0.2", "NP-CHO 0.5", "NP-CHO 0.1", "CUSUM 0.75", "CUSUM 1.0", "CUSUM 1.25", "CUSUM 1.5", "CUSUM 2.0", "A3 2 K=10"] + [f"A3 {o}" for o in (-2, -1, 0, 2, 4, 8, 10, 12)] + [f"NP-conv {a}" for a in (0.5, 0.3, 0.1, 0.05, 0.01)] + [f"NP-CHO {a}" for a in (0.3, 0.2, 0.05, 0.01)] + [f"CUSUM {h}" for h in (0.5, 2.5, 3.0)]
out = {}
try: out = json.load(open(OUTF))
except Exception: pass
t0 = time.time()
for nm in names:
    if nm in out: continue
    alg = config(nm); runs = [one(alg, s) for s in SEEDS]
    agg = {k: (float(np.mean([r[k] for r in runs])), float(1.96*np.std([r[k] for r in runs], ddof=1)/np.sqrt(len(runs)))) for k in runs[0]}
    out[nm] = agg
    print(nm, {k: f"{v[0]:.3f}±{v[1]:.3f}" for k, v in agg.items()}, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(OUTF, 'w'), indent=1)
