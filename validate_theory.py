"""Theory vs Monte Carlo at fixed geometry, for (a) the single-shot NP detector and (b) the CUSUM detector.
Outputs: results/validation.json and figures/validation.png"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sr_theory import SRModel

os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
M = SRModel(); rng = np.random.default_rng(2026); out = {"np": [], "cusum": []}

# (a) NP single-shot: P(trigger) vs true margin above the hysteresis boundary, alpha = 0.05
n = 400_000
for N in (1, 3):
    gam = M.np_threshold(0.05, N)
    gA = M.sample_g(n, rng); gj = np.stack([M.sample_g(n, rng) for _ in range(N)], 1)
    for dB in np.arange(0, 10.5, 1.0):
        rho = M.eta * 10 ** (dB / 10)
        T = np.column_stack([rho * gj[:, 0]] + [M.eta * gj[:, i] for i in range(1, N)]) / gA[:, None]
        out["np"].append(dict(N=N, margin_db=float(dB), theory=M.p_trigger(gam, rho, N), mc=float(np.mean(T.max(1) > gam))))

# (b) CUSUM: average run length (epochs) vs threshold h
def mc_arl(h, rho, R=8000, cap=200000):
    S = np.zeros(R); rl = np.full(R, -1); act = np.arange(R); t = 0
    while act.size and t < cap:
        t += 1
        S[act] = np.maximum(0, S[act] + M.llr_fast(rho * M.sample_g(act.size, rng) / M.sample_g(act.size, rng)))
        hit = S[act] > h; rl[act[hit]] = t; act = act[~hit]
    r = rl[rl > 0]; return float(r.mean()), float(1.96 * r.std() / np.sqrt(len(r)))
cases = {"rho=1": 1.0, "rho=eta": M.eta, "rho=2eta": 2 * M.eta, "rho=4eta": 4 * M.eta}
for h in (0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
    for name, rho in cases.items():
        th = M.arl(h, rho)
        if th > 1200: mc, ci = float("nan"), float("nan")
        else: mc, ci = mc_arl(h, rho)
        out["cusum"].append(dict(h=h, case=name, theory=th, mc=mc, ci95=ci))
        print(f"h={h:<4} {name:9s} theory={th:9.2f} MC={mc:9.2f} ±{ci:.2f}", flush=True)
json.dump(out, open("results/validation.json", "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for N, c in ((1, "#257de0"), (3, "#f47230")):
    r = [x for x in out["np"] if x["N"] == N]
    ax[0].plot([x["margin_db"] for x in r], [x["theory"] for x in r], color=c, lw=2, label=f"theory, N={N}")
    ax[0].plot([x["margin_db"] for x in r], [x["mc"] for x in r], "o", mfc="white", color=c, label=f"Monte Carlo, N={N}")
ax[0].set(xlabel="True margin above hysteresis boundary (dB)", ylabel="P(handover triggered)", title="Single-shot NP detector, α = 0.05")
cols = dict(zip(cases, ["#257de0", "#f47230", "#00b894", "#ee7fac"]))
for name in cases:
    r = [x for x in out["cusum"] if x["case"] == name]
    ax[1].semilogy([x["h"] for x in r], [x["theory"] for x in r], color=cols[name], lw=2, label=f"theory, {name}")
    ax[1].errorbar([x["h"] for x in r], [x["mc"] for x in r], yerr=[x["ci95"] for x in r], fmt="o", mfc="white", color=cols[name])
ax[1].set(xlabel="CUSUM threshold h", ylabel="Average run length (epochs)", title="CUSUM: false-alarm time (ρ≤η) and delay (ρ>η)")
for a in ax: a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False); a.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig("figures/validation.png", dpi=160)
