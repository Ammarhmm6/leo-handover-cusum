"""Figures from results/system.json and results/theory_curves.json (+ results/validation.json)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
S = json.load(open("results/system.json"))
STY = {"A3": ("#257de0", "o", "Event A3 (offset sweep)"), "NP-conv": ("#f47230", "s", "NP conventional (α sweep)"),
       "NP-CHO": ("#00b894", "^", "NP CHO (α sweep)"), "CUSUM": ("#c0392b", "D", "CUSUM (h sweep)")}

def family(prefix):
    rows = [(k, v) for k, v in S.items() if k.split()[0] == prefix and "K=10" not in k]
    return sorted(rows, key=lambda kv: float(kv[0].split()[1]))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for fam, (c, mk, lab) in STY.items():
    rows = family(fam)
    if not rows: continue
    ho = [v["HO"][0] for _, v in rows]; late = [v["LATE"][0] for _, v in rows]
    pp = [v["PP"][0] for _, v in rows]; pfa = [v["PFA"][0] for _, v in rows]; pd = [v["PD"][0] for _, v in rows]
    ax[0].errorbar(ho, late, yerr=[v["LATE"][1] for _, v in rows], color=c, marker=mk, ms=5, lw=1.8, capsize=2, label=lab)
    ax[1].plot(ho, pp, color=c, marker=mk, ms=5, lw=1.8, label=lab)
    if fam != "CUSUM":
        o = np.argsort(pfa); ax[2].plot(np.array(pfa)[o], np.array(pd)[o], color=c, marker=mk, ms=5, lw=1.8, label=lab)
for k in [k for k in S if "K=10" in k]:
    ax[0].plot(S[k]["HO"][0], S[k]["LATE"][0], "*", ms=12, color="#555", label=f"{k} (10× measurements)")
ax[0].set(xscale="log", xlabel="Handovers per 900 s", ylabel="Late-HO time (%)", title="Late-HO time vs. number of HOs (lower-left is better)")
ax[1].set(xscale="log", xlabel="Handovers per 900 s", ylabel="Ping-pongs per 900 s", title="Ping-pongs")
ax[2].set(xlabel="$P_{FA}$ per epoch", ylabel="$P_D$ per epoch", title="Per-epoch ROC (single sample)")
for a in ax: a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False); a.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig("figures/system_results.png", dpi=160)

if os.path.exists("results/theory_curves.json") and os.path.exists("results/validation.json"):
    C = json.load(open("results/theory_curves.json")); V = json.load(open("results/validation.json"))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for (nm, c) in zip(["rho=1", "rho=eta", "rho=2eta", "rho=4eta"], ["#257de0", "#f47230", "#00b894", "#ee7fac"]):
        h, L = zip(*C[nm]); ax.semilogy(h, L, color=c, lw=2, label=f"theory, {nm}")
        r = [x for x in V["cusum"] if x["case"] == nm and x["mc"] == x["mc"]]
        ax.errorbar([x["h"] for x in r], [x["mc"] for x in r], yerr=[x["ci95"] for x in r], fmt="o", mfc="white", color=c)
    ax.set(xlabel="CUSUM threshold h", ylabel="Average run length (epochs of 0.3 s)", title="CUSUM: theory (lines) vs. Monte Carlo (markers)")
    ax.grid(alpha=.25); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig("figures/cusum_theory_vs_mc.png", dpi=160)
print("figures written")
