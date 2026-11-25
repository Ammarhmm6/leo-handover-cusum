"""Animated LEO handover visualizer.

Top:    side view of the sky above a fixed UE in Montreal. Every visible satellite is drawn with a line to the UE;
        the CUSUM serving link is green, the A3 serving link is blue (dashed), and the CUSUM leading candidate
        (largest CUSUM statistic) is orange.
Middle: live calculations for the current decision epoch.
Bottom: (1) measured and true power difference of the best neighbour vs. the serving satellite,
        (2) CUSUM statistic vs. its threshold h,  (3) cumulative handovers and ping-pongs, A3 vs. CUSUM.

Both detectors see exactly the same RSRP samples (same fading), so the comparison is fair.

Usage:
    python handover_visualizer.py                 # live window
    python handover_visualizer.py --save ho.mp4   # write a video (needs ffmpeg)
    python handover_visualizer.py --duration 300 --h 1.25 --a3-offset 3
"""
import argparse
import numpy as np
import matplotlib
from sr_theory import SRModel

# ---------------- constellation and link model (same parameters as the paper) ----------------
MU, R_E, H_SAT, OMEGA_E = 3.986004418e14, 6371e3, 550e3, 7.2921159e-5
N_PLANES, N_PER_PLANE, INC = 30, 60, np.deg2rad(53)
LAT0, LON0, MIN_EL = np.deg2rad(45.5), np.deg2rad(-73.6), np.deg2rad(35)
FC, C = 30e9, 299792458.0
EPOCH = 0.3


def tx_gain_db(psi_deg):
    """ITU-R S.1528-type pattern used in the simulator (G_max 40 dBi, psi_b = 1 deg, L_N = -25 dB)."""
    G, LN, LF, pb, a, b, al = 40.0, -25.0, 0.0, 1.0, 2.58, 6.32, 1.5
    X = G + LN + 25 * np.log10(b * pb); Y = b * pb * 10 ** (0.04 * (G + LN - LF))
    p = np.asarray(psi_deg, float); g = np.full(p.shape, LF)
    g = np.where(p <= a * pb, G - 3 * (p / pb) ** al, g)
    g = np.where((p > a * pb) & (p <= 0.5 * b * pb), G + LN, g)
    g = np.where((p > 0.5 * b * pb) & (p <= b * pb), G + LN, g)
    g = np.where((p > b * pb) & (p <= Y), X - 25 * np.log10(np.maximum(p, 1e-9)), g)
    return g


def constellation(t):
    """ECEF positions (len(t), N_sat, 3)."""
    Rs = R_E + H_SAT; w = np.sqrt(MU / Rs ** 3)
    raan = np.linspace(0, 2 * np.pi, N_PLANES, endpoint=False)
    th0 = np.linspace(0, 2 * np.pi, N_PER_PLANE, endpoint=False)
    u = (th0[None, None, :] + w * t[:, None, None]) % (2 * np.pi)
    O = raan[None, :, None]
    x = Rs * (np.cos(O) * np.cos(u) - np.sin(O) * np.sin(u) * np.cos(INC))
    y = Rs * (np.sin(O) * np.cos(u) + np.cos(O) * np.sin(u) * np.cos(INC))
    z = Rs * np.sin(u) * np.sin(INC) * np.ones_like(O)
    th = (OMEGA_E * t)[:, None, None]
    xe, ye = np.cos(th) * x + np.sin(th) * y, -np.sin(th) * x + np.cos(th) * y
    return np.stack([xe, ye, z], -1).reshape(len(t), -1, 3)


def geometry(t):
    P = constellation(t)
    up = np.array([np.cos(LAT0) * np.cos(LON0), np.cos(LAT0) * np.sin(LON0), np.sin(LAT0)])
    east = np.array([-np.sin(LON0), np.cos(LON0), 0.0]); north = np.cross(up, east)
    ue = R_E * up
    d = P - ue; dist = np.linalg.norm(d, axis=-1)
    el = np.arcsin(np.einsum("tsk,k->ts", d, up) / dist)
    nadir = -P / np.linalg.norm(P, axis=-1, keepdims=True)
    psi = np.rad2deg(np.arccos(np.clip(np.einsum("tsk,tsk->ts", nadir, -d / dist[..., None]), -1, 1)))
    mean_db = tx_gain_db(psi) - 20 * np.log10(4 * np.pi * dist * FC / C)
    E, U = np.einsum("tsk,k->ts", d, east), np.einsum("tsk,k->ts", d, up)
    return dict(el=el, mean_db=mean_db, E=E / 1e3, U=U / 1e3, vis=el >= MIN_EL)


# ---------------- detectors ----------------
def run_detectors(geo, model, h, a3_offset_db, hys_db, rng):
    T, S = geo["vis"].shape
    g = model.sample_g(T * S, rng).reshape(T, S)             # one SR fading sample per satellite per epoch
    meas_db = geo["mean_db"] + 10 * np.log10(g)
    log = {k: [] for k in ("serv_c", "serv_a", "lead", "best_true", "meas_diff", "true_diff", "llr", "smax",
                           "ho_c", "ho_a", "pp_c", "pp_a")}
    sc = sa = None; cus = {}; hist_c, hist_a = [], []; n_ho_c = n_ho_a = n_pp_c = n_pp_a = 0
    for k in range(T):
        vis = np.flatnonzero(geo["vis"][k])
        if vis.size == 0:
            sc = sa = None; cus = {}
        else:
            if sc not in vis: sc = int(vis[np.argmax(meas_db[k, vis])]); cus = {}
            if sa not in vis: sa = int(vis[np.argmax(meas_db[k, vis])])
            nb = [j for j in vis if j != sc]
            # CUSUM (sums of the closed-form LLR, one per neighbour)
            lead, llr_lead = None, np.nan
            if nb:
                T_ratio = 10 ** ((meas_db[k, nb] - meas_db[k, sc]) / 10)
                z = model.llr_fast(T_ratio)
                cus = {j: max(0.0, cus.get(j, 0.0) + zi) for j, zi in zip(nb, z)}
                lead = max(cus, key=cus.get); llr_lead = float(z[nb.index(lead)])
                if cus[lead] > h:
                    hist_c.append((k, sc, lead)); n_ho_c += 1
                    if len(hist_c) > 1 and hist_c[-2][0] >= k - 1 and hist_c[-2][1] == lead: n_pp_c += 1
                    sc = lead; cus = {}
            # A3 (fixed offset on the same samples)
            nba = [j for j in vis if j != sa]
            if nba:
                tgt = nba[int(np.argmax(meas_db[k, nba]))]
                if meas_db[k, tgt] - meas_db[k, sa] > hys_db + a3_offset_db:
                    hist_a.append((k, sa, tgt)); n_ho_a += 1
                    if len(hist_a) > 1 and hist_a[-2][0] >= k - 1 and hist_a[-2][1] == tgt: n_pp_a += 1
                    sa = tgt
        nbv = [j for j in vis if j != sc] if sc is not None else []
        bt = nbv[int(np.argmax(geo["mean_db"][k, nbv]))] if nbv else None
        bm = nbv[int(np.argmax(meas_db[k, nbv]))] if nbv else None
        log["serv_c"].append(sc); log["serv_a"].append(sa); log["lead"].append(lead if vis.size else None)
        log["best_true"].append(bt)
        log["meas_diff"].append(meas_db[k, bm] - meas_db[k, sc] if bm is not None else np.nan)
        log["true_diff"].append(geo["mean_db"][k, bt] - geo["mean_db"][k, sc] if bt is not None else np.nan)
        log["llr"].append(llr_lead if vis.size else np.nan)
        log["smax"].append(max(cus.values()) if cus else 0.0)
        log["ho_c"].append(n_ho_c); log["ho_a"].append(n_ho_a); log["pp_c"].append(n_pp_c); log["pp_a"].append(n_pp_a)
    return log, meas_db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=600, help="simulated seconds")
    ap.add_argument("--h", type=float, default=1.25, help="CUSUM threshold")
    ap.add_argument("--a3-offset", type=float, default=3.0, help="A3 offset in dB (added to the 2 dB hysteresis)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save", default=None, help="output video file (.mp4 or .gif)")
    ap.add_argument("--fps", type=int, default=7, help="frames per second (one frame = one 0.3-s epoch; 7 fps = about 2x real time)")
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--start", type=float, default=0.0, help="start time of the clip (s)")
    a = ap.parse_args()
    if a.save: matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    model = SRModel(); hys = 2.0
    t = np.arange(0, a.duration, EPOCH)
    geo = geometry(t)
    log, meas = run_detectors(geo, model, a.h, a.a3_offset, hys, np.random.default_rng(a.seed))
    arl0 = model.arl(a.h, 1.0); arlb = model.arl(a.h, model.eta); dly = model.arl(a.h, model.rho_ho)

    fig = plt.figure(figsize=(14, 9.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.3, 0.62, 1.5], hspace=0.30, wspace=0.28, left=0.06, right=0.98, top=0.95, bottom=0.07)
    ax = fig.add_subplot(gs[0, :]); axt = fig.add_subplot(gs[1, :]); axt.axis("off")
    a1, a2, a3 = (fig.add_subplot(gs[2, i]) for i in range(3))
    # scene
    xs = np.linspace(-1300, 1300, 400); earth = -(xs ** 2) / (2 * R_E / 1e3)
    ax.fill_between(xs, earth, -400, color="#cfe8cf"); ax.plot(xs, earth, color="#2e7d32", lw=1.5)
    hmax = H_SAT / 1e3
    ax.plot(xs, earth + hmax, ls=":", color="#9c9c9c", lw=1); ax.text(1280, hmax + 15, "550 km shell", color="#777", ha="right", fontsize=8)
    ax.plot([0], [0], marker="^", ms=13, color="#222"); ax.text(0, -95, "UE (Montréal)", ha="center", fontsize=9)
    ax.set_xlim(-1300, 1300); ax.set_ylim(-130, 720); ax.set_xlabel("East (km)"); ax.set_ylabel("Up (km)")
    ax.set_title("LEO handover: CUSUM (green) vs. Event A3 (blue, dashed) on the same RSRP samples", fontsize=11)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    sats = ax.scatter([], [], s=22, color="#555", zorder=3)
    vis_lines = [ax.plot([], [], color="#bbb", lw=0.8, zorder=1)[0] for _ in range(40)]
    l_c, = ax.plot([], [], color="#2e9e44", lw=3, zorder=4, label="CUSUM serving link")
    l_a, = ax.plot([], [], color="#257de0", lw=2, ls="--", zorder=4, label="A3 serving link")
    l_l, = ax.plot([], [], color="#f47230", lw=1.6, ls=":", zorder=4, label="CUSUM leading candidate")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    txt = axt.text(0.0, 1.05, "", va="top", family="monospace", fontsize=9, transform=axt.transAxes, linespacing=1.5)
    # graphs
    W = int(120 / EPOCH)
    md, = a1.plot([], [], color="#bbb", lw=0.8, label="measured $R_B-R_A$")
    td, = a1.plot([], [], color="#222", lw=1.5, label="true (mean) difference")
    a1.axhline(hys, color="#2e9e44", lw=1, ls=":", label="hysteresis 2 dB")
    a1.axhline(hys + a.a3_offset, color="#257de0", lw=1, ls="--", label="A3 threshold")
    a1.set(ylim=(-20, 20), xlabel="time (s)", ylabel="dB", title="Best neighbour vs. serving")
    sm, = a2.plot([], [], color="#2e9e44", lw=1.5); a2.axhline(a.h, color="#d03b3b", ls="--", lw=1.2, label=f"threshold h = {a.h}")
    a2.set(ylim=(0, a.h * 1.6), xlabel="time (s)", ylabel="max$_j$ $S_j$", title="CUSUM statistic")
    hc, = a3.plot([], [], color="#2e9e44", lw=2, label="CUSUM HOs"); ha_, = a3.plot([], [], color="#257de0", lw=2, label="A3 HOs")
    pc, = a3.plot([], [], color="#2e9e44", lw=1, ls=":", label="CUSUM ping-pongs"); pa, = a3.plot([], [], color="#257de0", lw=1, ls=":", label="A3 ping-pongs")
    a3.set(xlabel="time (s)", ylabel="count", title="Cumulative handovers / ping-pongs")
    for g_ in (a1, a2, a3):
        g_.grid(alpha=.25); [g_.spines[s].set_visible(False) for s in ("top", "right")]; g_.legend(fontsize=7, frameon=False, loc="upper left")

    k0 = int(a.start / EPOCH)

    def draw(k):
        vis = np.flatnonzero(geo["vis"][k]); E, U = geo["E"][k], geo["U"][k]
        sats.set_offsets(np.c_[E[vis], U[vis]] if vis.size else np.empty((0, 2)))
        for i, ln in enumerate(vis_lines):
            if i < vis.size: ln.set_data([0, E[vis[i]]], [0, U[vis[i]]])
            else: ln.set_data([], [])
        for ln, key in ((l_c, "serv_c"), (l_a, "serv_a"), (l_l, "lead")):
            s = log[key][k]; ln.set_data(([0, E[s]], [0, U[s]]) if s is not None else ([], []))
        sc, sa, ld = log["serv_c"][k], log["serv_a"][k], log["lead"][k]
        el = np.rad2deg(geo["el"][k, sc]) if sc is not None else np.nan
        txt.set_text(
            f"t = {t[k]:6.1f} s   visible satellites: {vis.size:2d}   CUSUM serving: #{sc} (elevation {el:4.1f} deg)   A3 serving: #{sa}\n"
            f"best neighbour: measured R_B - R_A = {log['meas_diff'][k]:+6.2f} dB   true = {log['true_diff'][k]:+6.2f} dB   "
            f"leader #{ld}: ln Lambda = {log['llr'][k]:+5.2f}   CUSUM max S = {log['smax'][k]:4.2f}  (h = {a.h})\n"
            f"handovers   CUSUM: {log['ho_c'][k]:3d} (ping-pongs {log['pp_c'][k]:2d})     A3 ({a.a3_offset:g} dB offset): {log['ho_a'][k]:3d} (ping-pongs {log['pp_a'][k]:2d})\n"
            f"theory for h = {a.h}:  mean time to false alarm {arl0 * EPOCH:5.1f} s (equal powers), {arlb * EPOCH:4.1f} s (at the hysteresis boundary);"
            f"  mean delay {dly * EPOCH:3.1f} s (neighbour 3 dB above the boundary)")
        lo = max(0, k - W); tt = t[lo:k + 1]
        md.set_data(tt, log["meas_diff"][lo:k + 1]); td.set_data(tt, log["true_diff"][lo:k + 1]); sm.set_data(tt, log["smax"][lo:k + 1])
        for g_ in (a1, a2): g_.set_xlim(t[lo], t[lo] + W * EPOCH)
        tt0 = t[k0:k + 1]
        hc.set_data(tt0, log["ho_c"][k0:k + 1]); ha_.set_data(tt0, log["ho_a"][k0:k + 1])
        pc.set_data(tt0, log["pp_c"][k0:k + 1]); pa.set_data(tt0, log["pp_a"][k0:k + 1])
        a3.set_xlim(t[k0], max(t[k], t[k0] + 10)); a3.set_ylim(0, max(5, log["ho_a"][k] * 1.15))
        return []

    anim = FuncAnimation(fig, draw, frames=range(k0, len(t)), interval=1000 / a.fps, blit=False)
    if a.save:
        anim.save(a.save, fps=a.fps, dpi=a.dpi, extra_args=["-movflags", "+faststart", "-pix_fmt", "yuv420p", "-crf", "20"])
        print("saved", a.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
