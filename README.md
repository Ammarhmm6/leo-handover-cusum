# LEO handover as hypothesis testing: NP and CUSUM detectors

Code for an unpublished paper on LEO handover as hypothesis testing (NP and CUSUM detectors).
**Private repository. Do not make public before the paper is accepted.** The paper source is intentionally not included.

## Files

| File | What it does |
|---|---|
| `sr_theory.py` | Analytical core. Shadowed-Rician (SR) power gain as a gamma mixture (integer *m*), exact CDF/PDF of the RSRP ratio (beta-prime mixture), log-likelihood ratio, NP threshold for *N* neighbours (Gauss–Laguerre), and exact CUSUM run lengths (Brook–Evans Markov chain): mean time to false alarm and mean detection delay. |
| `validate_theory.py` | Theory vs. Monte Carlo at fixed geometry, for the single-shot NP detector and the CUSUM detector. Writes `results/validation.json` and `figures/validation.png`. |
| `system_benchmark.py` | Constellation-level benchmark on G. Dandil's simulator (`ele6701_project_graphs.py`): Event A3 (offset sweep), NP conventional HO, NP CHO, CUSUM (threshold sweep), and A3 with 10-sample averaging; 10 fading seeds, mean and 95% CI. Writes `results/system.json`. |
| `make_figures.py` | Draws the result figures from the JSON files into `figures/`. |
| `handover_visualizer.py` | Animated handover scene: Earth, visible satellites and UE links, CUSUM vs. A3 serving links, live calculations, and graphs (power difference, CUSUM statistic, cumulative HOs/ping-pongs). |

## Setup (virtual environment, not the base environment)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib
# ffmpeg is needed only to save the visualizer as .mp4
```

The system benchmark needs the simulator repository next to this folder:

```bash
git clone https://github.com/gdandil01/ele6701_project
```

## Run

```bash
python validate_theory.py                                   # ~10 min
python system_benchmark.py --sim-path ele6701_project       # ~20 min for the full sweep, 10 seeds
python make_figures.py
python handover_visualizer.py                               # live window
python handover_visualizer.py --duration 240 --save figures/handover_demo.mp4
```

## Parameters (as in the paper)

30 planes × 60 satellites, 550 km, 53°; UE in Montréal (45.5°N, 73.6°W); minimum elevation 35°; 30 GHz;
ITU-R S.1528-type satellite beam (40 dBi, ψ_b = 1°, −25 dB side lobes); SR light shadowing (b, m, Ω) = (0.158, 20, 1.29);
decision epoch 0.3 s; hysteresis 2 dB; ρ_HO = 2η.

## Main formulas implemented

* Gamma-mixture SR power: f_g(x) = Σ_k w_k x^(k−1) e^(−x/ϑ) / (Γ(k) ϑ^k), w_k = Binom(k−1; m−1, p), p = Ω/(2bm+Ω), ϑ = 2b + Ω/m.
* Ratio law: F_U(u) = Σ_k Σ_l w_k w_l I_{u/(1+u)}(k, l)  (regularised incomplete beta).
* Single-shot NP (A3 form): R_B − R_A > Δ_hys + 10 log10 F_Umax^{-1}(1−α).
* CUSUM: S_j(t) = max{0, S_j(t−1) + ln Λ(T_j(t))}; hand over to argmax S_j when it exceeds h.
* Run lengths: Brook–Evans Markov chain on [0, h) with the exact law of ln Λ(ρU).
