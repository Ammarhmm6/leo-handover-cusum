"""Analytical core of 'Handover in LEO Satellite Networks as a Hypothesis-Testing Problem'.

Shadowed-Rician (SR) power gain with integer m is a finite gamma mixture:
    f_g(x) = sum_k w_k Gamma(k, theta),  w_k = Binom(k-1; m-1, p),  p = Om/(2bm+Om),  theta = 2b + Om/m
The fading ratio U = g_j / g_A is then a finite mixture of beta-prime laws (closed form below).

Everything here is exact up to numerical quadrature / discretization.
"""
import numpy as np
from functools import lru_cache
from scipy.special import betainc, betaln, gammainc, gammaln, logsumexp, roots_genlaguerre
from scipy.stats import binom
from scipy.optimize import brentq


class SRModel:
    def __init__(self, m=20, b=0.158, Om=1.29, hys_db=2.0, rho_ho_over_eta=2.0):
        self.m, self.b, self.Om = int(m), b, Om
        self.eta = 10 ** (hys_db / 10)
        self.rho_ho = rho_ho_over_eta * self.eta
        self.k = np.arange(1, self.m + 1)
        self.w = binom.pmf(self.k - 1, self.m - 1, Om / (2 * b * self.m + Om))
        self.theta = 2 * b + Om / self.m
        K, L = np.meshgrid(self.k, self.k, indexing="ij")
        self.K, self.L, self.W = K, L, np.outer(self.w, self.w)
        self.LB = betaln(K, L)
        self._lg = None

    # ---------- single SR power gain ----------
    def F_g(self, x):
        x = np.asarray(x, float)
        return np.sum(self.w * gammainc(self.k, np.expand_dims(x, -1) / self.theta), axis=-1)

    def sample_g(self, n, rng):
        Z = rng.gamma(self.m, self.Om / self.m, n)
        h = np.sqrt(Z) * np.exp(1j * rng.uniform(0, 2 * np.pi, n)) + np.sqrt(self.b) * (
            rng.standard_normal(n) + 1j * rng.standard_normal(n))
        return np.abs(h) ** 2

    # ---------- ratio U = g_j / g_A (mixture of beta-prime) ----------
    def F_U(self, u):
        u = np.atleast_1d(np.asarray(u, float)); x = u / (1 + u)
        return np.einsum("kl,klu->u", self.W, betainc(self.K[..., None], self.L[..., None], x[None, None, :]))

    def log_f_U(self, u):
        u = np.atleast_1d(np.asarray(u, float))[None, None, :]
        t = (np.log(self.W)[..., None] + (self.K[..., None] - 1) * np.log(u)
             - (self.K + self.L)[..., None] * np.log1p(u) - self.LB[..., None])
        return logsumexp(t, axis=(0, 1))

    # ---------- likelihood ratio of T = rho*U :  H1 rho=rho_ho  vs  H0 rho=eta ----------
    def llr(self, T):
        return np.log(self.eta / self.rho_ho) + self.log_f_U(T / self.rho_ho) - self.log_f_U(T / self.eta)

    def llr_fast(self, T):
        """Interpolated LLR (for simulation loops)."""
        if self._lg is None:
            self._lg = np.linspace(np.log(1e-9), np.log(1e9), 200001)
            self._llr_tab = self.llr(np.exp(self._lg))
        return np.interp(np.log(T), self._lg, self._llr_tab)

    # ---------- NP threshold for max-ratio test over N neighbours ----------
    @lru_cache(maxsize=None)
    def _quad(self, order=64):
        xs, ws = [], []
        for shape, mw in zip(self.k, self.w):
            z, wz = roots_genlaguerre(order, float(shape - 1))
            xs.append(self.theta * z); ws.append(mw * wz / np.exp(gammaln(shape)))
        return np.concatenate(xs), np.concatenate(ws)

    def F_Umax(self, u, N):
        x, wq = self._quad()
        return float(np.dot(wq, self.F_g(u * x) ** N))

    def np_threshold(self, alpha, N=1):
        """gamma such that P(max_j T_j > gamma | all rho_j = eta) = alpha."""
        uq = brentq(lambda u: self.F_Umax(u, N) - (1 - alpha), 1e-8, 1e10)
        return self.eta * uq

    def p_trigger(self, gamma, rho, N=1):
        """P(max_j T_j > gamma) when one neighbour has ratio rho and N-1 sit at eta."""
        x, wq = self._quad()
        return 1 - float(np.dot(wq, self.F_g(gamma * x / rho) * self.F_g(gamma * x / self.eta) ** (N - 1)))

    # ---------- CUSUM: exact law of the increment and ARL (Brook-Evans Markov chain) ----------
    @lru_cache(maxsize=None)
    def _u_grid(self, n=40000):
        E = np.logspace(-7, 7, n + 1)
        P = np.diff(np.concatenate([[0.0], self.F_U(E), [1.0]]))
        mid = np.concatenate([[E[0] / 10], np.sqrt(E[:-1] * E[1:]), [E[-1] * 10]])
        return mid, P

    def increment_law(self, rho):
        mid, P = self._u_grid()
        return self.llr(rho * mid), P

    def arl(self, h, rho, n=1500):
        """Average run length (decision epochs) of the CUSUM started at 0 when the true ratio is rho.
        rho <= eta: mean time to a false alarm; rho > eta: mean detection delay."""
        z, p = self.increment_law(rho)
        wd = h / n
        s = np.concatenate([[0.0], (np.arange(1, n + 1) - 0.5) * wd])
        R = np.zeros((n + 1, n + 1))
        for i, si in enumerate(s):
            sn = si + z
            neg = sn <= 0
            R[i, 0] += p[neg].sum()
            ins = (~neg) & (sn < h)
            np.add.at(R[i], np.minimum(np.ceil(sn[ins] / wd).astype(int), n), p[ins])
        return float(np.linalg.solve(np.eye(n + 1) - R, np.ones(n + 1))[0])

    def h_for_arl0(self, target_epochs, rho=None):
        """Smallest h whose false-alarm ARL at ratio rho (default: boundary eta) reaches the target."""
        rho = self.eta if rho is None else rho
        return brentq(lambda h: self.arl(h, rho, n=600) - target_epochs, 0.05, 8.0, xtol=1e-3)
