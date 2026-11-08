"""Short-term platform motion forecasting with wave preview from an up-wave
buoy (120 m ahead).

PG-MF (proposed): deterministic linear wave propagation of the buoy record to
the platform via the deep-water dispersion relation produces a physics-based
forecast of incident wave elevation at the platform over the horizon; a
regularised linear response model maps it (plus motion and wind history) to
the future motion.
Baselines: persistence, AR (motion history only), DD-ridge on raw buoy lags,
DD-MLP on the same raw inputs.
"""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

G = 9.80665
DX = 120.0        # buoy -> platform distance (m), waves travel +x
FS = 2.0          # forecasting rate (Hz)


def propagation_kernels(taus, Lpast=240, fs=FS, wlo=0.25, whi=2.0):
    """FIR kernels g_tau such that eta_p(t+tau) ~= sum_j g_tau[j] eta_b(t - j/fs).
    Built from H(w) = exp(-i k(w) DX) * exp(i w tau) (deep water, k = w^2/g),
    band-limited to [wlo, whi] rad/s and truncated to the causal past."""
    n = 4096
    w = 2 * np.pi * np.fft.rfftfreq(n, 1 / fs)
    band = (w >= wlo) & (w <= whi)
    taper = np.zeros_like(w)
    taper[band] = np.sin(np.pi * (w[band] - wlo) / (whi - wlo)) ** 0.5
    K = []
    for tau in taus:
        H = np.exp(-1j * (w**2 / G) * DX + 1j * w * tau) * taper
        h = np.fft.irfft(H, n)          # impulse response in lag samples (j>=0 causal)
        K.append(h[:Lpast])
    return np.array(K)


def lagmat(x, L, step=1):
    N = len(x); out = np.zeros((N, L))
    for j in range(L):
        s = j * step
        out[s:, j] = x[:N - s]; out[:s, j] = x[0]
    return out


def build(S, target, h, mode, kern=None, Lm=20, Lb=120):
    """Return X, y for all valid t; target channel forecast at t + h seconds."""
    dec = int(round(4.0 / FS))
    y_all = S[target][::dec]
    n = len(y_all); H = int(round(h * FS))
    eta = S['eta_buoy'][::dec][:n]
    u = S['u_hub'][::dec][:n]
    mot = lagmat(y_all, Lm, 2)                       # 20 s of motion history
    wind = lagmat(u, 10, 4)                          # 20 s of wind history
    if mode == 'AR':
        X = np.c_[mot, wind]
    elif mode == 'DD':
        X = np.c_[mot, wind, lagmat(eta, Lb, 1)]     # 60 s raw buoy history
    elif mode == 'PG':
        E = lagmat(eta, kern.shape[1], 1) @ kern.T  # propagated wave at platform, tau in [-30 s, h]
        X = np.c_[mot, wind, E]
    valid = np.arange(150, n - H)
    return X[valid], y_all[valid + H] - y_all[valid], y_all[valid]   # predict increment


class Forecaster:
    def __init__(self, target, h, mode, mlp=False):
        self.target, self.h, self.mode, self.mlp = target, h, mode, mlp
        self.kern = propagation_kernels(np.arange(-30, h + 0.01, 1.0)) if mode == 'PG' else None

    def fit(self, wins):
        X, y = [], []
        for w in wins:
            a, b, _ = build(w['S'], self.target, self.h, self.mode, self.kern); X.append(a); y.append(b)
        X, y = np.vstack(X), np.concatenate(y)
        if self.mlp:
            self.m = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=60,
                                                                   early_stopping=True, random_state=0)).fit(X, y)
        else:
            self.sc = StandardScaler().fit(X)
            self.m = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(self.sc.transform(X), y)
        return self

    def predict(self, w):
        X, y, y0 = build(w['S'], self.target, self.h, self.mode, self.kern)
        Xs = X if self.mlp else self.sc.transform(X)
        return self.m.predict(Xs) + y0, y + y0, y0


def score(pred, true, y0):
    e = pred - true
    return dict(RMSE=float(np.sqrt(np.mean(e**2))), NRMSE=float(np.sqrt(np.mean(e**2)) / np.std(true)),
                skill_vs_persistence=float(1 - np.mean(e**2) / np.mean((y0 - true) ** 2)))
