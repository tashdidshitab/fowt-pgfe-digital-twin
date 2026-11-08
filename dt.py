"""Physics-guided digital twin: mooring degradation estimation from sparse
sensors (PG-FE), with data-driven baselines.

PG-FE idea: the quasi-static mooring model F_moor(q; delta) is accurate for
window-mean loads (validated vs MoorDyn to <0.2%), while the environmental
loading (rotor thrust, wave drift, tower drag) is *independent of mooring
health* and can be learned from SCADA + buoy data. Window-mean equilibrium
   sum_i Fbar_i(q(t); delta_i) + F_env = 0       (surge, sway, yaw [+ heave, roll, pitch])
is inverted for delta with a grid-exact Bayesian posterior.
"""
import numpy as np, json, warnings, zlib
from scipy.signal import welch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import physics, sensors, extract

DG = physics.DG
R_ROTOR = 63.0
COMP_H = [0, 1, 5]          # surge force, sway force, yaw moment
COMP_ALL = [0, 1, 2, 3, 4, 5]


# ------------------------------------------------------------------ windows
def windows(name, W=400.0, chans=None, noise=1.0, seed_off=0, offset=0.0):
    D = extract.load(name)
    t0s = np.arange(120.0, 520.0 - W + 1e-6, W)
    out = []
    for k, t0 in enumerate(t0s):
        seed = zlib.crc32(f'{name}|{k}|{seed_off}|{W}|{noise}'.encode())
        S = sensors.measure(D, chans, noise, seed, t0, W, offset)
        truth = {c: D[c][(D['t'] >= t0) & (D['t'] < t0 + W)] for c in ['T1', 'T2', 'T3', 'twr_my', 'thrust',
                                                                        'surge', 'sway', 'heave', 'roll', 'pitch', 'yaw']}
        out.append(dict(name=name, k=k, t0=t0, S=S, truth=truth, meta=D['meta'], delta=np.array(D['meta']['delta'])))
    return out


# ------------------------------------------------------------- thrust model
def scada_X(S):
    u = np.maximum(S['u_hub'], 1.0)
    om = S['rotspd'] * 2 * np.pi / 60
    lam = om * R_ROTOR / u
    return np.c_[u, S['rotspd'], S['pitchb'], S['genpwr'], lam, S['genpwr'] / u, u**2]


def block_mean(x, n=4):
    m = len(x) // n * n
    return x[:m].reshape(m // n, n, *x.shape[1:]).mean(1)


class ThrustModel:
    """SCADA -> rotor thrust (1-s resolution). Damage-independent."""
    def fit(self, wins):
        X = np.vstack([block_mean(scada_X(w['S'])) for w in wins])
        y = np.concatenate([block_mean(w['truth']['thrust'][::5][:len(w['S']['t'])]) for w in wins])
        n = min(len(X), len(y))
        self.m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31).fit(X[:n], y[:n] / 1e6)
        return self

    def predict(self, S):
        return self.m.predict(scada_X(S)) * 1e6


# ---------------------------------------------------------- env features
def wave_stats(x, fs=sensors.FS):
    f, P = welch(x - x.mean(), fs=fs, nperseg=min(len(x), 256))
    band = (f > 0.04) & (f < 0.4)
    m0 = np.trapezoid(P[band], f[band]); m1 = np.trapezoid(P[band] * f[band], f[band])
    return 4 * np.sqrt(m0), m0 / max(m1, 1e-9)


def env_features(w, thrust_model):
    S = w['S']
    T = thrust_model.predict(S)
    u, v = S['u_hub'], S['v_hub']
    psi = np.arctan2(v.mean(), u.mean())
    if 'eta_buoy' in S:
        Hs, Tm = wave_stats(S['eta_buoy'])
    else:  # wave proxy from heave response (heave is insensitive to mooring health)
        Hs, Tm = wave_stats(S['heave'])
    return np.array([T.mean() / 1e6, T.std() / 1e6, u.mean(), u.std(), psi, S['rotspd'].mean(), S['pitchb'].mean(),
                     S['genpwr'].mean() / 1e3, Hs, Tm, Hs**2, Hs**2 / Tm, T.mean() / 1e6 * np.sin(psi), u.mean()**2 * np.sin(psi) / 100])


# ------------------------------------------------------ physics precompute
def mooring_table(w, step=1):
    """Fbar[i, k, c]: window-mean load of line i for degradation DG[k]."""
    q = sensors.pose(w['S'])[::step]
    F = np.zeros((3, len(DG), 6))
    for i in range(3):
        for k, d in enumerate(DG):
            F[i, k] = physics.line_loads(q, d, i)[0].mean(0)
    return F


def mooring_true(w):
    q = sensors.pose(w['S'])
    return sum(physics.line_loads(q, w['delta'][i], i)[0].mean(0) for i in range(3))


SCALE = np.array([1e6, 1e6, 1e6, 1e7, 1e7, 1e7])


# ------------------------------------------------------------ env model
class EnvModel:
    """Learns the mean environmental load (negative of mean mooring load in
    horizontal DOFs) from damage-independent features. For heave/roll/pitch
    the hydrostatic restoring is included through the measured mean pose."""
    def __init__(self, comps=COMP_H, kind='gp'):
        self.comps, self.kind = comps, kind

    def _X(self, feats, poses):
        X = feats.copy()
        if any(c in (2, 3, 4) for c in self.comps):
            X = np.c_[X, poses[:, [2, 3, 4]]]
        return X

    def fit(self, feats, poses, targets):
        X = self._X(feats, poses)
        self.sc = StandardScaler().fit(X)
        Xs = self.sc.transform(X)
        self.models = []
        for j, c in enumerate(self.comps):
            y = targets[:, c] / SCALE[c]
            if self.kind == 'gp':
                k = C(1.0) * RBF(length_scale=np.ones(Xs.shape[1]) * 3, length_scale_bounds=(0.1, 1e3)) + WhiteKernel(1e-2, (1e-6, 1))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    m = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=0).fit(Xs, y)
            else:
                from sklearn.linear_model import RidgeCV
                m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xs, y)
            self.models.append(m)
        return self

    def predict(self, feats, poses):
        Xs = self.sc.transform(self._X(feats, poses))
        mu, sd = [], []
        for c, m in zip(self.comps, self.models):
            if self.kind == 'gp':
                a, b = m.predict(Xs, return_std=True)
            else:
                a, b = m.predict(Xs), np.zeros(len(Xs))
            mu.append(a * SCALE[c]); sd.append(b * SCALE[c])
        return np.array(mu).T, np.array(sd).T


# ---------------------------------------------------------- PG inversion
def posterior(Ftab, Fenv, sig, comps, prior='uniform', p0=0.5):
    """Grid-exact posterior over (delta1, delta2, delta3).
    Ftab (3,K,6), Fenv (len(comps),), sig (len(comps),)."""
    K = len(DG)
    tot = (Ftab[0][:, None, None, comps] + Ftab[1][None, :, None, comps] + Ftab[2][None, None, :, comps]) + Fenv
    ll = -0.5 * np.sum((tot / sig) ** 2, -1)
    lp = np.zeros(K)
    if prior == 'spike':
        lp = np.full(K, np.log((1 - p0) / (K - 1))); lp[0] = np.log(p0)
    ll = ll + lp[:, None, None] + lp[None, :, None] + lp[None, None, :]
    ll -= ll.max()
    P = np.exp(ll); P /= P.sum()
    marg = [P.sum((1, 2)), P.sum((0, 2)), P.sum((0, 1))]
    mean = np.array([m @ DG for m in marg])
    imap = np.unravel_index(np.argmax(P), P.shape)
    mapv = DG[list(imap)]
    ci = []
    for m in marg:
        c = np.cumsum(m)
        ci.append((DG[np.searchsorted(c, 0.05)], DG[min(np.searchsorted(c, 0.95), K - 1)]))
    p_dam = np.array([1 - m[0] - m[1] for m in marg])   # P(delta_i > 0.025)
    return dict(mean=mean, map=mapv, ci=np.array(ci), marg=np.array(marg), p_damaged=p_dam)


class PGFE:
    """Physics-guided fault estimator."""
    def __init__(self, comps=COMP_H, env_kind='gp', prior='uniform', p0=0.5):
        self.comps, self.env_kind, self.prior, self.p0 = comps, env_kind, prior, p0

    def fit(self, wins, cache):
        self.thrust = ThrustModel().fit(wins)
        feats = np.array([env_features(w, self.thrust) for w in wins])
        poses = np.array([sensors.pose(w['S']).mean(0) for w in wins])
        targ = -np.array([cache[(w['name'], w['k'])]['Ftrue'] for w in wins])
        # out-of-fold residuals -> noise level for the likelihood
        oof = np.zeros((len(wins), len(self.comps)))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(feats):
            m = EnvModel(self.comps, self.env_kind).fit(feats[tr], poses[tr], targ[tr])
            oof[te] = m.predict(feats[te], poses[te])[0]
        self.sig = np.std(oof - targ[:, self.comps], 0)
        self.env = EnvModel(self.comps, self.env_kind).fit(feats, poses, targ)
        return self

    def predict(self, w, cache):
        f = env_features(w, self.thrust)[None]
        p = sensors.pose(w['S']).mean(0)[None]
        mu, sd = self.env.predict(f, p)
        sig = np.sqrt(self.sig**2)
        return posterior(cache[(w['name'], w['k'])]['Ftab'], mu[0], sig, self.comps, self.prior, self.p0)


# ------------------------------------------------- data-driven baselines
def dd_features(w):
    S = w['S']
    fs = []
    for c in sorted(k for k in S if k != 't'):
        x = S[c]
        f, P = welch(x - x.mean(), fs=sensors.FS, nperseg=min(len(x), 256))
        lf = np.trapezoid(P[f < 0.04], f[f < 0.04]); wf = np.trapezoid(P[(f >= 0.04) & (f < 0.4)], f[(f >= 0.04) & (f < 0.4)])
        fpk = f[1:][np.argmax(P[1:])]
        fs += [x.mean(), x.std(), np.log(lf + 1e-12), np.log(wf + 1e-12), fpk]
    return np.array(fs)


# ------------------------------------------------ commissioning calibration
def calibrate_anchors(pg, cal_wins, base_pert, step=4):
    """Estimate per-line effective anchor-radius correction (absorbs as-installed
    anchor position and unstretched-length error) from windows recorded while
    the mooring is known to be healthy (commissioning period). Minimises the
    same normalised equilibrium residual the estimator uses, with delta = 0."""
    import physics
    from scipy.optimize import minimize
    qs, mus = [], []
    for w in cal_wins:
        f = env_features(w, pg.thrust)[None]; p = sensors.pose(w['S']).mean(0)[None]
        mus.append(pg.env.predict(f, p)[0][0]); qs.append(sensors.pose(w['S'])[::step])
    def obj(c):
        physics.ANCH_PERT = base_pert + c
        J = 0.0
        for q, mu in zip(qs, mus):
            tot = sum(physics.line_loads(q, 0.0, i)[0].mean(0) for i in range(3))[pg.comps] + mu
            J += np.sum((tot / pg.sig) ** 2)
        return J
    # coarse-to-fine: Nelder-Mead with restarts from the incumbent (large initial
    # geometry errors otherwise stall in the flat, strongly coupled valley)
    x = np.zeros(3); best = obj(x)
    for it in range(6):
        res = minimize(obj, x, method='Nelder-Mead',
                       options=dict(xatol=0.01, fatol=1e-4, maxiter=800, initial_simplex=x + np.vstack([np.zeros(3), 2.0 * np.eye(3)])))
        if best - res.fun < 1e-3 * max(best, 1e-9):
            x = res.x; break
        x, best = res.x, res.fun
    physics.ANCH_PERT = base_pert.copy()
    return x
