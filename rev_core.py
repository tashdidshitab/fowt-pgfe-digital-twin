"""Revision: estimator variants, physics baselines and consistent metrics.

Definitions used everywhere in the revised paper
  delta*        = 0.05 (5 % section loss) : damage threshold for ground truth and alarms
  truth damaged : delta_i > delta*;  intact run: all delta_i == 0;  sub-threshold: 0 < delta_i <= delta*
  PG-FE alarm   : P(delta_i > delta* | y) > 0.5   (grid point = bin centre, half-bin rule at delta*)
  baselines     : delta_hat_i > delta*
"""
from paths import ROOT, RTEST
import numpy as np, warnings
from scipy.optimize import minimize
from scipy.stats import beta as _beta
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
import dt, sensors, physics

DG = dt.DG; DSTEP = DG[1] - DG[0]; DSTAR = 0.05
COMPS = dt.COMP_ALL


# ---------------------------------------------------------------------------- estimator
class PGFE2:
    """Physics-guided fault estimator, revised.
    train   : windows used to fit the thrust and environmental-load models
    cov     : 'diag' | 'full'   likelihood covariance of the 6 equilibrium residuals
    hetero  : residual scale grows with predicted environmental load (fitted on CV residuals)
    p0      : prior P(delta_i = 0) per line (spike-and-slab)
    nuis    : None | 'mass'   marginalise an unknown platform weight change (vertical load)
    """
    def __init__(self, cov='full', hetero=True, p0=0.9, nuis=None, comps=COMPS, wmax=0.8e6, nw=33):
        self.cov, self.hetero, self.p0, self.nuis, self.comps = cov, hetero, p0, nuis, list(comps)
        self.wgrid = np.linspace(-wmax, wmax, nw) if nuis == 'mass' else np.zeros(1)

    def fit(self, wins):
        self.thrust = dt.ThrustModel().fit(wins)
        F = np.array([dt.env_features(w, self.thrust) for w in wins])
        P = np.array([sensors.pose(w['S']).mean(0) for w in wins])
        Y = -np.array([w['Ftrue'] for w in wins])[:, self.comps]       # = -sum_i Fbar_i(delta_i) (known in simulation)
        oof = np.zeros_like(Y)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(F):
            m = dt.EnvModel(self.comps, 'lin').fit(F[tr], P[tr], -np.array([wins[j]['Ftrue'] for j in tr]))
            oof[te] = m.predict(F[te], P[te])[0]
        R = (oof - Y) / dt.SCALE[self.comps]
        self.env = dt.EnvModel(self.comps, 'lin').fit(F, P, -np.array([w['Ftrue'] for w in wins]))
        # heteroscedastic scale: |r_c| ~ a_c + b_c * s, s = load level (thrust, wave drift proxy)
        S = self._lvl(F)
        if self.hetero:
            A = np.c_[np.ones(len(S)), S]
            self.hb = np.array([np.linalg.lstsq(A, np.abs(R[:, c]), rcond=None)[0] for c in range(R.shape[1])])
            self.hb = np.maximum(self.hb, 1e-6)
            scale = A @ self.hb.T
        else:
            scale = np.ones_like(R)
        Z = R / scale                                    # standardised residuals
        self.C = np.cov(Z.T) if self.cov == 'full' else np.diag(np.var(Z, 0))
        self.Ci = np.linalg.inv(self.C)
        self.resid = R
        return self

    @staticmethod
    def _lvl(F):
        return np.c_[F[:, 0], F[:, 10] / 10]           # mean thrust (MN), Hs^2/10

    def env_pred(self, w):
        f = dt.env_features(w, self.thrust)[None]
        p = sensors.pose(w['S']).mean(0)[None]
        mu = self.env.predict(f, p)[0][0]
        sc = (np.r_[1.0, self._lvl(f)[0]] @ self.hb.T) if self.hetero else np.ones(len(self.comps))
        return mu, sc

    def loglik(self, w, Ftab=None, mu=None):
        """log-likelihood grid (K,K,K[,W]) of window w."""
        Ftab = w['Ftab'] if Ftab is None else Ftab
        m_, sc = self.env_pred(w)
        mu = m_ if mu is None else mu
        c = self.comps
        tot = (Ftab[0][:, None, None, c] + Ftab[1][None, :, None, c] + Ftab[2][None, None, :, c]) + mu
        tot = tot[..., None, :] + np.outer(self.wgrid, np.eye(6)[2][c])[None, None, None]   # weight nuisance on Fz
        z = tot / (dt.SCALE[c] * sc)
        ll = -0.5 * np.einsum('...i,ij,...j->...', z, self.Ci, z)
        return ll

    def logprior(self):
        K = len(DG)
        lp = np.full(K, np.log((1 - self.p0) / (K - 1))); lp[0] = np.log(self.p0)
        return lp[:, None, None, None] + lp[None, :, None, None] + lp[None, None, :, None]

    def post_from_ll(self, ll):
        lp = ll + self.logprior()
        lp -= lp.max(); P = np.exp(lp); P /= P.sum()
        P3 = P.sum(-1)
        marg = np.array([P3.sum((1, 2)), P3.sum((0, 2)), P3.sum((0, 1))])
        return summarise(marg, P3, P.sum((0, 1, 2)))

    def predict(self, w, mu=None):
        return self.post_from_ll(self.loglik(w, mu=mu))


def summarise(marg, P3=None, pw=None):
    mean = marg @ DG
    cdf = np.cumsum(marg, 1)
    # bin-centre convention: grid point k represents [DG_k - step/2, DG_k + step/2)
    def interval(a):
        lo = np.array([DG[np.searchsorted(c, (1 - a) / 2)] - DSTEP / 2 for c in cdf])
        hi = np.array([DG[min(np.searchsorted(c, 1 - (1 - a) / 2), len(DG) - 1)] + DSTEP / 2 for c in cdf])
        return np.c_[np.maximum(lo, 0), hi]
    ci = {a: interval(a) for a in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)}
    kstar = int(np.argmin(np.abs(DG - DSTAR)))
    p_exc = marg[:, kstar + 1:].sum(1) + 0.5 * marg[:, kstar]       # P(delta_i > delta*)
    out = dict(mean=mean, ci=ci, p_exc=p_exc, marg=marg)
    if P3 is not None:
        out['map'] = DG[list(np.unravel_index(np.argmax(P3), P3.shape))]
    if pw is not None:
        out['pw'] = pw
    return out


# ------------------------------------------------------------- physics-based baselines
def _interp_F(Ftab, d):
    """continuous Fbar for each line, linear in delta between grid points. Ftab (3,K,6)."""
    out = np.zeros(6)
    for i in range(3):
        x = np.clip(d[i], 0, DG[-1]) / DSTEP; k = min(int(x), len(DG) - 2); a = x - k
        out += (1 - a) * Ftab[i, k] + a * Ftab[i, k + 1]
    return out


class PhysicsBaselines:
    """Same catenary core and environmental-load predictor as PG-FE.
    WLS : weighted least squares (Mahalanobis, same covariance), bound-constrained
    L1  : WLS + lambda * sum(delta)  (sparse regularised inverse)
    EKF : extended Kalman filter over consecutive sub-windows (random-walk delta)"""
    def __init__(self, pg):
        self.pg = pg

    def _obj(self, d, Ftab, mu, sc, lam=0.0):
        z = (_interp_F(Ftab, d)[self.pg.comps] + mu) / (dt.SCALE[self.pg.comps] * sc)
        return 0.5 * z @ self.pg.Ci @ z + lam * np.sum(d)

    def wls(self, w, lam=0.0):
        mu, sc = self.pg.env_pred(w)
        best = None
        for x0 in ([0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2], [0.15, 0.15, 0.15]):
            r = minimize(self._obj, np.array(x0, float), args=(w['Ftab'], mu, sc, lam), method='L-BFGS-B',
                         bounds=[(0, DG[-1])] * 3)
            if best is None or r.fun < best.fun:
                best = r
        return best.x

    def ekf(self, subwins, q=1e-5, p0=0.04):
        """subwins: consecutive windows (each with Ftab) of one record; returns final state and history."""
        x = np.zeros(3); Pm = np.eye(3) * p0; hist = []
        for w in subwins:
            Pm = Pm + np.eye(3) * q
            mu, sc = self.pg.env_pred(w)
            S = dt.SCALE[self.pg.comps] * sc
            h = lambda d: (_interp_F(w['Ftab'], d)[self.pg.comps] + mu) / S
            Hj = np.zeros((len(S), 3)); e = 1e-3
            for k in range(3):
                dp = x.copy(); dp[k] += e; dm = x.copy(); dm[k] = max(dm[k] - e, 0)
                Hj[:, k] = (h(dp) - h(dm)) / (dp[k] - dm[k])
            Rm = self.pg.C
            Sg = Hj @ Pm @ Hj.T + Rm
            K = Pm @ Hj.T @ np.linalg.inv(Sg)
            x = np.clip(x + K @ (0 - h(x)), 0, DG[-1])
            Pm = (np.eye(3) - K @ Hj) @ Pm
            hist.append(x.copy())
        return x, np.sqrt(np.diag(Pm)), hist


# ----------------------------------------------------------------------- identifiability
def jacobian(Ftab, d0=(0, 0, 0)):
    """J (6x3): dFbar_c/d delta_i at d0 (one-sided difference on the grid)."""
    J = np.zeros((6, 3))
    for i in range(3):
        k = int(round(d0[i] / DSTEP)); k = min(k, len(DG) - 2)
        J[:, i] = (Ftab[i, k + 1] - Ftab[i, k]) / DSTEP
    return J


# ---------------------------------------------------------------------------- metrics
def cp_interval(k, n, a=0.05):
    lo = 0.0 if k == 0 else _beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def cp_upper_one_sided(k, n, a=0.05):
    return 1.0 if k == n else _beta.ppf(1 - a, k + 1, n - k)


def detection_metrics(true, alarm):
    """true (N,3) delta; alarm (N,3) bool. Consistent definitions (see module doc)."""
    true = np.asarray(true); alarm = np.asarray(alarm, bool)
    dam = true > DSTAR; sub = (true > 0) & (true <= DSTAR)
    intact_run = (true == 0).all(1)
    dam_run = dam.any(1)
    out = {}
    # run level, sub-threshold-only runs excluded
    keep = intact_run | dam_run
    fa = alarm[intact_run].any(1)
    out['intact_n'] = int(intact_run.sum()); out['intact_fa_k'] = int(fa.sum())
    out['intact_fa'] = float(fa.mean()) if len(fa) else np.nan
    out['intact_fa_ci'] = cp_interval(int(fa.sum()), int(intact_run.sum()))
    out['intact_fa_upper95'] = cp_upper_one_sided(int(fa.sum()), int(intact_run.sum()))
    det = alarm[dam_run].any(1)
    out['run_recall'] = float(det.mean()) if len(det) else np.nan
    tp = int(det.sum()); fp = int(fa.sum()); fn = int((~det).sum())
    out['run_F1'] = 2 * tp / max(2 * tp + fp + fn, 1)
    # line level (lines with sub-threshold truth excluded)
    lk = ~sub
    TP = int((alarm & dam & lk).sum()); FP = int((alarm & ~dam & lk).sum()); FN = int((~alarm & dam & lk).sum())
    out['line_precision'] = TP / max(TP + FP, 1); out['line_recall'] = TP / max(TP + FN, 1)
    out['line_precision_ci'] = cp_interval(TP, TP + FP); out['line_recall_ci'] = cp_interval(TP, TP + FN)
    # exact identification of the damaged set
    exact = [(alarm[k] == dam[k])[lk[k]].all() for k in range(len(true)) if keep[k]]
    out['exact_set_acc'] = float(np.mean(exact)) if exact else np.nan
    for nd in (1, 2):
        sel = [k for k in range(len(true)) if keep[k] and dam[k].sum() == nd]
        out[f'exact_set_acc_{nd}line'] = float(np.mean([(alarm[k] == dam[k]).all() for k in sel])) if sel else np.nan
        out[f'n_{nd}line'] = len(sel)
    # confusion over number of alarmed lines vs number of damaged lines
    cm = np.zeros((3, 4), int)
    for k in range(len(true)):
        if keep[k]:
            cm[int(dam[k].sum()), int(alarm[k].sum())] += 1
    out['confusion_ndam_vs_nalarm'] = cm.tolist()
    out['n_subthreshold_lines'] = int(sub.sum()); out['subthreshold_alarm'] = int((alarm & sub).sum())
    return out


def severity_metrics(true, pred, rng=None, nboot=2000):
    true = np.asarray(true); pred = np.asarray(pred)
    err = np.abs(pred - true); dam = true > DSTAR
    out = dict(MAE_all=float(err.mean()), MAE_dam=float(err[dam].mean()) if dam.any() else np.nan,
               bias_dam=float((pred - true)[dam].mean()) if dam.any() else np.nan,
               MAE_intact_lines=float(err[true == 0].mean()) if (true == 0).any() else np.nan)
    # run-level bootstrap CI (resample runs)
    rng = rng or np.random.default_rng(0)
    N = len(true)
    if dam.any():
        bs = []
        for _ in range(nboot):
            ix = rng.integers(0, N, N); d = dam[ix]
            if d.any():
                bs.append(err[ix][d].mean())
        out['MAE_dam_ci'] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return out


def coverage(true, cis, levels=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    true = np.asarray(true)
    return {a: float(np.mean((true >= cis[a][:, :, 0] - 1e-12) & (true <= cis[a][:, :, 1] + 1e-12))) for a in levels}


# --------------------------------------------- physics hydrostatic + gravity restoring
HS_FILE = ROOT + '/static/hydrostatic_fit.json'


def fit_hydrostatics():
    """Identify F_hs(q) = a + K [heave, roll, pitch] for (Fz, Mx, My) from static OpenFAST tests
    on the healthy design model: F_applied + F_moor(q) + F_hs(q) = 0."""
    import json, os
    from openfast_toolbox.io import FASTOutputFile
    tests = {'base': [0] * 6, 'fz+': [0, 0, 1.0e6, 0, 0, 0], 'fz-': [0, 0, -1.0e6, 0, 0, 0], 'mx+': [0, 0, 0, 4e7, 0, 0],
             'mx-': [0, 0, 0, -4e7, 0, 0], 'my+': [0, 0, 0, 0, 4e7, 0], 'my-': [0, 0, 0, 0, -4e7, 0],
             'fzmy': [0, 0, 1.0e6, 0, 4e7, 0], 'fzmx': [0, 0, -1.0e6, 4e7, 0, 0]}
    Q, Fh = [], []
    for nm, f0 in tests.items():
        df = FASTOutputFile(f'{ROOT}/static/{nm}/5MW_OC4Semi_WSt_WavesWN.outb').toDataFrame()
        df = df[df['Time_[s]'] >= 200]
        q = np.array([df[c].mean() for c in ['PtfmSurge_[m]', 'PtfmSway_[m]', 'PtfmHeave_[m]', 'PtfmRoll_[deg]', 'PtfmPitch_[deg]', 'PtfmYaw_[deg]']])
        q[3:] = np.radians(q[3:])
        Fm = sum(physics.line_loads(q[None], 0.0, i)[0][0] for i in range(3))
        Fh.append(-np.array(f0) - Fm); Q.append(q)
    Q, Fh = np.array(Q), np.array(Fh)
    A = np.c_[np.ones(len(Q)), Q[:, [2, 3, 4]]]
    coef = {c: np.linalg.lstsq(A, Fh[:, c], rcond=None)[0].tolist() for c in (2, 3, 4)}
    res = {c: float(np.abs(A @ np.array(coef[c]) - Fh[:, c]).max()) for c in (2, 3, 4)}
    out = dict(coef=coef, max_resid=res, Q=Q.tolist(), Fh=Fh.tolist())
    json.dump(out, open(HS_FILE, 'w'), indent=1)
    return out


def F_hs(qbar):
    import json
    co = json.load(open(HS_FILE))['coef']
    out = np.zeros(6)
    for c in (2, 3, 4):
        a = np.array(co[str(c)] if str(c) in co else co[c])
        out[c] = a[0] + a[1:] @ qbar[[2, 3, 4]]
    return out


def feats_phys(w, thrust):
    """damage-independent features: SCADA + buoy (no pose), plus thrust rotated by measured heading."""
    f = dt.env_features(w, thrust)
    yaw = sensors.pose(w['S'])[:, 5].mean()
    T = f[0]
    return np.r_[f, T * np.cos(yaw), T * np.sin(yaw), f[11] * np.sin(yaw)]


class EnvPhys:
    """mu_env = F_hs(qbar) [physics, design model] + ridge(features) [learned, pose-free]."""
    def __init__(self, comps=COMPS, use_hs=True):
        self.comps, self.use_hs = list(comps), use_hs

    def target(self, w):
        y = -w['Ftrue']
        if self.use_hs:
            y = y - F_hs(sensors.pose(w['S']).mean(0))
        return y

    def fit(self, X, Y):
        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler().fit(X)
        self.m = [RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(self.sc.transform(X), Y[:, c] / dt.SCALE[c]) for c in self.comps]
        return self

    def predict(self, X, qbar):
        Xs = self.sc.transform(X)
        mu = np.array([m.predict(Xs) * dt.SCALE[c] for m, c in zip(self.m, self.comps)]).T
        if self.use_hs:
            mu = mu + np.array([F_hs(q)[self.comps] for q in qbar])
        return mu


class PGFE3(PGFE2):
    """Revised PG-FE: physics hydrostatics + pose-free learned environmental load (deployable with
    healthy-only training)."""
    def __init__(self, use_hs=True, **kw):
        super().__init__(**kw); self.use_hs = use_hs

    def fit(self, wins):
        self.thrust = dt.ThrustModel().fit(wins)
        X = np.array([feats_phys(w, self.thrust) for w in wins])
        Q = np.array([sensors.pose(w['S']).mean(0) for w in wins])
        e = EnvPhys(self.comps, self.use_hs)
        Yt = np.array([e.target(w) for w in wins])
        Ytrue = -np.array([w['Ftrue'] for w in wins])[:, self.comps]
        oof = np.zeros((len(wins), len(self.comps)))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            m = EnvPhys(self.comps, self.use_hs).fit(X[tr], Yt[tr]); oof[te] = m.predict(X[te], Q[te])
        R = (oof - Ytrue) / dt.SCALE[self.comps]
        self.env = EnvPhys(self.comps, self.use_hs).fit(X, Yt)
        F = np.array([dt.env_features(w, self.thrust) for w in wins])
        Sl = self._lvl(F)
        if self.hetero:
            A = np.c_[np.ones(len(Sl)), Sl]
            self.hb = np.maximum(np.array([np.linalg.lstsq(A, np.abs(R[:, c]), rcond=None)[0] for c in range(R.shape[1])]), 1e-6)
            scale = A @ self.hb.T
        else:
            scale = np.ones_like(R)
        Z = R / scale
        self.C = np.cov(Z.T) if self.cov == 'full' else np.diag(np.var(Z, 0))
        self.Ci = np.linalg.inv(self.C); self.resid = R
        return self

    def env_pred(self, w):
        X = feats_phys(w, self.thrust)[None]; q = sensors.pose(w['S']).mean(0)[None]
        mu = self.env.predict(X, q)[0]
        F = dt.env_features(w, self.thrust)[None]
        sc = (np.r_[1.0, self._lvl(F)[0]] @ self.hb.T) if self.hetero else np.ones(len(self.comps))
        return mu, sc


# --------------------------------------------- data-driven baselines with CV hyperparameter selection
def tuned_dd(kind, train_wins, seed=0):
    """GridSearchCV (5-fold, run-grouped, neg. MAE) over a small grid; returns fitted predictor and the
    selected hyperparameters. Features: response statistics of all available channels (dt.dd_features)."""
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.array([w['ddf'] for w in train_wins]); Y = np.array([w['delta'] for w in train_wins])
    groups = [w['name'] for w in train_wins]
    if kind == 'DD-GBM':
        est = MultiOutputRegressor(HistGradientBoostingRegressor(random_state=seed))
        grid = {'estimator__learning_rate': [0.03, 0.1], 'estimator__max_iter': [200, 500],
                'estimator__max_leaf_nodes': [7, 15, 31], 'estimator__min_samples_leaf': [3, 10]}
    elif kind == 'DD-RF':
        est = RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=1)
        grid = {'min_samples_leaf': [1, 2, 5], 'max_features': [0.33, 0.66, 1.0]}
    else:
        est = make_pipeline(StandardScaler(), MLPRegressor(max_iter=1000, random_state=seed))
        grid = {'mlpregressor__hidden_layer_sizes': [(32,), (64, 64)], 'mlpregressor__alpha': [1e-2, 1e-1, 1.0]}
    gs = GridSearchCV(est, grid, cv=GroupKFold(5), scoring='neg_mean_absolute_error', n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gs.fit(X, Y, groups=groups)
    best = gs.best_estimator_
    return (lambda Z: np.clip(best.predict(np.atleast_2d(Z)), 0, 0.6)), gs.best_params_
