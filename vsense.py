"""Virtual sensing + fatigue: reconstruct un-instrumented fairlead tensions
and tower-base fore-aft moment from sparse sensors, then compute damage-
equivalent loads (DEL) with rainflow counting.

Models
  QS      : quasi-static catenary tension at the measured pose with the twin's
            current degradation estimate (no data fitting)
  PG-VS   : QS + learned dynamic residual (ridge FIR on causal lags of fairlead
            kinematics and QS tension)                       <- proposed
  DD-VS   : ridge FIR / GBM directly from the same sensor lags (no physics)
For the tower base: rigid-body physics (thrust x lever + RNA/tower inertial and
gravity terms) with calibrated coefficients, + learned residual.
"""
import numpy as np, rainflow
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
import physics, sensors

FS = sensors.FS
M_CHAIN, M_STEEL = 3.0, 4.0


def DEL(x, m, fs, neq_hz=1.0):
    cyc = rainflow.count_cycles(np.asarray(x, float))
    if not cyc:
        return 0.0
    r, n = np.array(cyc).T
    Neq = neq_hz * len(x) / fs
    return float((np.sum(n * r**m) / Neq) ** (1 / m))


def lags(X, L, step=1):
    """causal lag stack: X (N,d) -> (N, d*L) with rows t using X[t - j*step]."""
    N, d = X.shape
    out = np.zeros((N, d * L))
    for j in range(L):
        s = j * step
        out[s:, j * d:(j + 1) * d] = X[:N - s]
        out[:s, j * d:(j + 1) * d] = X[0]
    return out


def fairlead_feats(S, delta, i):
    q = sensors.pose(S)
    F, Tqs = physics.line_loads(q, delta[i], i)
    pos, arm = physics.fairlead_kin(q)
    d = pos[i, :, :2] - physics.ANCH[i, :2]
    u = d / np.linalg.norm(d, axis=1)[:, None]
    # fairlead velocity / acceleration along the line (finite differences of pose
    # smoothed with MRU accelerations when available)
    v = np.gradient(pos[i], 1 / FS, axis=0)
    vl = np.sum(v[:, :2] * u, 1)
    vz = v[:, 2]
    a = np.c_[S.get('ax', np.zeros(len(q))), S.get('ay', np.zeros(len(q))), S.get('az', np.zeros(len(q)))]
    al = np.sum(a[:, :2] * u, 1)
    Tn = Tqs / 1e6
    X = np.c_[Tn, np.gradient(Tn, 1 / FS), vl, vz, al, a[:, 2]]
    return Tqs, X


def raw_feats(S):
    keys = [k for k in sorted(S) if k not in ('t',)]
    return np.c_[[S[k] for k in keys]].T


def trim(x, n):
    return np.asarray(x)[:n]


class TensionVS:
    def __init__(self, kind='PG', L=24, step=2):
        self.kind, self.L, self.step = kind, L, step

    def _X(self, S, delta, i):
        if self.kind == 'PG':
            Tqs, X = fairlead_feats(S, delta, i)
            return Tqs, lags(X, self.L, self.step)
        Xr = raw_feats(S)
        Xr = (Xr - self.mu) / self.sd
        X = lags(Xr, self.L, self.step)
        if self.kind == 'DDnod':
            return None, X
        dl = np.tile(np.asarray(delta, float) * 10, (len(X), 1))     # twin's degradation estimate as extra input
        X = np.c_[X, dl, X[:, :Xr.shape[1]] * dl[:, [i]]]
        if self.kind == 'DD-MLP':
            return None, X
        return None, X

    def fit(self, wins, deltas):
        if self.kind != 'PG':
            R = np.vstack([raw_feats(w['S']) for w in wins]); self.mu, self.sd = R.mean(0), R.std(0) + 1e-9
        self.models = []
        for i in range(3):
            Xs, ys = [], []
            for w, dl in zip(wins, deltas):
                Tqs, X = self._X(w['S'], dl, i)
                y = w['truth'][f'T{i+1}'][::5][:len(X)]
                X = X[:len(y)]
                ys.append((y - (Tqs[:len(y)] if Tqs is not None else 0)) / 1e6); Xs.append(X)
            X, y = np.vstack(Xs), np.concatenate(ys)
            if self.kind == 'DD-MLP':
                from sklearn.neural_network import MLPRegressor
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                m = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=80,
                                                                 early_stopping=True, random_state=0))
                self.models.append(m.fit(X[::2], y[::2]))
            else:
                self.models.append(RidgeCV(alphas=np.logspace(-4, 3, 15)).fit(X, y))
        return self

    def predict(self, S, delta, i):
        Tqs, X = self._X(S, delta, i)
        r = self.models[i].predict(X) * 1e6
        return r + (Tqs if Tqs is not None else 0)


def qs_tension(S, delta, i):
    return physics.line_loads(sensors.pose(S), delta[i], i)[1]


# ---------------------------------------------------------------- tower base
def tower_phys_X(S, thrust):
    th = np.radians(S.get('pitch', np.zeros(len(S['t']))))
    anac = S.get('nac_ax', np.zeros(len(th)))
    ax = S.get('ax', np.zeros(len(th)))
    qd = np.radians(S.get('q', np.zeros(len(th))))
    qdd = np.gradient(qd, 1 / FS)
    return np.c_[thrust / 1e6, np.sin(th), anac, ax, qdd, np.ones(len(th))]


class TowerVS:
    def __init__(self, kind='PG', L=24, step=2):
        self.kind, self.L, self.step = kind, L, step

    def _X(self, S, thrust):
        if self.kind == 'PG':
            P = tower_phys_X(S, thrust)
            return np.c_[P, lags(P[:, [0, 2]], self.L, self.step)]
        Xr = (raw_feats(S) - self.mu) / self.sd
        return lags(Xr, self.L, self.step)

    def fit(self, wins, thrusts):
        if self.kind != 'PG':
            R = np.vstack([raw_feats(w['S']) for w in wins]); self.mu, self.sd = R.mean(0), R.std(0) + 1e-9
        Xs, ys = [], []
        for w, T in zip(wins, thrusts):
            X = self._X(w['S'], T); y = w['truth']['twr_my'][::5][:len(X)] / 1e4
            Xs.append(X[:len(y)]); ys.append(y)
        self.m = RidgeCV(alphas=np.logspace(-4, 3, 15)).fit(np.vstack(Xs), np.concatenate(ys))
        return self

    def predict(self, S, thrust):
        return self.m.predict(self._X(S, thrust)) * 1e4
