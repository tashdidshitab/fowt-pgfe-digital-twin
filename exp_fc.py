"""Experiment 3: short-term motion forecasting with wave preview."""
import sys, json, numpy as np, warnings
import exp_fault as E, forecast as F, extract
warnings.filterwarnings('ignore')
TARGETS = ['pitch', 'surge', 'heave']
HORIZONS = [2.0, 5.0, 10.0, 15.0, 20.0]
MODES = [('AR', False), ('DD', False), ('DD', True), ('PG', False)]

def run(tr, tests):
    out = {}
    for tg in TARGETS:
        for h in HORIZONS:
            for mode, mlp in MODES:
                lab = f"{mode}{'-MLP' if mlp else ''}"
                m = F.Forecaster(tg, h, mode, mlp).fit(tr)
                for split, te in tests.items():
                    P, T, Y0 = [], [], []
                    for w in te:
                        p, t, y0 = m.predict(w); P.append(p); T.append(t); Y0.append(y0)
                    s = F.score(np.concatenate(P), np.concatenate(T), np.concatenate(Y0))
                    out[f'{split}|{tg}|{h}|{lab}'] = s
                    print(f'{split:5s} {tg:6s} h={h:4.0f}s {lab:7s} NRMSE={s["NRMSE"]:.3f} skill={s["skill_vs_persistence"]:.3f}', flush=True)
    return out

if __name__ == '__main__':
    S_ = E.split_names()
    if sys.argv[1:] == ['dev']:
        names = [n for n in extract.available() if n.startswith('train')]
        S_ = dict(train=names[:50], test=names[50:])
        TARGETS[:] = ['pitch']; HORIZONS[:] = [5.0, 10.0, 15.0]
    tr = E.get_windows(S_['train'], 'S3_full_sparse', 400.0, 1.0)
    tests = {k: E.get_windows(S_[k], 'S3_full_sparse', 400.0, 1.0) for k in S_ if k != 'train' and S_[k]}
    out = run(tr, tests)
    json.dump(out, open(f"{E.RES}/fc_{'dev' if sys.argv[1:] == ['dev'] else 'main'}.json", 'w'))
