"""Experiment 2: virtual sensing of fairlead tensions + tower-base fore-aft
moment, and fatigue (DEL) estimation from sparse sensors.
Test-time degradation used by the physics models is the twin's own PG-FE
estimate (no access to truth)."""
import sys, json, numpy as np, warnings
import exp_fault as E, dt, vsense as V, sensors
warnings.filterwarnings('ignore')
M = dict(T=3.0, twr=4.0)   # Wohler exponents: chain, welded steel


def nrmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.std(b))


def r2(a, b):
    return float(1 - np.mean((a - b) ** 2) / np.var(b))


def run(tr, tests, delta_hat):
    """delta_hat: dict (name,k)->estimated delta for test windows."""
    thr = dt.ThrustModel().fit(tr)
    dtr = [w['delta'] for w in tr]
    models = {'PG-VS': V.TensionVS('PG').fit(tr, dtr), 'DD-VS': V.TensionVS('DD').fit(tr, dtr), 'DD-MLP': V.TensionVS('DD-MLP').fit(tr, dtr),
              'DD-VS(no δ)': V.TensionVS('DDnod').fit(tr, dtr)}
    tmods = {'PG-VS': V.TowerVS('PG').fit(tr, [thr.predict(w['S']) for w in tr]),
             'DD-VS': V.TowerVS('DD').fit(tr, [None] * len(tr))}
    out = {}
    for split, te in tests.items():
        rows = []
        for w in te:
            S = w['S']; dh = delta_hat[(w['name'], w['k'])]
            rec = dict(name=w['name'], delta=w['delta'].tolist(), delta_hat=list(map(float, dh)))
            for i in range(3):
                y = w['truth'][f'T{i+1}'][::5][:len(S['t'])].astype(float)
                n = len(y)
                preds = {'QS(true δ)': V.qs_tension(S, w['delta'], i)[:n], 'QS(est δ)': V.qs_tension(S, dh, i)[:n],
                         'QS(healthy)': V.qs_tension(S, np.zeros(3), i)[:n],
                         'PG-VS': models['PG-VS'].predict(S, dh, i)[:n], 'DD-VS': models['DD-VS'].predict(S, dh, i)[:n],
                         'DD-MLP': models['DD-MLP'].predict(S, dh, i)[:n], 'DD-VS(no δ)': models['DD-VS(no δ)'].predict(S, dh, i)[:n]}
                ref_del = V.DEL(y, M['T'], V.FS)
                for k, p in preds.items():
                    rec[f'T{i+1}|{k}'] = dict(nrmse=nrmse(p, y), r2=r2(p, y), max_err=float(abs(p.max() - y.max()) / y.max()),
                                             del_err=float(V.DEL(p, M['T'], V.FS) / ref_del - 1))
            y = w['truth']['twr_my'][::5][:len(S['t'])].astype(float); n = len(y)
            ref_del = V.DEL(y, M['twr'], V.FS)
            T = thr.predict(S)
            preds = {'PG-VS': tmods['PG-VS'].predict(S, T)[:n], 'DD-VS': tmods['DD-VS'].predict(S, None)[:n]}
            for k, p in preds.items():
                rec[f'twr|{k}'] = dict(nrmse=nrmse(p, y), r2=r2(p, y), del_err=float(V.DEL(p, M['twr'], V.FS) / ref_del - 1))
            rows.append(rec)
        out[split] = rows
    return out


def table(out):
    for split, rows in out.items():
        keys = [k for k in rows[0] if '|' in k]
        groups = {}
        for k in keys:
            ch, mdl = k.split('|'); ch = 'T' if ch.startswith('T') else ch
            groups.setdefault((ch, mdl), []).extend([r[k] for r in rows])
        print(split)
        for (ch, mdl), v in groups.items():
            print(f"  {ch:4s} {mdl:12s} NRMSE={np.mean([x['nrmse'] for x in v]):.3f} R2={np.mean([x['r2'] for x in v]):.3f} "
                  f"|DELerr| mean={np.mean([abs(x['del_err']) for x in v])*100:5.1f}% p95={np.percentile([abs(x['del_err']) for x in v],95)*100:5.1f}%")


if __name__ == '__main__':
    import extract
    S_ = E.split_names()
    if sys.argv[1:] == ['dev']:
        names = [n for n in extract.available() if n.startswith('train')]
        S_ = dict(train=names[:50], test=names[50:])
    tr = E.get_windows(S_['train'], 'S3_full_sparse', 400.0, 1.0)
    tests = {k: E.get_windows(S_[k], 'S3_full_sparse', 400.0, 1.0) for k in S_ if k != 'train' and S_[k]}
    r, m = E.run_pgfe(tr, tests, comps=dt.COMP_ALL, env_kind='lin', prior='spike')
    dh = {}
    for k, te in tests.items():
        for w, p in zip(te, r[k]['pred']):
            dh[(w['name'], w['k'])] = np.array(p)
    out = run(tr, tests, dh)
    table(out)
    json.dump(out, open(f"{E.RES}/vs_{'dev' if sys.argv[1:] == ['dev'] else 'main'}.json", 'w'))
