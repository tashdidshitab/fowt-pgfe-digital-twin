"""Experiment 1: mooring degradation estimation from sparse sensors.
Compares the physics-guided estimator (PG-FE) with data-driven baselines
across sensor sets, window lengths, noise levels and training-set sizes."""
from paths import ROOT, RTEST
import os, sys, json, pickle, time, numpy as np, warnings
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import dt, sensors, extract

warnings.filterwarnings('ignore')
CACHE = ROOT + '/cache'; os.makedirs(CACHE, exist_ok=True)
RES = ROOT + '/results'; os.makedirs(RES, exist_ok=True)
THR = 0.05   # damage-detection threshold on area loss


def split_names():
    av = set(extract.available())
    design = json.load(open(ROOT + '/design.json'))
    return {s: [c['name'] for c in design if c['split'] == s and c['name'] in av] for s in ['train', 'test', 'ood']}


def get_windows(names, sset, W, noise, seed_off=0, offset=0.0):
    key = f"{sset}_{int(W)}_{noise}_{seed_off}" + (f"_o{offset}" if offset else "")
    out = []
    for n in names:
        f = os.path.join(CACHE, f'{n}__{key}.pkl')
        if os.path.exists(f):
            out += pickle.load(open(f, 'rb')); continue
        ws = dt.windows(n, W, sensors.SETS[sset], noise, seed_off, offset)
        for w in ws:
            w['Ftab'] = dt.mooring_table(w)
            w['Ftrue'] = dt.mooring_true(w)
            w['ddf'] = dt.dd_features(w)
        pickle.dump(ws, open(f, 'wb'))
        out += ws
    return out


def cache_of(ws):
    return {(w['name'], w['k']): w for w in ws}


def metrics(dhat, dtrue, p_dam=None):
    dhat, dtrue = np.asarray(dhat), np.asarray(dtrue)
    err = np.abs(dhat - dtrue)
    dam_t = dtrue > THR
    dam_h = (p_dam > 0.5) if p_dam is not None else dhat > THR
    # run-level: any damaged line
    run_t, run_h = dam_t.any(1), dam_h.any(1)
    tp = np.sum(run_t & run_h); fp = np.sum(~run_t & run_h); fn = np.sum(run_t & ~run_h)
    # localization for single-line damage
    single = dam_t.sum(1) == 1
    loc = np.mean(np.argmax(dhat[single], 1) == np.argmax(dtrue[single], 1)) if single.any() else np.nan
    # severity error on damaged lines only
    sev = np.mean(err[dam_t]) if dam_t.any() else np.nan
    return dict(MAE=float(err.mean()), RMSE=float(np.sqrt((err**2).mean())), MAE_damaged_lines=float(sev),
                det_F1=float(2 * tp / max(2 * tp + fp + fn, 1)), false_alarm=float(fp / max(np.sum(~run_t), 1)),
                miss=float(fn / max(np.sum(run_t), 1)), loc_acc=float(loc), n=int(len(dtrue)))


def run_pgfe(tr, tests, **kw):
    t = time.time()
    m = dt.PGFE(**kw).fit(tr, cache_of(tr))
    out = {}
    for nm, te in tests.items():
        P = [m.predict(w, cache_of(te)) for w in te]
        dtrue = np.array([w['delta'] for w in te])
        dh = np.array([p['mean'] for p in P])
        ci = np.array([p['ci'] for p in P])
        cover = np.mean((dtrue >= ci[:, :, 0] - 1e-9) & (dtrue <= ci[:, :, 1] + 0.0125))
        r = metrics(dh, dtrue, np.array([p['p_damaged'] for p in P]))
        r['CI90_coverage'] = float(cover); r['CI90_width'] = float(np.mean(ci[:, :, 1] - ci[:, :, 0]))
        out[nm] = dict(metrics=r, pred=dh.tolist(), true=dtrue.tolist(), ci=ci.tolist(),
                       names=[w['name'] for w in te], p_dam=[p['p_damaged'].tolist() for p in P])
    out['fit_time_s'] = time.time() - t
    out['sigma'] = m.sig.tolist()
    return out, m


def dd_models():
    return {
        'DD-GBM': lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=5),
        'DD-RF': lambda: RandomForestRegressor(400, min_samples_leaf=2, random_state=0, n_jobs=1),
        'DD-MLP': lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-2, max_iter=3000,
                                                                       early_stopping=False, random_state=0)),
    }


def run_dd(kind, tr, tests):
    X = np.array([w['ddf'] for w in tr]); Y = np.array([w['delta'] for w in tr])
    sc = StandardScaler().fit(X)
    mk = dd_models()[kind]
    if kind == 'DD-MLP':
        models = [mk().fit(X, Y)]
        pred = lambda Z: models[0].predict(Z)
    else:
        models = [mk().fit(X, Y[:, i]) for i in range(3)]
        pred = lambda Z: np.c_[[m.predict(Z) for m in models]].T
    out = {}
    for nm, te in tests.items():
        Z = np.array([w['ddf'] for w in te]); dtrue = np.array([w['delta'] for w in te])
        dh = np.clip(pred(Z), 0, 0.6)
        out[nm] = dict(metrics=metrics(dh, dtrue), pred=dh.tolist(), true=dtrue.tolist(), names=[w['name'] for w in te])
    return out


def summarize(res, label):
    for nm in ['test', 'ood']:
        if nm in res:
            m = res[nm]['metrics']
            print(f"  {label:28s} {nm:4s} MAE={m['MAE']*100:5.2f}%  MAEdam={m['MAE_damaged_lines']*100:5.2f}%  "
                  f"F1={m['det_F1']:.2f} FA={m['false_alarm']:.2f} loc={m['loc_acc']:.2f}"
                  + (f"  cov={m.get('CI90_coverage', np.nan):.2f}" if 'CI90_coverage' in m else ''), flush=True)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'main'
    S = split_names()
    print({k: len(v) for k, v in S.items()}, flush=True)
    allres = {}
    PROP = dict(comps=dt.COMP_ALL, env_kind='lin', prior='spike')
    ABL = {'PG-FE': PROP,
           'abl: horizontal DOFs only': dict(comps=dt.COMP_H, env_kind='lin', prior='spike'),
           'abl: uniform prior': dict(comps=dt.COMP_ALL, env_kind='lin', prior='uniform'),
           'abl: GP env model': dict(comps=dt.COMP_ALL, env_kind='gp', prior='spike')}
    if mode == 'main':
        for sset in sensors.SETS:
            for seed in [0, 1, 2]:
                tr = get_windows(S['train'], sset, 400.0, 1.0, seed)
                tests = {k: get_windows(S[k], sset, 400.0, 1.0, seed) for k in ['test', 'ood']}
                key = f'{sset}|s{seed}'
                print(key, flush=True)
                for lab, kw in ABL.items():
                    if lab != 'PG-FE' and sset != 'S3_full_sparse':
                        continue
                    r, _ = run_pgfe(tr, tests, **kw); summarize(r, lab); allres[key + '|' + lab] = r
                for kind in dd_models():
                    r = run_dd(kind, tr, tests); summarize(r, kind); allres[key + '|' + kind] = r
    elif mode == 'noise':
        for ns, off in [(0.5, 0), (1, 0), (2, 0), (4, 0), (1, 1), (1, 2), (2, 2)]:
            tr = get_windows(S['train'], 'S3_full_sparse', 400.0, ns, 0, off)
            tests = {k: get_windows(S[k], 'S3_full_sparse', 400.0, ns, 0, off) for k in ['test', 'ood']}
            key = f'n{ns}|o{off}'; print(key, flush=True)
            r, _ = run_pgfe(tr, tests, **PROP); summarize(r, 'PG-FE'); allres[key + '|PG-FE'] = r
            for kind in ['DD-GBM', 'DD-RF']:
                r = run_dd(kind, tr, tests); summarize(r, kind); allres[key + '|' + kind] = r
    elif mode == 'win':
        for W in [100.0, 200.0, 400.0]:
            tr = get_windows(S['train'], 'S3_full_sparse', W, 1.0)
            tests = {k: get_windows(S[k], 'S3_full_sparse', W, 1.0) for k in ['test', 'ood']}
            key = f'W{int(W)}'; print(key, flush=True)
            r, _ = run_pgfe(tr, tests, **PROP); summarize(r, 'PG-FE'); allres[key + '|PG-FE'] = r
            for kind in ['DD-GBM', 'DD-RF']:
                r = run_dd(kind, tr, tests); summarize(r, kind); allres[key + '|' + kind] = r
    elif mode == 'lc':
        trall = get_windows(S['train'], 'S3_full_sparse', 400.0, 1.0)
        tests = {k: get_windows(S[k], 'S3_full_sparse', 400.0, 1.0) for k in ['test', 'ood']}
        rng = np.random.default_rng(1)
        for n in [10, 20, 40, 70, len(trall)]:
            for rep in range(3 if n < len(trall) else 1):
                tr = [trall[j] for j in rng.choice(len(trall), n, replace=False)]
                key = f'N{n}|r{rep}'; print(key, flush=True)
                r, _ = run_pgfe(tr, tests, **PROP); summarize(r, 'PG-FE'); allres[key + '|PG-FE'] = r
                for kind in ['DD-GBM', 'DD-RF']:
                    r = run_dd(kind, tr, tests); summarize(r, kind); allres[key + '|' + kind] = r
    elif mode == 'model':
        import physics, copy
        base_tr = get_windows(S['train'], 'S3_full_sparse', 400.0, 1.0)
        base_te = {k: get_windows(S[k], 'S3_full_sparse', 400.0, 1.0) for k in ['test', 'ood']}
        healthy = [w for w in base_tr if np.all(w['delta'] == 0)]
        cal = healthy[:6]                                   # commissioning period: 6 x 400 s healthy windows
        tr = [w for w in base_tr if w not in cal]           # env model never sees the commissioning data
        print('healthy train windows', len(healthy), 'cal', len(cal), 'train', len(tr), flush=True)
        pg = dt.PGFE(**PROP).fit(tr, cache_of(tr))
        rng = np.random.default_rng(7)
        def redo(ws):
            return [dict(w, Ftab=dt.mooring_table(w), Ftrue=dt.mooring_true(w)) for w in ws]
        def evaluate(tests):
            out = {}
            for nm, te in tests.items():
                P = [pg.predict(w, cache_of(te)) for w in te]
                dtrue = np.array([w['delta'] for w in te]); dh = np.array([p['mean'] for p in P])
                ci = np.array([p['ci'] for p in P])
                r = metrics(dh, dtrue, np.array([p['p_damaged'] for p in P]))
                r['CI90_coverage'] = float(np.mean((dtrue >= ci[:, :, 0] - 1e-9) & (dtrue <= ci[:, :, 1] + 0.0125)))
                out[nm] = dict(metrics=r, pred=dh.tolist(), true=dtrue.tolist())
            return out
        for sig in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            for rep in range(5 if sig > 0 else 1):
                pert = rng.normal(0, sig, 3)
                physics.ANCH_PERT = pert.copy()
                tests = {k: redo(v) for k, v in base_te.items()}
                key = f'sig{sig}|r{rep}'; print(key, np.round(pert, 2), flush=True)
                r = evaluate(tests); r['pert'] = pert.tolist(); summarize(r, 'PG-FE (no calibration)'); allres[key + '|PG-FE'] = r
                c = dt.calibrate_anchors(pg, cal, pert)
                physics.ANCH_PERT = pert + c
                tests = {k: redo(v) for k, v in base_te.items()}
                r = evaluate(tests); r['pert'] = pert.tolist(); r['corr'] = c.tolist()
                print('   residual geometry error after calibration (m):', np.round(pert + c, 2), flush=True)
                summarize(r, 'PG-FE + commissioning cal.'); allres[key + '|PG-FE-cal'] = r
        physics.ANCH_PERT = np.zeros(3)
    json.dump(allres, open(f'{RES}/fault_{mode}.json', 'w'))
