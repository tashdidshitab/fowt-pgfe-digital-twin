"""Revised experiments (response to review). Results -> results/rev_*.json"""
from paths import ROOT, RTEST
import os, sys, json, numpy as np, warnings, copy
import exp_fault as E, rev_core as RC, dt, sensors, physics, extract
from catenary import solve_line, line_props, L0, ANCH, FAIR
warnings.filterwarnings('ignore')
RES = ROOT + '/results'
D0 = json.load(open(ROOT + '/design.json')); DR = json.load(open(ROOT + '/design_rev.json'))
META = {c['name']: c for c in D0 + DR}
AV = set(extract.available())


def names(pred):
    return [c['name'] for c in D0 + DR if pred(c) and c['name'] in AV]


POOLS = {
    'ID': names(lambda c: c['split'] == 'test' or c.get('group') in ('A1-intact', 'A3-reseed')),
    'OOD-severity': names(lambda c: (c['split'] == 'ood' and c['name'] < 'ood_008') or c.get('group') == 'A5-grid-edge'),
    'OOD-two-line': names(lambda c: (c['split'] == 'ood' and 'ood_008' <= c['name'] < 'ood_016') or c.get('group') == 'A2-two-line'),
    'OOD-extreme-sea': names(lambda c: c['split'] == 'ood' and c['name'] >= 'ood_016'),
    'OOD-misalign': names(lambda c: c.get('group') == 'A4-misalign'),
}
TRAIN_MIXED = names(lambda c: c['split'] == 'train')
TRAIN_HEALTHY = names(lambda c: (c['split'] == 'train' and max(c['delta']) == 0) or c['split'] == 'htrain')


def W(nms, sset='S3_full_sparse', Wl=400.0, noise=1.0, seed=0, off=0.0):
    return E.get_windows(nms, sset, Wl, noise, seed, off)


def run_pg(pg, ws, mu=None):
    P = [pg.predict(w) if mu is None else pg.predict(w, mu=mu[i]) for i, w in enumerate(ws)]
    T = np.array([w['delta'] for w in ws])
    return P, T


def evaluate(P, T):
    M = np.array([p['mean'] for p in P]); A = np.array([p['p_exc'] > 0.5 for p in P])
    cis = {a: np.array([p['ci'][a] for p in P]) for a in P[0]['ci']}
    out = dict(sev=RC.severity_metrics(T, M), det=RC.detection_metrics(T, A), cov=RC.coverage(T, cis),
               pred=M.tolist(), true=T.tolist(), p_exc=[p['p_exc'].tolist() for p in P], ci90=cis[0.9].tolist())
    return out


def evaluate_pt(M, T):
    M = np.asarray(M); T = np.asarray(T)
    return dict(sev=RC.severity_metrics(T, M), det=RC.detection_metrics(T, M > RC.DSTAR), pred=M.tolist(), true=T.tolist())


def short(r):
    s, d = r['sev'], r['det']
    c = r.get('cov', {}).get(0.9, np.nan)
    return (f"MAEdam={100*s['MAE_dam']:.2f} MAEall={100*s['MAE_all']:.2f} FA={d['intact_fa_k']}/{d['intact_n']} "
            f"rec={d['run_recall']:.2f} P/R={d['line_precision']:.2f}/{d['line_recall']:.2f} exact={d['exact_set_acc']:.2f} cov90={c:.2f}")


def dump(obj, fn):
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return conv(o.tolist())
        return o
    json.dump(conv(obj), open(f'{RES}/{fn}', 'w'))


# ===================================================================== E1 main comparison
def e1_main():
    fn = f'{RES}/rev_e1_main.json'
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    for sset in sensors.SETS:
        if any(k.startswith(sset + '|DD-MLP|') for k in out):
            continue
        trh = W(TRAIN_HEALTHY, sset); trm = W(TRAIN_MIXED, sset)
        pools = {k: W(v, sset) for k, v in POOLS.items() if v}
        est = {'PG-FE (healthy-trained)': RC.PGFE3(cov='diag', hetero=False, p0=0.9).fit(trh)}
        if sset == 'S3_full_sparse':
            est.update({
                'PG-FE (label-trained)': RC.PGFE3(cov='diag', hetero=False, p0=0.9).fit(trm),
                'PG-FE + weight nuisance': RC.PGFE3(cov='diag', hetero=False, p0=0.9, nuis='mass').fit(trh),
                'v1 PG-FE, pose-learned, label-trained': RC.PGFE2(cov='diag', hetero=False, p0=0.5).fit(trm),
                'v1 PG-FE, pose-learned, healthy-trained': RC.PGFE2(cov='diag', hetero=False, p0=0.5).fit(trh)})
        for lab, pg in est.items():
            for pn, ws in pools.items():
                r = evaluate(*run_pg(pg, ws)); out[f'{sset}|{lab}|{pn}'] = r
                print(f'{sset[:2]} {lab:40s} {pn:16s}', short(r), flush=True)
        # oracle environmental load (simulation truth) -> isolates env-model error
        if sset == 'S3_full_sparse':
            pg = est['PG-FE (healthy-trained)']
            for pn, ws in pools.items():
                mu = [-w['Ftrue'][pg.comps] for w in ws]
                r = evaluate(*run_pg(pg, ws, mu)); out[f'{sset}|PG-FE oracle env load|{pn}'] = r
                print(f'{sset[:2]} {"PG-FE oracle env load":40s} {pn:16s}', short(r), flush=True)
            # env-load prediction error per component, in units of the CV residual scale
            err = {}
            for pn, ws in pools.items():
                e = np.array([(pg.env_pred(w)[0] + w['Ftrue'][pg.comps]) / dt.SCALE[pg.comps] for w in ws])
                err[pn] = (np.abs(e).mean(0) / np.sqrt(np.diag(pg.C))).tolist()
            out['env_err_over_sigma'] = err
            # physics baselines (same core + env predictor)
            pb = RC.PhysicsBaselines(pg)
            hw = trh
            lam_grid = [0, 1, 3, 10, 30]
            # L1 penalty chosen on healthy training windows only: smallest lambda giving <=5% lines > delta*
            lam = 0
            for L in lam_grid:
                frac = np.mean([pb.wls(w, L) > RC.DSTAR for w in hw])
                if frac <= 0.05:
                    lam = L; break
            out['L1_lambda'] = lam
            for pn, ws in pools.items():
                for lab, f in [('WLS (physics)', lambda w: pb.wls(w, 0.0)), ('L1-WLS (physics)', lambda w: pb.wls(w, lam))]:
                    r = evaluate_pt([f(w) for w in ws], [w['delta'] for w in ws]); out[f'{sset}|{lab}|{pn}'] = r
                    print(f'{sset[:2]} {lab:40s} {pn:16s}', short(r), flush=True)
        # data-driven baselines (supervised; need damage labels)
        for kind in ['DD-GBM', 'DD-RF', 'DD-MLP']:
            f, hp = RC.tuned_dd(kind, trm); out[f'{sset}|{kind}|hyperparameters'] = {k: str(v) for k, v in hp.items()}
            for pn, ws in pools.items():
                rr = evaluate_pt(f(np.array([w['ddf'] for w in ws])), [w['delta'] for w in ws]); out[f'{sset}|{kind}|{pn}'] = rr
                print(f'{sset[:2]} {kind:40s} {pn:16s}', short(rr), flush=True)
        dump(out, 'rev_e1_main.json')


# ======================================================= E1b sub-window EKF and sequential PG-FE
def e1b_sequential():
    out = {}
    trh = W(TRAIN_HEALTHY, Wl=400.0)
    tr100 = W(TRAIN_HEALTHY, Wl=100.0)
    pg100 = RC.PGFE3(cov='diag', hetero=False, p0=0.9).fit(tr100)
    pb = RC.PhysicsBaselines(pg100)
    for pn, nms in POOLS.items():
        if not nms:
            continue
        ws = W(nms, Wl=100.0)
        byrun = {}
        for w in ws:
            byrun.setdefault(w['name'], []).append(w)
        T, Mekf, Mseq, Pseq, ci_seq = [], [], [], [], []
        for nm, sw in byrun.items():
            sw = sorted(sw, key=lambda w: w['k'])
            x, sd, _ = pb.ekf(sw); Mekf.append(x)
            ll = sum(pg100.loglik(w) for w in sw)          # sequential Bayesian fusion of 4 x 100 s
            p = pg100.post_from_ll(ll); Mseq.append(p['mean']); Pseq.append(p['p_exc'] > 0.5); ci_seq.append(p['ci'][0.9])
            T.append(sw[0]['delta'])
        T = np.array(T)
        out[f'EKF (physics, 4x100 s)|{pn}'] = evaluate_pt(Mekf, T)
        r = dict(sev=RC.severity_metrics(T, np.array(Mseq)), det=RC.detection_metrics(T, np.array(Pseq)),
                 cov={0.9: RC.coverage(T, {0.9: np.array(ci_seq)}, (0.9,))[0.9]})
        out[f'PG-FE sequential (4x100 s)|{pn}'] = r
        print(f'{"EKF":28s} {pn:16s}', short(out[f"EKF (physics, 4x100 s)|{pn}"]), flush=True)
        print(f'{"PG-FE seq 4x100":28s} {pn:16s}', short(r), flush=True)
    dump(out, 'rev_e1b_seq.json')


# ===================================================== E2 identifiability, prior, covariance, reliability
def e2_uq():
    out = {}
    trh = W(TRAIN_HEALTHY)
    pools = {k: W(v) for k, v in POOLS.items() if v}
    pg = RC.PGFE3(cov='diag', hetero=False, p0=0.9).fit(trh)
    # identifiability: weighted Fisher information (per window) at delta = 0 and at the true delta
    ident = []
    for pn, ws in pools.items():
        for w in ws:
            _, sc = pg.env_pred(w)
            for d0, tag in [((0, 0, 0), 'intact'), (tuple(np.clip(w['delta'], 0, 0.575)), 'true')]:
                J = RC.jacobian(w['Ftab'], d0)[pg.comps] / (dt.SCALE[pg.comps] * sc)[:, None]
                Jh = J[[0, 1, 5]]
                I6 = J.T @ pg.Ci @ J
                Ch = np.diag(1 / np.diag(pg.C)[[0, 1, 5]])
                Ih = Jh.T @ Ch @ Jh
                crb = np.sqrt(np.diag(np.linalg.inv(I6)))
                ident.append(dict(pool=pn, name=w['name'], at=tag, U=META[w['name']]['U'], Hs=META[w['name']]['Hs'],
                                  cond6=float(np.linalg.cond(I6)), condH=float(np.linalg.cond(Ih)), crb=crb.tolist(),
                                  colcorr=np.corrcoef(J.T).tolist()))
    out['ident'] = ident
    # prior and covariance sensitivity
    for cov, het in [('diag', False), ('full', False), ('full', True), ('diag', True)]:
        for p0 in [0.5, 0.75, 0.9, 0.97, 0.99]:
            pgx = RC.PGFE3(cov=cov, hetero=het, p0=p0).fit(trh)
            for pn, ws in pools.items():
                r = evaluate(*run_pg(pgx, ws)); key = f'{cov}|het={het}|p0={p0}|{pn}'
                out[key] = {k: r[k] for k in ('sev', 'det', 'cov')}
                if pn in ('ID', 'OOD-two-line'):
                    print(f'{key:40s}', short(r), flush=True)
    # severity-stratified coverage + reliability (primary configuration)
    strat = {}
    for pn, ws in pools.items():
        P, T = run_pg(pg, ws)
        cis = {a: np.array([p['ci'][a] for p in P]) for a in P[0]['ci']}
        bins = [(0, 0), (1e-9, 0.05), (0.05, 0.15), (0.15, 0.35), (0.35, 1.0)]
        for lo, hi in bins:
            m = (T >= lo) & (T <= hi) if lo == hi else (T > lo) & (T <= hi)
            if m.any():
                strat[f'{pn}|{lo:.2f}-{hi:.2f}'] = dict(n=int(m.sum()), cov={a: float(((T >= cis[a][:, :, 0]) & (T <= cis[a][:, :, 1]))[m].mean()) for a in cis})
    out['stratified'] = strat
    dump(out, 'rev_e2_uq.json')




# ======================================================= helpers: true six-component mooring load
def true_moor_load(name, t0=120.0, dur=400.0):
    """window-mean six-component mooring load on the platform from MoorDyn fairlead force outputs
    (independent of the twin's quasi-static model). Returns None if outputs unavailable."""
    D = extract.load(name)
    if 'P4FX' not in D:
        return None
    m = (D['t'] >= t0) & (D['t'] < t0 + dur)
    meta = META[name]; fair = np.array(json.load(open(f'{ROOT}/runs/{name}/meta.json')).get('fair', FAIR.tolist()))
    q = np.c_[D['surge'], D['sway'], D['heave'], np.radians(D['roll']), np.radians(D['pitch']), np.radians(D['yaw'])][m]
    r, p, y = q[:, 3], q[:, 4], q[:, 5]
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = cy * cp; R[:, 0, 1] = cy * sp * sr - sy * cr; R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp; R[:, 1, 1] = sy * sp * sr + cy * cr; R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp;     R[:, 2, 1] = cp * sr;                R[:, 2, 2] = cp * cr
    tot = np.zeros(6)
    for i, j in enumerate((4, 5, 6)):
        f = np.c_[D[f'P{j}FX'][m], D[f'P{j}FY'][m], D[f'P{j}FZ'][m]].astype(float)
        arm = np.einsum('nij,j->ni', R, fair[i])
        tot += np.r_[f.mean(0), np.cross(arm, f).mean(0)]
    return tot


# ===================================================== E3 physical mismatch / nuisance changes
def e3_mismatch():
    out = {}
    trh = W(TRAIN_HEALTHY)
    pg = RC.PGFE3(cov='diag', hetero=False, p0=0.9).fit(trh)
    pgm = RC.PGFE3(cov='diag', hetero=False, p0=0.9, nuis='mass').fit(trh)
    sig = np.sqrt(np.diag(pg.C)) * dt.SCALE[pg.comps]
    # (a) six-component validation of the quasi-static core against MoorDyn, nominal reality
    nom = names(lambda c: c['split'] in ('htrain', 'rev'))
    val = []
    for nm in nom:
        F = true_moor_load(nm)
        if F is None:
            continue
        w = W([nm])[0]
        val.append(dict(name=nm, md=F.tolist(), qs=w['Ftrue'].tolist()))
    if val:
        md = np.array([v['md'] for v in val]); qs = np.array([v['qs'] for v in val])
        out['qs_vs_md_nominal'] = dict(n=len(val), mean_abs=np.abs(qs - md).mean(0).tolist(), max_abs=np.abs(qs - md).max(0).tolist(),
                                       over_sigma=(np.abs(qs - md).mean(0) / np.abs(sig)).tolist(), rel_to_range=(np.abs(qs - md).mean(0) / (md.max(0) - md.min(0))).tolist(),
                                       rows=val)
        print('QS vs MoorDyn (nominal), mean |err|/sigma per component:', np.round(out['qs_vs_md_nominal']['over_sigma'], 3), flush=True)
    groups = sorted({c['group'] for c in DR if c['split'] == 'mm'})
    for g in groups:
        nms = names(lambda c: c.get('group') == g)
        if not nms:
            continue
        ws = W(nms)
        rows = []
        for w in ws:
            F = true_moor_load(w['name'])
            p = pg.predict(w); pm = pgm.predict(w)
            mods = META[w['name']]['mods']
            rec = dict(name=w['name'], delta=w['delta'].tolist(), pred=p['mean'].tolist(), alarm=(p['p_exc'] > 0.5).tolist(),
                       pred_mass=pm['mean'].tolist(), alarm_mass=(pm['p_exc'] > 0.5).tolist(), ci90=p['ci'][0.9].tolist())
            if F is not None:
                rec['eq_err_over_sigma'] = ((w['Ftrue'] - F)[pg.comps] / sig).tolist()     # twin model at true delta vs MoorDyn
            if 'segs' in mods:
                i = mods['loc_line']; sg = mods['segs'][i]; L = sum(s[0] for s in sg)
                rec['loc'] = dict(line=i, delta_loc=mods['loc_delta'], where=mods['loc_where'],
                                  delta_eq_weight=sum(s[0] * s[1] for s in sg) / L,
                                  delta_eq_stiff=1 - L / sum(s[0] / (1 - s[1]) for s in sg))
            rows.append(rec)
        out[g] = rows
        T = np.array([r['delta'] for r in rows]); M = np.array([r['pred'] for r in rows]); A = np.array([r['alarm'] for r in rows])
        intact = (T == 0).all(1) & np.array(['loc' not in r for r in rows])
        msg = f'{g:18s} intact FA {int(A[intact].any(1).sum())}/{int(intact.sum())}'
        if (T > 0.05).any():
            msg += f'  MAEdam {100*np.abs(M - T)[T > 0.05].mean():.2f}'
        if any('loc' in r for r in rows):
            msg += '  loc: ' + ', '.join(f"{r['loc']['where'][:4]} d={r['loc']['delta_loc']:.1f} eqW={r['loc']['delta_eq_weight']:.3f} est={r['pred'][r['loc']['line']]:.3f}" for r in rows)
        if 'eq_err_over_sigma' in rows[0]:
            msg += '  eq.err/sigma=' + str(np.round(np.mean([np.abs(r['eq_err_over_sigma']) for r in rows if 'eq_err_over_sigma' in r], 0), 2))
        print(msg, flush=True)
    dump(out, 'rev_e3_mismatch.json')


# ====================================================== E4 installations with true geometry error
def effective_dr(dr, dl, i):
    """anchor-radius shift of a nominal-length line giving the same fairlead pretension as (dr, dL)."""
    from scipy.optimize import brentq
    EA, w = line_props(0.0)
    l0 = np.linalg.norm(FAIR[i, :2] - ANCH[i, :2]); h = FAIR[i, 2] - ANCH[i, 2]
    Ht, _, _ = solve_line(l0 + dr, h, L0 + dl, EA, w)
    return brentq(lambda x: solve_line(l0 + x, h, L0, EA, w)[0] - Ht, dr - dl - 5, dr - dl + 5)


def calibrate(pg, wins, with_mass=False):
    """estimate per-line effective anchor-radius corrections (and optionally a platform weight
    change) from intact commissioning windows; Laplace (Gauss-Newton) 1-sigma uncertainty."""
    from scipy.optimize import least_squares
    c = pg.comps
    pre = [(sensors.pose(w['S'])[::4], *pg.env_pred(w)) for w in wins]
    def res(x):
        physics.ANCH_PERT = np.array(x[:3])
        r = []
        Li = np.linalg.cholesky(pg.Ci)
        for q, mu, sc in pre:
            F = sum(physics.line_loads(q, 0.0, i)[0].mean(0) for i in range(3))
            if with_mass:
                F = F + np.r_[0, 0, x[3], 0, 0, 0]
            z = (F[c] + mu) / (dt.SCALE[c] * sc)
            r.append(Li.T @ z)
        physics.ANCH_PERT = np.zeros(3)
        return np.concatenate(r)
    x0 = np.zeros(4 if with_mass else 3)
    sol = least_squares(res, x0, diff_step=1e-3, x_scale=np.r_[np.ones(3), 1e5][:len(x0)])
    Jm = sol.jac
    cov = np.linalg.pinv(Jm.T @ Jm)
    return sol.x, np.sqrt(np.diag(cov))


def e4_install():
    out = {}
    trh = W(TRAIN_HEALTHY)
    pg = RC.PGFE3(cov='diag', hetero=False, p0=0.9).fit(trh)
    pgm = RC.PGFE3(cov='diag', hetero=False, p0=0.9, nuis='mass').fit(trh)
    for I in ['I1', 'I2', 'I3', 'I4', 'I5']:
        cases = [c for c in DR if c.get('group') == f'C-{I}' and c['name'] in AV]
        if not cases:
            continue
        mods = cases[0]['mods']
        com = [c['name'] for c in cases if c['mods']['role'] == 'commission']
        tst = [c['name'] for c in cases if c['mods']['role'] == 'test']
        cw = W(com, Wl=200.0); tw = W(tst)
        actual = [effective_dr(mods['anch_dr'][i], mods['dL'][i], i) for i in range(3)]
        rec = dict(anch_dr=mods['anch_dr'], dL=mods['dL'], ptfm_m_s=mods.get('ptfm_m_s', 1.0), actual_eff_dr=actual)
        for lab, model, wm in [('geometry', pg, False), ('geometry+mass', pgm, True)]:
            x, sd = calibrate(model, cw, with_mass=wm)
            rec[f'cal_{lab}'] = dict(x=x.tolist(), sd=sd.tolist())
            for cal in (False, True):
                physics.ANCH_PERT = np.array(x[:3]) if cal else np.zeros(3)
                ws = [dict(w, Ftab=dt.mooring_table(w)) for w in tw]
                physics.ANCH_PERT = np.zeros(3)
                if wm and cal:
                    P = [pg.predict(w, mu=pg.env_pred(w)[0] + np.r_[0, 0, x[3], 0, 0, 0][pg.comps]) for w in ws]
                else:
                    P = [model.predict(w) for w in ws]
                T = np.array([w['delta'] for w in ws])
                key = f'{lab}|{"cal" if cal else "nocal"}'
                rec[key] = dict(pred=[p['mean'].tolist() for p in P], alarm=[(p['p_exc'] > 0.5).tolist() for p in P], true=T.tolist(),
                                ci90=[p['ci'][0.9].tolist() for p in P])
        out[I] = rec
        print(I, 'actual eff dr', np.round(actual, 2), '| cal geom', np.round(rec['cal_geometry']['x'], 2), '+-', np.round(rec['cal_geometry']['sd'], 2),
              '| cal geom+mass', np.round(rec['cal_geometry+mass']['x'][:3], 2), 'dW=%.0f kN' % (rec['cal_geometry+mass']['x'][3] / 1e3), flush=True)
        for key in ['geometry|nocal', 'geometry|cal', 'geometry+mass|cal']:
            T = np.array(rec[key]['true']); M = np.array(rec[key]['pred']); A = np.array(rec[key]['alarm'])
            print(f'   {key:20s} MAEdam={100*np.abs(M-T)[T>0.05].mean():.2f}  intact FA={int(A[(T==0).all(1)].any(1).sum())}/{int((T==0).all(1).sum())}  pred={np.round(M,3).tolist()}', flush=True)
    dump(out, 'rev_e4_install.json')


# ================================================= E5 data efficiency, noise/bias, window (healthy-trained)
def e5_robust(part='all'):
    out = {}
    rng = np.random.default_rng(1)
    pools = {k: W(v) for k, v in POOLS.items() if v}
    trh = W(TRAIN_HEALTHY)
    for n in ([10, 20, 30, len(trh)] if part in ('all', 'lc') else []):
        for rep in range(3 if n < len(trh) else 1):
            sub = [trh[j] for j in rng.choice(len(trh), n, replace=False)]
            pg = RC.PGFE3(cov='diag', hetero=False, p0=0.9).fit(sub)
            for pn, ws in pools.items():
                r = evaluate(*run_pg(pg, ws)); out[f'lc|N{n}|r{rep}|{pn}'] = {k: r[k] for k in ('sev', 'det', 'cov')}
            print('lc', n, rep, short(out[f'lc|N{n}|r{rep}|ID']), flush=True)
            trm_all = W(TRAIN_MIXED)
            subm = [trm_all[j] for j in rng.choice(len(trm_all), min(n, len(trm_all)), replace=False)] if n < len(trh) else trm_all
            f, _ = RC.tuned_dd('DD-GBM', subm)
            for pn, ws in pools.items():
                rr = evaluate_pt(f(np.array([w['ddf'] for w in ws])), [w['delta'] for w in ws]); out[f'lc|N{n}|r{rep}|DD-GBM|{pn}'] = {k: rr[k] for k in ('sev', 'det')}
    for ns, off in ([(0.5, 0), (1, 0), (2, 0), (4, 0), (1, 1), (1, 2), (2, 2)] if part in ('all', 'noise') else []):
        trn = W(TRAIN_HEALTHY, noise=ns, off=off); trm = W(TRAIN_MIXED, noise=ns, off=off)
        pp = {k: W(v, noise=ns, off=off) for k, v in POOLS.items() if v}
        pg = RC.PGFE3(cov='diag', hetero=False, p0=0.9).fit(trn)
        for pn, ws in pp.items():
            r = evaluate(*run_pg(pg, ws)); out[f'noise|n{ns}|o{off}|PG-FE|{pn}'] = {k: r[k] for k in ('sev', 'det', 'cov')}
        f, _ = RC.tuned_dd('DD-GBM', trm)
        for pn, ws in pp.items():
            rr = evaluate_pt(f(np.array([w['ddf'] for w in ws])), [w['delta'] for w in ws]); out[f'noise|n{ns}|o{off}|DD-GBM|{pn}'] = {k: rr[k] for k in ('sev', 'det')}
        print('noise', ns, off, short(out[f'noise|n{ns}|o{off}|PG-FE|ID']), flush=True)
    for Wl in ([100.0, 200.0, 400.0] if part in ('all', 'win') else []):
        pg = RC.PGFE3(cov='diag', hetero=False, p0=0.9).fit(W(TRAIN_HEALTHY, Wl=Wl))
        trm = W(TRAIN_MIXED, Wl=Wl)
        pp = {k: W(v, Wl=Wl) for k, v in POOLS.items() if v}
        for pn, ws in pp.items():
            r = evaluate(*run_pg(pg, ws)); out[f'win|W{int(Wl)}|PG-FE|{pn}'] = {k: r[k] for k in ('sev', 'det', 'cov')}
        f, _ = RC.tuned_dd('DD-GBM', trm)
        for pn, ws in pp.items():
            rr = evaluate_pt(f(np.array([w['ddf'] for w in ws])), [w['delta'] for w in ws]); out[f'win|W{int(Wl)}|DD-GBM|{pn}'] = {k: rr[k] for k in ('sev', 'det')}
        print('win', Wl, short(out[f'win|W{int(Wl)}|PG-FE|ID']), flush=True)
    dump(out, f'rev_e5_{part}.json')


# ================================================ E6 causal online detection (sliding 100 s windows)
def e6_causal():
    out = {}
    tr100 = W(TRAIN_HEALTHY, Wl=100.0)
    pg = RC.PGFE3(cov='diag', hetero=False, p0=0.9).fit(tr100)
    intact = [n for n in POOLS['ID'] if max(META[n]['delta']) == 0]
    dam = [n for n in POOLS['ID'] + POOLS['OOD-two-line'] + POOLS['OOD-severity'] if max(META[n]['delta']) > 0.05]
    ll = {}
    for n in set(intact + dam):
        ws = sorted(W([n], Wl=100.0), key=lambda w: w['k'])
        ll[n] = [pg.loglik(w) for w in ws]
    rng = np.random.default_rng(5)
    rows = []
    for k_fuse in (1, 2, 4):
        lat, fa_pre, correct = [], 0, []
        for d in dam:
            h = intact[rng.integers(len(intact))]
            seq = ll[h] + ll[d]               # 4 intact windows, then onset, 4 damaged windows
            truth = np.array(META[d]['delta']) > 0.05
            hit = None
            for t in range(1, len(seq) + 1):
                p = pg.post_from_ll(sum(seq[max(0, t - k_fuse):t]))
                al = p['p_exc'] > 0.5
                if t <= len(ll[h]) and al.any():
                    fa_pre += 1
                if t > len(ll[h]) and hit is None and al.any():
                    hit = (t - len(ll[h])) * 100.0
                    correct.append(bool((al == truth).all()))
            lat.append(hit if hit is not None else np.nan)
        lat = np.array(lat)
        out[f'k{k_fuse}'] = dict(latency_s=lat.tolist(), median=float(np.nanmedian(lat)), p90=float(np.nanpercentile(lat, 90)),
                                 detected=float(np.mean(~np.isnan(lat))), alarms_before_onset=fa_pre, n_seq=len(dam),
                                 pre_onset_windows=len(dam) * 4, correct_lines_at_first_alarm=float(np.mean(correct)) if correct else np.nan)
        print('fuse', k_fuse, {k: v for k, v in out[f"k{k_fuse}"].items() if k != 'latency_s'}, flush=True)
    import time
    w = W([intact[0]], Wl=100.0)[0]
    t0 = time.time(); [dt.mooring_table(w) for _ in range(5)]; tt = (time.time() - t0) / 5
    t0 = time.time(); [pg.predict(w) for _ in range(10)]; tp = (time.time() - t0) / 10
    out['compute_s'] = dict(table=tt, posterior=tp)
    dump(out, 'rev_e6_causal.json')


# ================================================ E7 virtual sensing + fatigue, retrospective vs causal
def e7_vs():
    import vsense as V
    out = {}
    trm200 = W(TRAIN_MIXED, Wl=200.0)
    trh200 = W(TRAIN_HEALTHY, Wl=200.0)
    pg = RC.PGFE3(cov='diag', hetero=False, p0=0.9).fit(trh200)
    dtr = [w['delta'] for w in trm200]
    mods = {'PG-VS': V.TensionVS('PG').fit(trm200, dtr), 'DD ridge + d': V.TensionVS('DD').fit(trm200, dtr),
            'DD-MLP + d': V.TensionVS('DD-MLP').fit(trm200, dtr), 'DD ridge, no d': V.TensionVS('DDnod').fit(trm200, dtr)}
    A0 = np.pi * 0.0766**2 / 4 * 2   # chain: two legs per link -> nominal steel area (m^2)
    for pn in ['ID', 'OOD-severity', 'OOD-two-line', 'OOD-extreme-sea']:
        ws = W(POOLS[pn], Wl=200.0); by = {}
        for w in ws:
            by.setdefault(w['name'], {})[w['k']] = w
        rows = []
        for nm, d in by.items():
            if 0 not in d or 1 not in d:
                continue
            w0, w1 = d[0], d[1]
            dh = {'retrospective': pg.predict(w1)['mean'], 'causal': pg.predict(w0)['mean']}
            S = w1['S']
            for i in range(3):
                y = w1['truth'][f'T{i+1}'][::5][:len(S['t'])].astype(float); n = len(y)
                dl_true = w1['delta'][i]
                ref_T = V.DEL(y, 3.0, V.FS); ref_S = ref_T / (A0 * (1 - dl_true))
                for mode, dd in dh.items():
                    for lab, m in mods.items():
                        p = m.predict(S, dd, i)[:n]
                        dT = V.DEL(p, 3.0, V.FS)
                        rows.append(dict(name=nm, line=i, mode=mode, model=lab, nrmse=float(np.sqrt(np.mean((p - y)**2)) / np.std(y)),
                                         del_T_err=float(dT / ref_T - 1), del_S_err=float(dT / (A0 * (1 - dd[i])) / ref_S - 1)))
        out[pn] = rows
        for mode in ('retrospective', 'causal'):
            for lab in mods:
                rr = [r for r in rows if r['mode'] == mode and r['model'] == lab]
                print(f'{pn:16s} {mode:13s} {lab:16s} NRMSE={np.mean([r["nrmse"] for r in rr]):.3f} |DEL_T err|={100*np.mean([abs(r["del_T_err"]) for r in rr]):.1f}% |DEL_stress err|={100*np.mean([abs(r["del_S_err"]) for r in rr]):.1f}%', flush=True)
    dump(out, 'rev_e7_vs.json')


# ============================================ E8 forecasting: causality check + data-scarce regime
def e8_forecast():
    import forecast as F
    out = {}
    # causality: kernels must have no weight on future buoy samples (lag index >= 0 only by construction)
    K = F.propagation_kernels(np.arange(-30, 20.01, 1.0))
    out['kernel_lags'] = dict(n_lags=int(K.shape[1]), first_lag_s=0.0, last_lag_s=float((K.shape[1] - 1) / F.FS))
    trall = W(TRAIN_MIXED); te = W(POOLS['ID'])
    rng = np.random.default_rng(3)
    for n in [3, 5, 10, 20, len(trall)]:
        for rep in range(3 if n < len(trall) else 1):
            tr = [trall[j] for j in rng.choice(len(trall), n, replace=False)]
            for tg in ['pitch', 'surge']:
                for h in [5.0, 10.0]:
                    for mode in ['DD', 'PG']:
                        m = F.Forecaster(tg, h, mode).fit(tr)
                        P, T_, Y0 = [], [], []
                        for w in te:
                            p, t, y0 = m.predict(w); P.append(p); T_.append(t); Y0.append(y0)
                        s = F.score(np.concatenate(P), np.concatenate(T_), np.concatenate(Y0))
                        out[f'N{n}|r{rep}|{tg}|{h}|{mode}'] = s
            print('fc N', n, rep, {k.split('|', 2)[2]: round(v['skill_vs_persistence'], 3) for k, v in out.items() if k.startswith(f'N{n}|r{rep}|')}, flush=True)
    dump(out, 'rev_e8_fc.json')


# ============================================ E9 reduced- vs full-fidelity OpenFAST settings
def e9_fidelity():
    from openfast_toolbox.io import FASTOutputFile
    out = {}
    ch = ['PtfmSurge_[m]', 'PtfmSway_[m]', 'PtfmHeave_[m]', 'PtfmRoll_[deg]', 'PtfmPitch_[deg]', 'PtfmYaw_[deg]',
          'FAIRTEN1_[N]', 'FAIRTEN2_[N]', 'FAIRTEN3_[N]', 'TwrBsMyt_[kN-m]', 'GenPwr_[kW]']
    for nm in ['train_001', 'test_010', 'ood_009', 'ood_019']:
        fp = f'{ROOT}/runs_ff/{nm}/5MW_OC4Semi_WSt_WavesWN.outb'
        if not os.path.exists(fp):
            continue
        a = FASTOutputFile(f'{ROOT}/runs/{nm}/5MW_OC4Semi_WSt_WavesWN.outb').toDataFrame()
        b = FASTOutputFile(fp).toDataFrame()
        t = np.arange(120, 520, 0.05)
        rec = {}
        for c in ch:
            x = np.interp(t, a['Time_[s]'], a[c]); y = np.interp(t, b['Time_[s]'], b[c])
            rec[c] = dict(nrmse=float(np.sqrt(np.mean((x - y) ** 2)) / np.std(y)), mean_diff_rel=float((x.mean() - y.mean()) / (abs(y.mean()) + 1e-9)),
                          std_ratio=float(x.std() / y.std()))
        # effect on the estimator: run PG-FE on windows extracted from the full-fidelity run
        out[nm] = dict(meta=META[nm], channels=rec)
        print(nm, {c.split('_')[0]: round(v['nrmse'], 3) for c, v in rec.items()}, flush=True)
    dump(out, 'rev_e9_fidelity.json')


def e5_lc(): e5_robust('lc')
def e5_noise(): e5_robust('noise')
def e5_win(): e5_robust('win')


if __name__ == '__main__':
    for k in sys.argv[1:]:
        globals()[k]()
