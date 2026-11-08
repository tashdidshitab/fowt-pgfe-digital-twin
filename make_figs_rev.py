"""Revision figures. All numbers are read from results/rev_*.json. Output: figures/Figure_N.pdf (+png)."""
from paths import ROOT, RTEST
import json, os, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = ROOT + '/results'; OUT = ROOT + '/figures'; os.makedirs(OUT, exist_ok=True)
C = dict(pg='#2a78d6', gbm='#eb6834', rf='#1baf7a', mlp='#eda100', vio='#6250d6', grey='#8a8984', ink='#0b0b0b', ink2='#52514e', grid='#e4e3df')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': C['ink2'], 'axes.labelcolor': C['ink'],
                     'xtick.color': C['ink2'], 'ytick.color': C['ink2'], 'axes.grid': True, 'grid.color': C['grid'],
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'lines.linewidth': 2, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})
J = lambda f: json.load(open(f'{R}/{f}'))


def save(fig, n):
    fig.savefig(f'{OUT}/Figure_{n}.pdf'); fig.savefig(f'{OUT}/Figure_{n}.png'); plt.close(fig)


def fig_installs(n):
    E4 = J('rev_e4_install.json')
    I = [k for k in ['I1', 'I2', 'I3', 'I4', 'I5'] if k in E4]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
    a = np.array([E4[k]['actual_eff_dr'] for k in I]).ravel()
    for lab, key, mk, col in [('Geometry calibration', 'cal_geometry', 'o', C['pg']), ('Geometry + weight offset', 'cal_geometry+mass', 's', C['vio'])]:
        e = np.array([E4[k][key]['x'][:3] for k in I]).ravel(); s = np.array([E4[k][key]['sd'][:3] for k in I]).ravel()
        ax[0].errorbar(a, e, yerr=2 * s, fmt=mk, ms=5, color=col, mec='white', mew=0.6, label=lab, elinewidth=1, capsize=0, zorder=3)
    for j, k in enumerate(I):
        if k == 'I5':
            for x, y in zip(E4[k]['actual_eff_dr'], E4[k]['cal_geometry']['x'][:3]):
                ax[0].annotate('I5', (x, y), xytext=(4, 2), textcoords='offset points', fontsize=7, color=C['ink2'])
    lim = [min(a.min(), -8) - 1, max(a.max(), 12) + 1]
    ax[0].plot(lim, lim, color=C['grey'], lw=1, zorder=1); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel('Actual effective anchor-radius error (m)'); ax[0].set_ylabel('Calibrated estimate (m)')
    ax[0].set_title('(a) Commissioning calibration (bars: ±2σ Laplace)', loc='left', fontsize=9); ax[0].legend(loc='upper left', fontsize=8)
    x = np.arange(len(I)); wd = 0.27
    for j, (key, lab, col) in enumerate([('geometry|nocal', 'No calibration', C['grey']), ('geometry|cal', 'Geometry calibration', C['pg']),
                                          ('geometry+mass|cal', 'Geometry + weight offset', C['vio'])]):
        v = []
        for k in I:
            T = np.array(E4[k][key]['true']); M = np.array(E4[k][key]['pred']); v.append(100 * np.abs(M - T)[T > 0.05].mean())
        b = ax[1].bar(x + (j - 1) * wd, v, wd - 0.03, color=col, label=lab, zorder=3)
    ax[1].set_yscale('log'); ax[1].set_xticks(x); ax[1].set_xticklabels(I); ax[1].set_xlabel('Installation')
    ax[1].set_ylabel('MAE on degraded lines (% points)'); ax[1].set_title('(b) Degradation error on test runs', loc='left', fontsize=9)
    ax[1].legend(fontsize=7.5, loc='upper left', ncol=1)
    fig.tight_layout(); save(fig, n)


def fig_vs(n):
    E7 = J('rev_e7_vs.json')
    pools = ['ID', 'OOD-severity', 'OOD-two-line', 'OOD-extreme-sea']
    plab = ['ID', 'OOD\nseverity', 'OOD\ntwo-line', 'OOD\nextreme sea']
    models = [('DD ridge, no d', 'Ridge, no δ̂', C['gbm']), ('DD ridge + d', 'Ridge + δ̂', C['rf']), ('PG-VS', 'PG-VS + δ̂', C['pg']), ('DD-MLP + d', 'MLP + δ̂', C['mlp'])]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.2))
    x = np.arange(len(pools)); wd = 0.2
    for j, (k, lab, col) in enumerate(models):
        for a_, key in [(ax[0], 'nrmse'), (ax[1], 'del_T_err')]:
            v = [np.mean([abs(r[key]) for r in E7[p] if r['mode'] == 'causal' and r['model'] == k]) * (1 if key == 'nrmse' else 100) for p in pools]
            vr = [np.mean([abs(r[key]) for r in E7[p] if r['mode'] == 'retrospective' and r['model'] == k]) * (1 if key == 'nrmse' else 100) for p in pools]
            a_.bar(x + (j - 1.5) * wd, v, wd - 0.03, color=col, label=lab if a_ is ax[0] else None, zorder=3)
            a_.scatter(x + (j - 1.5) * wd, vr, marker='_', s=60, color=C['ink'], zorder=4, label='retrospective' if (a_ is ax[0] and j == 0) else None)
    for a_, yl, tl in [(ax[0], 'Fairlead-tension NRMSE (–)', '(a) Tension reconstruction'), (ax[1], '|DEL error| (%)', '(b) Fatigue, tension DEL (m = 3)')]:
        a_.set_xticks(x); a_.set_xticklabels(plab); a_.set_ylabel(yl); a_.set_title(tl + '; bars causal, ticks retrospective', loc='left', fontsize=8.5)
    ax[0].set_ylim(0, 3.2); ax[0].legend(fontsize=7.5, ncol=3, loc='upper right')
    fig.tight_layout(); save(fig, n)


def fig_fc(n):
    FC0 = json.load(open(f'{R}/fc_main.json')); E8 = J('rev_e8_fc.json')
    H = [2.0, 5.0, 10.0, 15.0, 20.0]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for lab, name, col, mk in [('AR', 'Motion history only', C['grey'], 'v'), ('DD', 'Ridge + raw buoy', C['gbm'], 's'),
                               ('DD-MLP', 'MLP + raw buoy', C['mlp'], '^'), ('PG', 'Ridge + propagated wave', C['pg'], 'o')]:
        for a_, tg in zip(ax[:2], ['pitch', 'surge']):
            a_.plot(H, [FC0[f'test|{tg}|{h}|{lab}']['skill_vs_persistence'] for h in H], marker=mk, ms=4, color=col, label=name)
    for a_, tg, tl in zip(ax[:2], ['pitch', 'surge'], ['(a) Pitch', '(b) Surge']):
        a_.set_xlabel('Forecast horizon (s)'); a_.set_xticks(H); a_.set_ylim(0, 1.02); a_.set_title(tl, loc='left', fontsize=9)
    ax[0].set_ylabel('Skill vs persistence'); ax[0].legend(fontsize=7, loc='lower left')
    Ns = [3, 5, 10, 20, 110]
    for tg, ls in [('pitch', '-'), ('surge', '--')]:
        for m, col in [('DD', C['gbm']), ('PG', C['pg'])]:
            v = [np.mean([E8[k]['skill_vs_persistence'] for k in E8 if k.startswith(f'N{N}|') and k.endswith(f'|{tg}|10.0|{m}')]) for N in Ns]
            ax[2].plot(Ns, v, ls, marker='o', ms=4, color=col, label=f'{"Ridge + raw buoy" if m=="DD" else "Propagated wave"}, {tg}')
    ax[2].set_xscale('log'); ax[2].set_xticks(Ns); ax[2].set_xticklabels(Ns); ax[2].set_ylim(-0.4, 1.0)
    ax[2].set_xlabel('Training simulations'); ax[2].set_title('(c) 10 s horizon vs training size', loc='left', fontsize=9); ax[2].legend(fontsize=7, loc='lower right')
    fig.tight_layout(); save(fig, n)


POOLS = ['ID', 'OOD-severity', 'OOD-two-line', 'OOD-extreme-sea', 'OOD-misalign']
PLAB = ['ID', 'OOD\nseverity', 'OOD\ntwo-line', 'OOD\nextreme sea', 'OOD\nmisalign.']


def fig_campaign(n):
    import extract, physics
    D0 = json.load(open(ROOT + '/design.json')); DR = json.load(open(ROOT + '/design_rev.json'))
    E3 = J('rev_e3_mismatch.json')
    fig, ax = plt.subplots(1, 3, figsize=(10.8, 3.1), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    for cs, col, lab in [([c for c in D0 if c['split'] == 'train'] + [c for c in DR if c['split'] == 'htrain'], C['pg'], 'Training'),
                         ([c for c in D0 if c['split'] == 'test'] + [c for c in DR if c['split'] == 'rev'], C['gbm'], 'Evaluation (ID/OOD, rev.)'),
                         ([c for c in D0 if c['split'] == 'ood'], C['rf'], 'OOD (main)'),
                         ([c for c in DR if c['split'] in ('mm', 'inst')], C['vio'], 'Mismatch / installations')]:
        ax[0].scatter([c['U'] for c in cs], [c['Hs'] for c in cs], s=10, color=col, label=lab, edgecolor='white', linewidth=0.4, zorder=3)
    ax[0].set_xlabel('Mean hub-height wind speed (m/s)'); ax[0].set_ylabel('$H_s$ (m)'); ax[0].legend(fontsize=6.5, loc='upper left')
    ax[0].set_title('(a) Environmental conditions', loc='left', fontsize=9)
    a, b = [], []
    for nm in extract.available():
        D = extract.load(nm)
        if D['meta'].get('split') in ('mm', 'inst') or 'segs' in D['meta'] and any(len(s) > 1 for s in D['meta']['segs']):
            continue
        q = np.c_[D['surge'], D['sway'], D['heave'], np.radians(D['roll']), np.radians(D['pitch']), np.radians(D['yaw'])][::5]
        for i in range(3):
            a.append(D[f'T{i+1}'].mean() / 1e3); b.append(physics.line_loads(q, D['meta']['delta'][i], i)[1].mean() / 1e3)
    a, b = np.array(a), np.array(b); err = (b - a) / a * 100
    ax[1].plot([a.min(), a.max()], [a.min(), a.max()], color=C['grey'], lw=1)
    ax[1].scatter(a, b, s=6, color=C['pg'], zorder=3)
    ax[1].set_xlabel('MoorDyn mean fairlead tension (kN)'); ax[1].set_ylabel('Quasi-static core (kN)')
    ax[1].set_title('(b) Mean tension, true pose', loc='left', fontsize=9)
    ax[1].text(0.04, 0.95, f'n = {len(a)} line-windows\nmean |err| = {np.mean(abs(err)):.2f}%\nmax |err| = {np.max(abs(err)):.2f}%', transform=ax[1].transAxes, va='top', fontsize=7.5, color=C['ink2'])
    comps = ['$F_x$', '$F_y$', '$F_z$', '$M_x$', '$M_y$', '$M_z$']
    nom = np.array(E3['qs_vs_md_nominal']['over_sigma'])
    groups = [('B1-EA-0.9', 'EA×0.9'), ('B2-mass-0.97', 'w×0.97'), ('B3-friction', 'friction'), ('B4-fair-random', 'fairlead')]
    x = np.arange(6); wd = 0.16
    ax[2].bar(x - 2 * wd, nom, wd - 0.02, color=C['pg'], label='nominal', zorder=3)
    cols = [C['gbm'], C['rf'], C['mlp'], C['vio']]
    for j, (g, lab) in enumerate(groups):
        v = np.mean([np.abs(r['eq_err_over_sigma']) for r in E3[g] if 'eq_err_over_sigma' in r], 0)
        ax[2].bar(x + (j - 1) * wd, v, wd - 0.02, color=cols[j], label=lab, zorder=3)
    ax[2].axhline(1, color=C['ink2'], lw=0.8, ls='--')
    ax[2].set_yscale('log'); ax[2].set_xticks(x); ax[2].set_xticklabels(comps); ax[2].set_ylabel('|core − MoorDyn| / σ (–)')
    ax[2].set_ylim(5e-3, 60); ax[2].set_title('(c) Six-component mean-load error', loc='left', fontsize=9); ax[2].legend(fontsize=6.5, ncol=3, loc='upper left')
    fig.tight_layout(); save(fig, n)
    json.dump(dict(n=int(len(a)), mean_abs_pct=float(np.mean(abs(err))), max_abs_pct=float(np.max(abs(err)))), open(f'{R}/rev_core_tension.json', 'w'))


def fig_parity(n):
    E1 = J('rev_e1_main.json')
    fig, ax = plt.subplots(1, 3, figsize=(10.8, 3.4), gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    for a_, pools, title in [(ax[0], ['ID'], '(a) In-distribution'), (ax[1], ['OOD-severity', 'OOD-two-line', 'OOD-extreme-sea', 'OOD-misalign'], '(b) Out-of-distribution')]:
        a_.plot([0, 0.7 * 100], [0, 70], color=C['grey'], lw=1, zorder=1); a_.axvspan(35, 70, color=C['grid'], alpha=0.5, lw=0)
        for m, lab, mk, col in [('DD-GBM', 'DD-GBM (tuned)', 's', C['gbm']), ('PG-FE (healthy-trained)', 'PG-FE (intact-trained)', 'o', C['pg'])]:
            T = np.concatenate([np.array(E1[f'S3_full_sparse|{m}|{p}']['true']) for p in pools]) * 100
            P = np.concatenate([np.array(E1[f'S3_full_sparse|{m}|{p}']['pred']) for p in pools]) * 100
            a_.scatter(T.ravel(), P.ravel(), s=12, marker=mk, color=col, edgecolor='white', linewidth=0.4, label=lab, zorder=3)
        a_.set_xlim(-2, 70); a_.set_ylim(-2, 70); a_.set_xlabel('True area loss δ (%)'); a_.set_title(title, loc='left', fontsize=9)
    ax[0].set_ylabel('Estimated δ (%)'); ax[0].legend(fontsize=7.5, loc='upper left')
    ax[1].axvline(60, color=C['ink2'], lw=0.8, ls=':'); ax[1].text(61, 3, 'grid\nlimit', fontsize=6.5, color=C['ink2'])
    x = np.arange(len(POOLS)); meth = [('PG-FE (healthy-trained)', 'PG-FE', C['pg']), ('WLS (physics)', 'WLS (same core)', C['vio']),
                                      ('DD-GBM', 'DD-GBM', C['gbm']), ('DD-RF', 'DD-RF', C['rf']), ('DD-MLP', 'DD-MLP', C['mlp'])]
    wd = 0.16
    for j, (m, lab, col) in enumerate(meth):
        v = [100 * E1[f'S3_full_sparse|{m}|{p}']['sev']['MAE_dam'] for p in POOLS]
        ci = [E1[f'S3_full_sparse|{m}|{p}']['sev'].get('MAE_dam_ci', [np.nan, np.nan]) for p in POOLS]
        lo = [v[k] - 100 * ci[k][0] for k in range(len(v))]; hi = [100 * ci[k][1] - v[k] for k in range(len(v))]
        ax[2].bar(x + (j - 2) * wd, v, wd - 0.02, color=col, label=lab, zorder=3, yerr=[lo, hi], error_kw=dict(lw=0.7, ecolor=C['ink2']))
    ax[2].set_yscale('log'); ax[2].set_xticks(x); ax[2].set_xticklabels(PLAB, fontsize=7.5); ax[2].set_ylabel('MAE, degraded lines (% points)')
    ax[2].set_ylim(0.08, 200); ax[2].set_title('(c) Severity error (95% bootstrap CI)', loc='left', fontsize=9); ax[2].legend(fontsize=6.5, ncol=3, loc='upper left')
    fig.tight_layout(); save(fig, n)


def fig_uq(n):
    E1 = J('rev_e1_main.json'); E2 = J('rev_e2_uq.json')
    fig, ax = plt.subplots(1, 3, figsize=(10.8, 3.2))
    lv = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    ax[0].plot([0.45, 1], [0.45, 1], color=C['grey'], lw=1)
    S = E2['stratified']
    for p, col, mk in zip(POOLS, [C['pg'], C['gbm'], C['rf'], C['mlp'], C['vio']], 'osd^v'):
        ks = [k for k in S if k.startswith(p + '|') and float(k.split('|')[1].split('-')[0]) >= 0.05]
        nn = sum(S[k]['n'] for k in ks)
        cv = [sum(S[k]['n'] * S[k]['cov'][str(a)] for k in ks) / nn for a in lv]
        ax[0].plot(lv, cv, marker=mk, ms=4, color=col, label=f'{p} (n={nn})')
    ax[0].set_xlabel('Nominal credible level'); ax[0].set_ylabel('Empirical coverage, degraded lines'); ax[0].set_ylim(0.0, 1.02)
    ax[0].set_title('(a) Reliability of credible intervals', loc='left', fontsize=9); ax[0].legend(fontsize=6.5, loc='lower right')
    I = [x for x in E2['ident'] if x['at'] == 'intact']
    ax[1].hist(np.log10([x['cond6'] for x in I]), bins=20, color=C['pg'], label='6-DOF balance', alpha=0.9)
    ax[1].hist(np.log10([x['condH'] for x in I]), bins=20, color=C['gbm'], label='horizontal only', alpha=0.9)
    ax[1].set_xlabel('log$_{10}$ cond(Fisher information)'); ax[1].set_ylabel('Windows'); ax[1].legend(fontsize=7)
    ax[1].set_title('(b) Identifiability at δ = 0', loc='left', fontsize=9)
    p0s = [0.5, 0.75, 0.9, 0.97, 0.99]
    for cov, het, col, lab in [('diag', 'False', C['pg'], 'diag. Σ'), ('full', 'False', C['gbm'], 'full Σ'), ('full', 'True', C['rf'], 'full Σ, heterosc.')]:
        ax[2].plot(p0s, [100 * E2[f'{cov}|het={het}|p0={p}|OOD-two-line']['sev']['MAE_all'] for p in p0s], marker='o', ms=4, color=col, label=lab)
    ax[2].set_xlabel('Prior $p_0$ = P(δ$_i$ = 0)'); ax[2].set_ylabel('MAE, all lines, two-line OOD (% pts)')
    ax[2].set_title('(c) Prior and covariance sensitivity', loc='left', fontsize=9); ax[2].legend(fontsize=7)
    fig.tight_layout(); save(fig, n)


def fig_mismatch(n):
    E3 = J('rev_e3_mismatch.json')
    groups = ['B1-EA-0.9', 'B1-EA-1.1', 'B2-mass-0.97', 'B2-mass-1.03', 'B3-drag', 'B3-friction', 'B4-fair-uniform', 'B4-fair-random',
              'B5-ptfm-mass-0.99', 'B5-ptfm-mass-0.995', 'B5-ptfm-mass-1.005', 'B5-ptfm-mass-1.01']
    lab = ['EA×0.9', 'EA×1.1', 'w×0.97', 'w×1.03', 'drag', 'friction', 'fair. unif.', 'fair. rand.', 'ptfm −1%', 'ptfm −0.5%', 'ptfm +0.5%', 'ptfm +1%']
    fig, ax = plt.subplots(1, 2, figsize=(10.8, 3.3), gridspec_kw=dict(width_ratios=[1.7, 1]))
    x = np.arange(len(groups)); wd = 0.38
    for j, (pk, ak, col, nm) in enumerate([('pred', 'alarm', C['pg'], 'PG-FE'), ('pred_mass', 'alarm_mass', C['vio'], 'PG-FE + weight nuisance')]):
        v, fa = [], []
        for g in groups:
            rows = E3[g]; T = np.array([r['delta'] for r in rows]); M = np.array([r[pk] for r in rows]); A = np.array([r[ak] for r in rows])
            v.append(100 * np.abs(M - T)[T > 0.05].mean()); intact = (T == 0).all(1); fa.append(A[intact].any(1).sum())
        ax[0].bar(x + (j - 0.5) * wd, v, wd - 0.03, color=col, label=nm, zorder=3)
        for k in range(len(groups)):
            if fa[k]:
                ax[0].text(x[k] + (j - 0.5) * wd, v[k] * 1.08, f'FA{fa[k]}', ha='center', fontsize=6.5, color=C['ink'])
    ax[0].set_yscale('log'); ax[0].set_xticks(x); ax[0].set_xticklabels(lab, rotation=35, ha='right', fontsize=7.5)
    ax[0].set_ylabel('MAE, degraded lines (% points)'); ax[0].legend(fontsize=7.5, loc='upper left')
    ax[0].set_title('(a) Twin keeps nominal model; "FAk": false alarms on intact runs', loc='left', fontsize=8.5)
    rows = E3['B6-localized']
    for r in rows:
        L = r['loc']; col = C['gbm'] if L['where'] == 'touchdown' else C['pg']; mk = 'o' if L['delta_loc'] < 0.4 else 's'
        ax[1].scatter(100 * L['delta_eq_weight'], 100 * r['pred'][L['line']], color=col, marker=mk, s=28, edgecolor='white', zorder=3)
    ax[1].plot([0, 10], [0, 10], color=C['grey'], lw=1); ax[1].axhline(5, color=C['ink2'], lw=0.8, ls=':')
    from matplotlib.lines import Line2D
    ax[1].legend(handles=[Line2D([], [], color=C['pg'], marker='o', ls='', label='fairlead region'), Line2D([], [], color=C['gbm'], marker='o', ls='', label='touchdown region'),
                          Line2D([], [], color=C['ink2'], marker='o', ls='', label='30% local loss'), Line2D([], [], color=C['ink2'], marker='s', ls='', label='50% local loss')], fontsize=6.5, loc='upper left')
    ax[1].set_xlabel('Weight-equivalent uniform loss (%)'); ax[1].set_ylabel('Estimated δ (%)'); ax[1].set_xlim(0, 10); ax[1].set_ylim(-0.5, 10)
    ax[1].set_title('(b) Localised section loss (dotted: δ*)', loc='left', fontsize=9)
    fig.tight_layout(); save(fig, n)


def fig_robust(n):
    L = J('rev_e5_lc.json'); Nz = J('rev_e5_noise.json'); Wn = J('rev_e5_win.json')
    fig, ax = plt.subplots(1, 3, figsize=(10.8, 3.1), gridspec_kw=dict(width_ratios=[1, 1.35, 0.8]))
    Ns = sorted({int(k.split('|')[1][1:]) for k in L})
    for m, key, col, mk in [('PG-FE (intact runs)', '', C['pg'], 'o'), ('DD-GBM (labelled runs)', 'DD-GBM|', C['gbm'], 's')]:
        mu, lo, hi = [], [], []
        for N in Ns:
            v = [100 * L[k]['sev']['MAE_dam'] for k in L if k.startswith(f'lc|N{N}|') and k.endswith(f'|{key}ID') and (key or 'DD-GBM' not in k)]
            mu.append(np.mean(v)); lo.append(min(v)); hi.append(max(v))
        ax[0].fill_between(Ns, lo, hi, color=col, alpha=0.15, lw=0); ax[0].plot(Ns, mu, marker=mk, ms=4, color=col, label=m)
    ax[0].set_yscale('log'); ax[0].set_xticks(Ns); ax[0].set_xticklabels([str(v) for v in Ns[:-1]] + [f'{Ns[-1]} | 110'], fontsize=8); ax[0].set_xlabel('Training runs (last point: PG-FE | DD-GBM)'); ax[0].set_ylabel('MAE, degraded lines, ID (% pts)')
    ax[0].set_title('(a) Data efficiency', loc='left', fontsize=9); ax[0].legend(fontsize=7)
    cfg = [('n0.5|o0', '0.5×'), ('n1|o0', '1×'), ('n2|o0', '2×'), ('n4|o0', '4×'), ('n1|o1', '1×+off'), ('n1|o2', '1×+2off'), ('n2|o2', '2×+2off')]
    x = np.arange(len(cfg)); wd = 0.38
    for j, (m, col) in enumerate([('PG-FE', C['pg']), ('DD-GBM', C['gbm'])]):
        ax[1].bar(x + (j - 0.5) * wd, [100 * Nz[f'noise|{c}|{m}|ID']['sev']['MAE_dam'] for c, _ in cfg], wd - 0.03, color=col, label=m, zorder=3)
    ax[1].set_xticks(x); ax[1].set_xticklabels([l for _, l in cfg], fontsize=7.5); ax[1].set_xlabel('Noise level × nominal; per-window offsets')
    ax[1].set_title('(b) Sensor noise and bias (ID)', loc='left', fontsize=9); ax[1].legend(fontsize=7)
    Ws = [100, 200, 400]; x = np.arange(3)
    for j, (m, col) in enumerate([('PG-FE', C['pg']), ('DD-GBM', C['gbm'])]):
        ax[2].bar(x + (j - 0.5) * wd, [100 * Wn[f'win|W{W_}|{m}|ID']['sev']['MAE_dam'] for W_ in Ws], wd - 0.03, color=col, zorder=3)
    ax[2].set_xticks(x); ax[2].set_xticklabels([f'{W_} s' for W_ in Ws]); ax[2].set_xlabel('Averaging window')
    ax[2].set_title('(c) Window length (ID)', loc='left', fontsize=9)
    fig.tight_layout(); save(fig, n)


if __name__ == '__main__':
    todo = sys.argv[1:]
    for t in todo:
        nm, num = t.split(':')
        globals()[f'fig_{nm}'](int(num)); print('done', nm, num)
