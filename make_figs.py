"""Figures and tables for the manuscript. All numbers are read from results/*.json."""
from paths import ROOT, RTEST
import json, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import extract, physics, sensors

R = ROOT + '/results'; FIG = ROOT + '/figures/first_version'; os.makedirs(FIG, exist_ok=True)
C = dict(pg='#2a78d6', gbm='#eb6834', rf='#1baf7a', mlp='#eda100', grey='#8a8984', ink='#0b0b0b', ink2='#52514e', grid='#e4e3df')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': C['ink2'], 'axes.labelcolor': C['ink'],
                     'xtick.color': C['ink2'], 'ytick.color': C['ink2'], 'axes.grid': True, 'grid.color': C['grid'],
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'lines.linewidth': 2, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
MC = {'PG-FE': C['pg'], 'DD-GBM': C['gbm'], 'DD-RF': C['rf'], 'DD-MLP': C['mlp']}
LAB = {'PG-FE': 'PG-FE (proposed)', 'DD-GBM': 'DD-GBM', 'DD-RF': 'DD-RF', 'DD-MLP': 'DD-MLP'}
load = lambda f: json.load(open(f'{R}/{f}'))
T = {}   # numbers for the manuscript


def agg(res, prefix, method, split, metric):
    v = [res[k][split]['metrics'][metric] for k in res if k.startswith(prefix) and k.endswith('|' + method)]
    return np.mean(v), np.std(v), len(v)


# ---------------------------------------------------------------- Fig 2: campaign + model validation
def fig_campaign():
    design = json.load(open(ROOT + '/design.json'))
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for s, col, lab in [('train', C['pg'], 'Train (110)'), ('test', C['gbm'], 'Test (30)'), ('ood', C['rf'], 'OOD (24)')]:
        cs = [c for c in design if c['split'] == s]
        ax[0].scatter([c['U'] for c in cs], [c['Hs'] for c in cs], s=16, color=col, label=lab, edgecolor='white', linewidth=0.6, zorder=3)
        dmax = [max(c['delta']) * 100 for c in cs]; ndam = [sum(np.array(c['delta']) > 0) for c in cs]
        ax[1].scatter(np.array(ndam) + {'train': -0.15, 'test': 0, 'ood': 0.15}[s] + np.random.default_rng(1).uniform(-0.05, 0.05, len(cs)),
                      dmax, s=16, color=col, edgecolor='white', linewidth=0.6, zorder=3)
    ax[0].set_xlabel('Hub-height mean wind speed U (m/s)'); ax[0].set_ylabel('Significant wave height Hs (m)'); ax[0].legend(loc='upper left')
    ax[0].set_title('(a) Environmental conditions', loc='left', fontsize=9)
    ax[1].set_xticks([0, 1, 2]); ax[1].set_xlabel('Number of degraded lines'); ax[1].set_ylabel('Max. area loss δ (%)')
    ax[1].set_title('(b) Degradation states', loc='left', fontsize=9)
    # QS catenary vs MoorDyn window-mean tension, all runs, true pose
    a, b = [], []
    for nm in extract.available():
        D = extract.load(nm)
        q = np.c_[D['surge'], D['sway'], D['heave'], np.radians(D['roll']), np.radians(D['pitch']), np.radians(D['yaw'])][::5]
        for i in range(3):
            a.append(D[f'T{i+1}'].mean() / 1e3); b.append(physics.line_loads(q, D['meta']['delta'][i], i)[1].mean() / 1e3)
    a, b = np.array(a), np.array(b); err = (b - a) / a * 100
    T['qs_val'] = dict(n=len(a), mean_abs_pct=float(np.mean(abs(err))), max_abs_pct=float(np.max(abs(err))), rng=[float(a.min()), float(a.max())])
    ax[2].plot([a.min() * 0.95, a.max() * 1.05], [a.min() * 0.95, a.max() * 1.05], color=C['grey'], lw=1, zorder=1)
    ax[2].scatter(a, b, s=10, color=C['pg'], edgecolor='white', linewidth=0.4, zorder=3)
    ax[2].set_xlabel('MoorDyn mean fairlead tension (kN)'); ax[2].set_ylabel('Quasi-static model (kN)')
    ax[2].set_title('(c) Physics-core validation', loc='left', fontsize=9)
    ax[2].text(0.04, 0.92, f'n = {len(a)} line-windows\nmean |error| = {np.mean(abs(err)):.2f}%\nmax |error| = {np.max(abs(err)):.2f}%',
               transform=ax[2].transAxes, va='top', fontsize=8, color=C['ink2'])
    fig.tight_layout(); fig.savefig(f'{FIG}/fig2_campaign_validation.png'); plt.close(fig)


# ---------------------------------------------------------------- Table 2 + Fig 3 parity
def tab_main():
    res = load('fault_main.json')
    rows = []
    for sset in sensors.SETS:
        for m in ['PG-FE', 'DD-GBM', 'DD-RF', 'DD-MLP']:
            row = dict(sset=sset, method=m)
            for split in ['test', 'ood']:
                for met in ['MAE', 'MAE_damaged_lines', 'det_F1', 'false_alarm', 'loc_acc', 'CI90_coverage']:
                    try:
                        mu, sd, n = agg(res, sset, m, split, met)
                        row[f'{split}_{met}'] = (mu, sd)
                    except KeyError:
                        pass
            rows.append(row)
    abl = []
    for m in ['PG-FE', 'abl: horizontal DOFs only', 'abl: uniform prior', 'abl: GP env model']:
        row = dict(method=m)
        for split in ['test', 'ood']:
            for met in ['MAE', 'MAE_damaged_lines', 'det_F1', 'false_alarm', 'CI90_coverage']:
                row[f'{split}_{met}'] = agg(res, 'S3_full_sparse', m, split, met)[:2]
        abl.append(row)
    T['main'] = rows; T['ablation'] = abl
    # parity figure for S3, seed 0
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for a_, split, title in [(ax[0], 'test', '(a) Held-out test'), (ax[1], 'ood', '(b) Out-of-distribution')]:
        a_.plot([0, 55], [0, 55], color=C['grey'], lw=1, zorder=1)
        a_.axvspan(35, 55, color=C['grid'], alpha=0.5, lw=0, zorder=0)
        for m, mk in [('DD-GBM', 's'), ('PG-FE', 'o')]:
            r = res[f'S3_full_sparse|s0|{m}'][split]
            tr_, pr = np.array(r['true']) * 100, np.array(r['pred']) * 100
            a_.scatter(tr_.ravel(), pr.ravel(), s=18, marker=mk, color=MC[m], edgecolor='white', linewidth=0.5, label=LAB[m], zorder=3)
        a_.set_xlim(-2, 55); a_.set_ylim(-2, 55); a_.set_xlabel('True area loss δ (%)'); a_.set_title(title, loc='left', fontsize=9)
        a_.text(36, 2, 'beyond training\nrange', fontsize=7.5, color=C['ink2'])
    ax[0].set_ylabel('Estimated area loss (%)'); ax[0].legend(loc='upper left')
    fig.tight_layout(); fig.savefig(f'{FIG}/fig3_parity.png'); plt.close(fig)


# ---------------------------------------------------------------- Fig 4 posterior example
def fig_posterior():
    import exp_fault as E, dt
    S_ = E.split_names()
    tr = E.get_windows(S_['train'], 'S3_full_sparse', 400.0, 1.0)
    te = E.get_windows(S_['ood'], 'S3_full_sparse', 400.0, 1.0)
    pg = dt.PGFE(comps=dt.COMP_ALL, env_kind='lin', prior='spike').fit(tr, E.cache_of(tr))
    two = [w for w in te if np.sum(w['delta'] > 0) == 2][:1] + [w for w in te if w['delta'].max() > 0.38][:1]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    cols = [C['pg'], C['gbm'], C['rf']]
    for a_, w, title in zip(ax, two, ['(a) Two degraded lines', '(b) Severity beyond training range']):
        p = pg.predict(w, E.cache_of(te))
        for i in range(3):
            a_.bar(dt.DG * 100 + (i - 1) * 0.7, p['marg'][i], width=0.7, color=cols[i], label=f'Line {i+1}', zorder=3)
            a_.axvline(w['delta'][i] * 100, color=cols[i], lw=1.2, ls='--', zorder=2)
        a_.set_xlabel('Area loss δ (%)'); a_.set_title(title + f"  [{w['name']}]", loc='left', fontsize=9); a_.set_xlim(-2, 55)
    ax[0].set_ylabel('Posterior probability'); ax[0].legend(loc='upper right')
    fig.text(0.5, -0.02, 'Dashed lines: true δ per line', ha='center', fontsize=7.5, color=C['ink2'])
    fig.tight_layout(); fig.savefig(f'{FIG}/fig4_posterior.png'); plt.close(fig)


# ---------------------------------------------------------------- Fig 5 learning curve + window length
def fig_lc():
    res = load('fault_lc.json'); win = load('fault_win.json')
    Ns = sorted({int(k.split('|')[0][1:]) for k in res})
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    T['lc'] = {}
    for m, mk in [('PG-FE', 'o'), ('DD-GBM', 's'), ('DD-RF', '^')]:
        mu, lo, hi = [], [], []
        for n in Ns:
            v = [res[k]['ood']['metrics']['MAE_damaged_lines'] * 100 for k in res if k.startswith(f'N{n}|') and k.endswith('|' + m)]
            vt = [res[k]['test']['metrics']['MAE_damaged_lines'] * 100 for k in res if k.startswith(f'N{n}|') and k.endswith('|' + m)]
            mu.append(np.mean(v)); lo.append(min(v)); hi.append(max(v))
            T['lc'][f'{m}|{n}'] = dict(test=float(np.mean(vt)), ood=float(np.mean(v)))
        ax[0].fill_between(Ns, lo, hi, color=MC[m], alpha=0.15, lw=0)
        ax[0].plot(Ns, mu, marker=mk, ms=5, color=MC[m], label=LAB[m])
    ax[0].set_xscale('log'); ax[0].set_xticks(Ns); ax[0].set_xticklabels(Ns); ax[0].set_yscale('log')
    ax[0].set_xlabel('Number of training simulations'); ax[0].set_ylabel('OOD MAE on degraded lines (%)')
    ax[0].set_title('(a) Data efficiency', loc='left', fontsize=9); ax[0].legend()
    Ws = [100, 200, 400]; x = np.arange(3); wd = 0.26
    T['win'] = {}
    for j, m in enumerate(['PG-FE', 'DD-GBM', 'DD-RF']):
        v = [win[f'W{W}|{m}']['test']['metrics']['MAE_damaged_lines'] * 100 for W in Ws]
        ax[1].bar(x + (j - 1) * wd, v, wd - 0.03, color=MC[m], label=LAB[m], zorder=3)
        T['win'][m] = v
    ax[1].set_xticks(x); ax[1].set_xticklabels([f'{W} s' for W in Ws]); ax[1].set_xlabel('Averaging window')
    ax[1].set_ylabel('Test MAE on degraded lines (%)'); ax[1].set_title('(b) Window length', loc='left', fontsize=9); ax[1].legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig5_learning_window.png'); plt.close(fig)


# ---------------------------------------------------------------- Fig 6 robustness: noise + model error
def fig_robust():
    res = load('fault_noise.json')
    keys = [('n0.5|o0', '0.5×'), ('n1|o0', '1×'), ('n2|o0', '2×'), ('n4|o0', '4×'), ('n1|o1', '1× + offset'), ('n1|o2', '1× + 2×offset'), ('n2|o2', '2× + 2×offset')]
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.1), gridspec_kw=dict(width_ratios=[1.25, 1]))
    x = np.arange(len(keys)); wd = 0.26; T['noise'] = {}
    for j, m in enumerate(['PG-FE', 'DD-GBM', 'DD-RF']):
        v = [res[f'{k}|{m}']['ood']['metrics']['MAE_damaged_lines'] * 100 for k, _ in keys]
        ax[0].bar(x + (j - 1) * wd, v, wd - 0.03, color=MC[m], label=LAB[m], zorder=3)
        T['noise'][m] = {lab: (res[f'{k}|{m}']['test']['metrics']['MAE_damaged_lines'] * 100,
                               res[f'{k}|{m}']['ood']['metrics']['MAE_damaged_lines'] * 100,
                               res[f'{k}|{m}']['test']['metrics']['false_alarm']) for k, lab in keys}
    ax[0].set_xticks(x); ax[0].set_xticklabels([l for _, l in keys], rotation=25, ha='right')
    ax[0].set_xlabel('Sensor noise level (× nominal) and bias'); ax[0].set_ylabel('OOD MAE on degraded lines (%)')
    ax[0].set_title('(a) Sensor noise and bias', loc='left', fontsize=9); ax[0].set_ylim(0, 11); ax[0].legend(loc='upper center', ncol=3, fontsize=7.5)
    if os.path.exists(f'{R}/fault_model.json'):
        mres = load('fault_model.json')
        sigs = sorted({float(k.split('|')[0][3:]) for k in mres})
        T['model'] = {}
        for m, lab, mk in [('PG-FE', 'No calibration', 'o'), ('PG-FE-cal', 'With commissioning calibration', 's')]:
            mu, lo, hi, fa = [], [], [], []
            for s in sigs:
                v = [mres[k]['test']['metrics']['MAE_damaged_lines'] * 100 for k in mres if k.startswith(f'sig{s}|') and k.endswith('|' + m)]
                f = [mres[k]['test']['metrics']['false_alarm'] for k in mres if k.startswith(f'sig{s}|') and k.endswith('|' + m)]
                vo = [mres[k]['ood']['metrics']['MAE_damaged_lines'] * 100 for k in mres if k.startswith(f'sig{s}|') and k.endswith('|' + m)]
                mu.append(np.mean(v)); lo.append(min(v)); hi.append(max(v)); fa.append(np.mean(f))
                T['model'][f'{m}|{s}'] = dict(test=float(np.mean(v)), test_max=float(max(v)), ood=float(np.mean(vo)), fa=float(np.mean(f)))
            col = C['grey'] if m == 'PG-FE' else C['pg']
            ax[1].fill_between(sigs, lo, hi, color=col, alpha=0.15, lw=0)
            ax[1].plot(sigs, mu, marker=mk, ms=5, color=col, label=lab)
        ax[1].set_xlabel('Anchor-radius / line-length error σ (m)'); ax[1].set_ylabel('Test MAE on degraded lines (%)')
        ax[1].set_title('(b) Mooring model-form error', loc='left', fontsize=9); ax[1].legend(loc='upper left')
    fig.tight_layout(); fig.savefig(f'{FIG}/fig6_robustness.png'); plt.close(fig)


# ---------------------------------------------------------------- Fig 7 virtual sensing + fatigue
def fig_vs():
    import exp_fault as E, dt, vsense as V
    out = load('vs_main.json')
    T['vs'] = {}
    for split in ['test', 'ood']:
        rows = out[split]
        for mdl in ['QS(est δ)', 'QS(healthy)', 'PG-VS', 'DD-VS', 'DD-MLP', 'DD-VS(no δ)']:
            v = [r[f'T{i}|{mdl}'] for r in rows for i in (1, 2, 3)]
            T['vs'][f'T|{mdl}|{split}'] = dict(nrmse=float(np.mean([x['nrmse'] for x in v])), r2=float(np.mean([x['r2'] for x in v])),
                                               del_mae=float(np.mean([abs(x['del_err']) for x in v]) * 100),
                                               del_p95=float(np.percentile([abs(x['del_err']) for x in v], 95) * 100))
        for mdl in ['PG-VS', 'DD-VS']:
            v = [r[f'twr|{mdl}'] for r in rows]
            T['vs'][f'twr|{mdl}|{split}'] = dict(nrmse=float(np.mean([x['nrmse'] for x in v])), r2=float(np.mean([x['r2'] for x in v])),
                                                 del_mae=float(np.mean([abs(x['del_err']) for x in v]) * 100),
                                                 del_p95=float(np.percentile([abs(x['del_err']) for x in v], 95) * 100))
    # example time series on a degraded OOD run
    S_ = E.split_names()
    tr = E.get_windows(S_['train'], 'S3_full_sparse', 400.0, 1.0)
    te = E.get_windows(S_['ood'], 'S3_full_sparse', 400.0, 1.0)
    w = [x for x in te if x['delta'].max() > 0.38][0]
    i = int(np.argmax(w['delta']))
    rr = [r for r in out['ood'] if r['name'] == w['name']][0]
    dh = np.array(rr['delta_hat'])
    pgvs = V.TensionVS('PG').fit(tr, [x['delta'] for x in tr]); ddn = V.TensionVS('DDnod').fit(tr, [x['delta'] for x in tr])
    S = w['S']; t = S['t'] - S['t'][0]
    y = w['truth'][f'T{i+1}'][::5][:len(t)] / 1e3
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.0), gridspec_kw=dict(width_ratios=[1.6, 1]))
    sl = slice(int(150 * 4), int(250 * 4))
    ax[0].plot(t[sl], y[sl], color=C['ink'], lw=1.6, label='MoorDyn (truth, not measured)')
    ax[0].plot(t[sl], pgvs.predict(S, dh, i)[sl] / 1e3, color=C['pg'], lw=1.4, label='PG-VS with twin δ̂')
    ax[0].plot(t[sl], ddn.predict(S, dh, i)[sl] / 1e3, color=C['gbm'], lw=1.2, label='DD-VS without δ̂')
    ax[0].set_xlabel('Time in window (s)'); ax[0].set_ylabel(f'Fairlead tension, line {i+1} (kN)')
    ax[0].set_title(f"(a) {w['name']}: line {i+1} δ = {w['delta'][i]*100:.0f}% (est. {dh[i]*100:.0f}%)", loc='left', fontsize=9)
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=7.5, ncol=3)
    mdls = ['QS(est δ)', 'DD-VS(no δ)', 'DD-VS', 'PG-VS', 'DD-MLP']
    names = ['Quasi-static', 'DD ridge, no δ̂', 'DD ridge + δ̂', 'PG-VS + δ̂', 'DD-MLP + δ̂']
    cols = [C['grey'], C['gbm'], C['rf'], C['pg'], C['mlp']]
    data = [[abs(r[f'T{k}|{m}']['del_err']) * 100 for r in out['ood'] for k in (1, 2, 3)] for m in mdls]
    bp = ax[1].boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color=C['ink'], lw=1.2))
    for b, c in zip(bp['boxes'], cols):
        b.set_facecolor(c); b.set_alpha(0.75); b.set_edgecolor(c)
    ax[1].set_yticks(range(1, len(mdls) + 1)); ax[1].set_yticklabels(names)
    ax[1].set_xlabel('|DEL error|, fairlead tension, OOD (%)'); ax[1].set_title('(b) Fatigue (DEL, m = 3)', loc='left', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig7_virtual_sensing.png'); plt.close(fig)


# ---------------------------------------------------------------- Fig 8 forecasting
def fig_fc():
    fc = load('fc_main.json')
    H = [2.0, 5.0, 10.0, 15.0, 20.0]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=True)
    T['fc'] = {}
    for a_, tg in zip(ax, ['pitch', 'surge', 'heave']):
        for lab, name, col, mk in [('AR', 'Motion history only (AR)', C['grey'], 'v'), ('DD', 'Ridge + raw buoy', C['gbm'], 's'),
                                   ('DD-MLP', 'MLP + raw buoy', C['mlp'], '^'), ('PG', 'Ridge + propagated wave (PG)', C['pg'], 'o')]:
            v = [fc[f'test|{tg}|{h}|{lab}']['skill_vs_persistence'] for h in H]
            vo = [fc[f'ood|{tg}|{h}|{lab}']['skill_vs_persistence'] for h in H]
            T['fc'][f'{tg}|{lab}'] = dict(test=v, ood=vo)
            a_.plot(H, v, marker=mk, ms=5, color=col, label=name)
        a_.set_title(f'({"abc"[["pitch","surge","heave"].index(tg)]}) {tg.capitalize()}', loc='left', fontsize=9)
        a_.set_xlabel('Forecast horizon (s)'); a_.set_xticks(H)
    ax[0].set_ylabel('Skill vs persistence (test)'); ax[0].set_ylim(0, 1.02); ax[0].legend(loc='lower left', fontsize=7.5)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig8_forecast.png'); plt.close(fig)


if __name__ == '__main__':
    import sys
    todo = sys.argv[1:] or ['campaign', 'main', 'posterior', 'lc', 'robust', 'vs', 'fc']
    if os.path.exists(f'{R}/numbers.json'):
        T.update(json.load(open(f'{R}/numbers.json')))
    for k in todo:
        {'campaign': fig_campaign, 'main': tab_main, 'posterior': fig_posterior, 'lc': fig_lc, 'robust': fig_robust, 'vs': fig_vs, 'fc': fig_fc}[k]()
        print('done', k, flush=True)
    json.dump(T, open(f'{R}/numbers.json', 'w'), indent=1, default=float)
