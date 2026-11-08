"""Every number quoted in the manuscript, computed from results/*.json -> results/numbers.tex and results/numbers_rev.json."""
from paths import ROOT, RTEST
import json, os, numpy as np
R = ROOT + '/results'
J = lambda f: json.load(open(f'{R}/{f}'))
M = {}


def put(name, val, fmt='{:.2f}'):
    M[name] = fmt.format(val) if not isinstance(val, str) else val


E1 = J('rev_e1_main.json'); E2 = J('rev_e2_uq.json'); E1b = J('rev_e1b_seq.json'); E3 = J('rev_e3_mismatch.json')
E4 = J('rev_e4_install.json'); E6 = J('rev_e6_causal.json'); E7 = J('rev_e7_vs.json'); E8 = J('rev_e8_fc.json'); E9 = J('rev_e9_fidelity.json')
E5w = J('rev_e5_win.json'); E5l = J('rev_e5_lc.json'); E5n = J('rev_e5_noise.json')
D0 = json.load(open(ROOT + '/design.json')); DR = json.load(open(ROOT + '/design_rev.json'))
put('NsimOrig', len(D0), '{:d}'); put('NsimRev', len(DR), '{:d}'); put('NsimTotal', len(D0) + len(DR), '{:d}')
put('NtrainMixed', sum(c['split'] == 'train' for c in D0), '{:d}')
put('NtrainHealthy', sum(c['split'] == 'train' and max(c['delta']) == 0 for c in D0) + sum(c['split'] == 'htrain' for c in DR), '{:d}')

POOLS = {'ID': 'ID', 'Sev': 'OOD-severity', 'Two': 'OOD-two-line', 'Ext': 'OOD-extreme-sea', 'Mis': 'OOD-misalign'}
METH = {'PG': 'PG-FE (healthy-trained)', 'PGlab': 'PG-FE (label-trained)', 'PGw': 'PG-FE + weight nuisance', 'Vlab': 'v1 PG-FE, pose-learned, label-trained',
        'Vh': 'v1 PG-FE, pose-learned, healthy-trained', 'Orc': 'PG-FE oracle env load', 'WLS': 'WLS (physics)', 'Lone': 'L1-WLS (physics)',
        'GBM': 'DD-GBM', 'RF': 'DD-RF', 'MLP': 'DD-MLP'}
SS = {'A': 'S1_GNSS+SCADA', 'B': 'S2_GNSS+MRU+SCADA', 'C': 'S3_full_sparse'}
for sk, s in SS.items():
    for mk, m in METH.items():
        for pk, p in POOLS.items():
            key = f'{s}|{m}|{p}'
            if key not in E1:
                continue
            r = E1[key]; sv, dd = r['sev'], r['det']
            put(f'{mk}{sk}{pk}dam', 100 * sv['MAE_dam']); put(f'{mk}{sk}{pk}all', 100 * sv['MAE_all'])
            if 'MAE_dam_ci' in sv:
                put(f'{mk}{sk}{pk}lo', 100 * sv['MAE_dam_ci'][0]); put(f'{mk}{sk}{pk}hi', 100 * sv['MAE_dam_ci'][1])
            put(f'{mk}{sk}{pk}fak', dd['intact_fa_k'], '{:d}'); put(f'{mk}{sk}{pk}fan', dd['intact_n'], '{:d}')
            put(f'{mk}{sk}{pk}prec', dd['line_precision']); put(f'{mk}{sk}{pk}rec', dd['line_recall']); put(f'{mk}{sk}{pk}exact', dd['exact_set_acc'])
            put(f'{mk}{sk}{pk}runrec', dd['run_recall'])
            if dd.get('exact_set_acc_2line') == dd.get('exact_set_acc_2line'):
                put(f'{mk}{sk}{pk}exacttwo', dd['exact_set_acc_2line'] if dd['exact_set_acc_2line'] is not None else float('nan'))
            if 'cov' in r:
                put(f'{mk}{sk}{pk}covninety', 100 * r['cov']['0.9'], '{:.0f}')
for sk, s in SS.items():
    for mk, m in METH.items():
        ks = [f'{s}|{m}|{p}' for p in POOLS.values() if f'{s}|{m}|{p}' in E1]
        if ks:
            M[f'{mk}{sk}FAtot'] = f"{sum(E1[k]['det']['intact_fa_k'] for k in ks)}/{sum(E1[k]['det']['intact_n'] for k in ks)}"
# intact false-alarm totals (S3, PG-FE) across all evaluation pools
fk = sum(E1[f'S3_full_sparse|PG-FE (healthy-trained)|{p}']['det']['intact_fa_k'] for p in POOLS.values())
fn = sum(E1[f'S3_full_sparse|PG-FE (healthy-trained)|{p}']['det']['intact_n'] for p in POOLS.values())
put('FAintactK', fk, '{:d}'); put('FAintactN', fn, '{:d}')
put('FAupper', 100 * (1 - 0.05 ** (1 / fn)) if fk == 0 else float('nan'), '{:.1f}\\%')
gk = sum(E1[f'S3_full_sparse|DD-GBM|{p}']['det']['intact_fa_k'] for p in POOLS.values())
put('FAgbmK', gk, '{:d}')
put('LambdaSel', E1['L1_lambda'], '{}')
for kind in ['DD-GBM', 'DD-RF', 'DD-MLP']:
    M[f'HP{kind.replace("DD-", "")}'] = '; '.join(f'{k.split("__")[-1].replace("_", " ")} = {v}' for k, v in E1[f'S3_full_sparse|{kind}|hyperparameters'].items())
env = E1['env_err_over_sigma']
put('EnvErrIDmax', max(env['ID']), '{:.2f}'); put('EnvErrMisMax', max(env['OOD-misalign']), '{:.2f}')

# sequential / EKF
for pk, p in POOLS.items():
    put(f'EKF{pk}', 100 * E1b[f'EKF (physics, 4x100 s)|{p}']['sev']['MAE_dam'])
    put(f'Seq{pk}', 100 * E1b[f'PG-FE sequential (4x100 s)|{p}']['sev']['MAE_dam'])
    put(f'EKF{pk}all', 100 * E1b[f'EKF (physics, 4x100 s)|{p}']['sev']['MAE_all'])
    put(f'Seq{pk}all', 100 * E1b[f'PG-FE sequential (4x100 s)|{p}']['sev']['MAE_all'])
    put(f'EKF{pk}exact', E1b[f'EKF (physics, 4x100 s)|{p}']['det']['exact_set_acc'])

# identifiability / prior / stratified coverage
I = [x for x in E2['ident'] if x['at'] == 'intact']
put('CondSixMed', np.median([x['cond6'] for x in I]), '{:.0f}'); put('CondSixMax', np.max([x['cond6'] for x in I]), '{:.0f}')
put('CondHMed', np.median([x['condH'] for x in I]), '{:.0f}'); put('CondHMax', np.max([x['condH'] for x in I]), '{:.1e}')
put('CRBmed', 100 * np.median([np.mean(x['crb']) for x in I]))
S = E2['stratified']
for pk, p in POOLS.items():
    ks = [k for k in S if k.startswith(p + '|') and float(k.split('|')[1].split('-')[0]) >= 0.05]
    if ks:
        nn = sum(S[k]['n'] for k in ks)
        put(f'CovDam{pk}', 100 * sum(S[k]['n'] * S[k]['cov']['0.9'] for k in ks) / nn, '{:.0f}'); put(f'CovDam{pk}n', nn, '{:d}')
p0s = ['0.5', '0.75', '0.9', '0.97', '0.99']
v = [100 * E2[f'diag|het=False|p0={p}|ID']['sev']['MAE_dam'] for p in p0s]; put('PriorIDmin', min(v)); put('PriorIDmax', max(v))
v = [100 * E2[f'diag|het=False|p0={p}|OOD-two-line']['sev']['MAE_dam'] for p in p0s]; put('PriorTwoMin', min(v)); put('PriorTwoMax', max(v))
put('FullCovTwo', 100 * E2['full|het=False|p0=0.9|OOD-two-line']['sev']['MAE_dam']); put('FullCovTwoCov', 100 * E2['full|het=False|p0=0.9|OOD-two-line']['cov']['0.9'], '{:.0f}')
put('HetTwo', 100 * E2['full|het=True|p0=0.9|OOD-two-line']['sev']['MAE_dam'])

# mismatch
nom = E3['qs_vs_md_nominal']
put('QSmaxSigma', max(nom['over_sigma']), '{:.2f}'); put('QSnN', nom['n'], '{:d}')
def grp(g, pk='pred', ak='alarm'):
    rows = E3[g]; T = np.array([r['delta'] for r in rows]); P = np.array([r[pk] for r in rows]); A = np.array([r[ak] for r in rows])
    intact = (T == 0).all(1)
    return 100 * np.abs(P - T)[T > 0.05].mean(), int(A[intact].any(1).sum()), int(intact.sum())
benign = ['B1-EA-0.9', 'B1-EA-1.1', 'B2-mass-0.97', 'B2-mass-1.03', 'B3-drag', 'B3-friction', 'B4-fair-uniform', 'B4-fair-random']
vals = [grp(g) for g in benign]
put('MMbenignMin', min(v[0] for v in vals)); put('MMbenignMax', max(v[0] for v in vals)); put('MMbenignFA', sum(v[1] for v in vals), '{:d}'); put('MMbenignN', sum(v[2] for v in vals), '{:d}')
for g, k in [('B5-ptfm-mass-0.99', 'Pm'), ('B5-ptfm-mass-0.995', 'Ph'), ('B5-ptfm-mass-1.005', 'Pp'), ('B5-ptfm-mass-1.01', 'Pq')]:
    a = grp(g); b = grp(g, 'pred_mass', 'alarm_mass')
    put(f'MM{k}', a[0]); put(f'MM{k}fa', a[1], '{:d}'); put(f'MM{k}w', b[0]); put(f'MM{k}wfa', b[1], '{:d}')
rowsL = E3['B6-localized']
for r in rowsL:
    L = r['loc']
put('LocFairDetected', sum(r['alarm'][r['loc']['line']] for r in rowsL if r['loc']['where'] == 'fairlead'), '{:d}')
put('LocTouchDetected', sum(r['alarm'][r['loc']['line']] for r in rowsL if r['loc']['where'] == 'touchdown'), '{:d}')
eqB2 = np.mean([np.abs(r['eq_err_over_sigma']) for r in E3['B2-mass-0.97'] if 'eq_err_over_sigma' in r], 0)
put('BtwoFz', eqB2[2]); eqB4 = np.mean([np.abs(r['eq_err_over_sigma']) for r in E3['B4-fair-random'] if 'eq_err_over_sigma' in r], 0)
put('BfourMax', eqB4.max())
core = json.load(open(f'{R}/rev_core_tension.json')) if os.path.exists(f'{R}/rev_core_tension.json') else None
if core:
    put('CoreN', core['n'], '{:d}'); put('CoreMean', core['mean_abs_pct']); put('CoreMax', core['max_abs_pct'])

# installations
for I_, rec in E4.items():
    for key, k2 in [('geometry|nocal', 'nocal'), ('geometry|cal', 'cal'), ('geometry+mass|cal', 'calw')]:
        T = np.array(rec[key]['true']); P = np.array(rec[key]['pred'])
        put(f'Inst{I_}{k2}', 100 * np.abs(P - T)[T > 0.05].mean())
    put(f'Inst{I_}geoerr', np.max(np.abs(np.array(rec['cal_geometry']['x'][:3]) - np.array(rec['actual_eff_dr']))))
    put(f'Inst{I_}geowerr', np.max(np.abs(np.array(rec['cal_geometry+mass']['x'][:3]) - np.array(rec['actual_eff_dr']))))
    put(f'Inst{I_}sd', np.max(rec['cal_geometry']['sd']))
    put(f'Inst{I_}dW', rec['cal_geometry+mass']['x'][3] / 1e3, '{:.0f}')
nc = [np.abs(np.array(E4[k]['geometry|nocal']['pred']) - np.array(E4[k]['geometry|nocal']['true']))[np.array(E4[k]['geometry|nocal']['true']) > 0.05].mean() for k in E4]
ca = [np.abs(np.array(E4[k]['geometry|cal']['pred']) - np.array(E4[k]['geometry|cal']['true']))[np.array(E4[k]['geometry|cal']['true']) > 0.05].mean() for k in E4]
put('InstNocalMin', 100 * min(nc)); put('InstNocalMax', 100 * max(nc)); put('InstCalMin', 100 * min(ca[:4])); put('InstCalMax', 100 * max(ca[:4]))
put('InstFAnocal', sum(np.array(E4[k]['geometry|nocal']['alarm'])[(np.array(E4[k]['geometry|nocal']['true']) == 0).all(1)].any(1).sum() for k in E4), '{:d}')
put('InstFAcal', sum(np.array(E4[k]['geometry|cal']['alarm'])[(np.array(E4[k]['geometry|cal']['true']) == 0).all(1)].any(1).sum() for k in E4), '{:d}')
put('InstGeoErrMaxIfour', max(np.max(np.abs(np.array(E4[k]['cal_geometry']['x'][:3]) - np.array(E4[k]['actual_eff_dr']))) for k in ['I1', 'I2', 'I3', 'I4']))
put('InstGeoErrMinIfour', min(np.max(np.abs(np.array(E4[k]['cal_geometry']['x'][:3]) - np.array(E4[k]['actual_eff_dr']))) for k in ['I1', 'I2', 'I3', 'I4']))
put('PtfmMassIfive', 0.005 * 1.3473e7 * 9.80665 / 1e3, '{:.0f}')

# robustness
def rget(d, key, f='MAE_dam'): return 100 * d[key]['sev'][f]
put('WinHundred', rget(E5w, 'win|W100|PG-FE|ID')); put('WinTwoHundred', rget(E5w, 'win|W200|PG-FE|ID')); put('WinFourHundred', rget(E5w, 'win|W400|PG-FE|ID'))
put('WinHundredGBM', rget(E5w, 'win|W100|DD-GBM|ID')); put('WinFourHundredGBM', rget(E5w, 'win|W400|DD-GBM|ID'))
for n_ in ['10', '20', '30']:
    v = [rget(E5l, f'lc|N{n_}|r{r}|ID') for r in range(3)]; put(f'LcPG{n_}', np.mean(v)); put(f'LcPG{n_}max', np.max(v))
    v = [rget(E5l, f'lc|N{n_}|r{r}|DD-GBM|ID') for r in range(3)]; put(f'LcGBM{n_}', np.mean(v))
    v = [rget(E5l, f'lc|N{n_}|r{r}|OOD-two-line') for r in range(3)]; put(f'LcPGtwo{n_}', np.mean(v))
fa10 = [E5l[f'lc|N10|r{r}|ID']['det']['intact_fa_k'] for r in range(3)]; put('LcFAtenMax', max(fa10), '{:d}')
nk = [k for k in E5n if '|PG-FE|ID' in k]; v = [rget(E5n, k) for k in nk]; put('NoisePGmin', min(v)); put('NoisePGmax', max(v))
v = [rget(E5n, k.replace('|PG-FE|ID', '|PG-FE|OOD-two-line')) for k in nk]; put('NoisePGtwoMax', max(v))
v = [rget(E5n, k.replace('PG-FE', 'DD-GBM')) for k in nk]; put('NoiseGBMmin', min(v)); put('NoiseGBMmax', max(v))
put('NoisePGfa', sum(E5n[k]['det']['intact_fa_k'] for k in nk), '{:d}')

# online detection
k1 = E6['k1']; put('OnlineN', k1['n_seq'], '{:d}'); put('OnlinePre', k1['pre_onset_windows'], '{:d}'); put('OnlinePreFA', k1['alarms_before_onset'], '{:d}')
put('OnlineMed', k1['median'], '{:.0f}'); put('OnlineCorrect', 100 * k1['correct_lines_at_first_alarm'], '{:.0f}')
put('OnlineCorrectFour', 100 * E6['k4']['correct_lines_at_first_alarm'], '{:.0f}'); put('OnlineFourP', E6['k4']['p90'], '{:.0f}')
put('CompTable', E6['compute_s']['table'], '{:.2f}'); put('CompPost', 1000 * E6['compute_s']['posterior'], '{:.0f}')

# virtual sensing
def vs(pool, mode, model, key):
    return np.mean([abs(r[key]) for r in E7[pool] if r['mode'] == mode and r['model'] == model])
for pk, p in [('ID', 'ID'), ('Sev', 'OOD-severity'), ('Two', 'OOD-two-line'), ('Ext', 'OOD-extreme-sea')]:
    for mk, m in [('PG', 'PG-VS'), ('Rid', 'DD ridge + d'), ('MLP', 'DD-MLP + d'), ('No', 'DD ridge, no d')]:
        put(f'VS{mk}{pk}', 100 * vs(p, 'causal', m, 'del_T_err'), '{:.1f}'); put(f'VS{mk}{pk}r', 100 * vs(p, 'retrospective', m, 'del_T_err'), '{:.1f}')
        put(f'VS{mk}{pk}n', vs(p, 'causal', m, 'nrmse')); put(f'VS{mk}{pk}S', 100 * vs(p, 'causal', m, 'del_S_err'), '{:.1f}')

# forecasting
FC0 = json.load(open(f'{R}/fc_main.json'))
for tg in ['pitch', 'surge', 'heave']:
    for lab in ['AR', 'DD', 'DD-MLP', 'PG']:
        put(f'FC{tg}{lab.replace("-", "")}', FC0[f'test|{tg}|10.0|{lab}']['skill_vs_persistence'])
for N in ['3', '10']:
    for m in ['DD', 'PG']:
        put(f'FCscarce{m}{N}', np.mean([E8[k]['skill_vs_persistence'] for k in E8 if k.startswith(f'N{N}|') and k.endswith(f'|pitch|10.0|{m}')]))

# fidelity
U = [v['meta']['U'] for v in E9.values()]; H = [v['meta']['Hs'] for v in E9.values()]
put('FFUmin', min(U), '{:.1f}'); put('FFUmax', max(U), '{:.1f}'); put('FFHsMin', min(H), '{:.1f}'); put('FFHsMax', max(H), '{:.1f}')
mot = ['PtfmSurge_[m]', 'PtfmSway_[m]', 'PtfmYaw_[deg]', 'FAIRTEN1_[N]', 'FAIRTEN2_[N]', 'FAIRTEN3_[N]']
put('FFmaxNRMSEmotTen', max(v['channels'][c]['nrmse'] for v in E9.values() for c in mot), '{:.2f}')
put('FFmaxPitch', max(v['channels']['PtfmPitch_[deg]']['nrmse'] for v in E9.values()), '{:.2f}')
put('FFmaxHeave', max(v['channels']['PtfmHeave_[m]']['nrmse'] for v in E9.values()), '{:.2f}')
put('FFmaxTwr', max(v['channels']['TwrBsMyt_[kN-m]']['nrmse'] for v in E9.values()), '{:.2f}')
put('FFmaxMeanTen', 100 * max(abs(v['channels'][f'FAIRTEN{i}_[N]']['mean_diff_rel']) for v in E9.values() for i in (1, 2, 3)), '{:.2f}\\%')
put('FFmaxMeanSurge', 100 * max(abs(v['channels']['PtfmSurge_[m]']['mean_diff_rel']) for v in E9.values()), '{:.2f}\\%')

W_ = {'0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'}
import re as _re
M = {_re.sub(r'\d+', lambda m: ''.join(W_[c] for c in m.group(0)), k): v for k, v in M.items()}
out = '%% generated by build_numbers.py -- do not edit\n' + '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(M.items()) if k.isalpha())
bad = [k for k in M if not k.isalpha()]
open(ROOT + '/results/numbers.tex', 'w').write(out + '\n')
json.dump(M, open(f'{R}/numbers_rev.json', 'w'), indent=1)
print(len(M), 'macros; non-alpha names skipped:', bad)
