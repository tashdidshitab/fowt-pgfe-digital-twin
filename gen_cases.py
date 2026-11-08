"""Generate OpenFAST + TurbSim cases for the FOWT digital-twin campaign.
OC4-DeepCwind semisubmersible + NREL 5-MW, MoorDyn mooring with per-line
corrosion/wear degradation (fraction of load-bearing area lost, delta)."""
from paths import ROOT, RTEST
import os, re, json, shutil, numpy as np, sys

BASE = RTEST + '/glue-codes/openfast'
TEMPLATE = ROOT + '/openfast_templates/fast_ua'   # reduced-fidelity settings (paper Sec. 2.1)
RUNS = ROOT + '/runs'
CASE = '5MW_OC4Semi_WSt_WavesWN'   # OpenFAST input-file root name
TMAX = 520.0          # 120 s transient + 400 s analysis window
D0, M0, EA0 = 0.0766, 113.35, 7.536e8


def set_val(txt, key, val):
    pat = re.compile(r'^(\s*)(\S+)(\s+' + re.escape(key) + r'\b)', re.M)
    new, n = pat.subn(lambda m: m.group(1) + str(val) + m.group(3), txt, count=1)
    assert n == 1, key
    return new


def set_outlist(txt, channels, start_pat=r'OutList', end_pat=r'^END'):
    lines = txt.split('\n')
    i0 = next(i for i, l in enumerate(lines) if re.search(start_pat, l))
    i1 = next(i for i in range(i0 + 1, len(lines)) if re.match(end_pat, lines[i]))
    return '\n'.join(lines[:i0 + 1] + [f'"{c}"' for c in channels] + lines[i1:])


ED_CH = ['PtfmSurge', 'PtfmSway', 'PtfmHeave', 'PtfmRoll', 'PtfmPitch', 'PtfmYaw',
         'PtfmTAxt', 'PtfmTAyt', 'PtfmTAzt', 'PtfmRVxt', 'PtfmRVyt', 'PtfmRVzt',
         'NcIMUTAxs', 'NcIMUTAys', 'RotSpeed', 'BldPitch1', 'TTDspFA',
         'TwrBsFxt', 'TwrBsMxt', 'TwrBsMyt', 'TwrBsMzt', 'YawBrMyp']


def make_case(name, U, TI, Hs, Tp, wdir, delta, seed):
    d = os.path.join(RUNS, name)
    if os.path.exists(os.path.join(d, CASE + '.outb')):
        return d
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(TEMPLATE):
        if f.endswith('.dat') or f.endswith('.fst'):
            shutil.copy(os.path.join(TEMPLATE, f), d)
    abs_base = BASE + '/5MW_Baseline'
    for f in os.listdir(d):
        p = os.path.join(d, f)
        t = open(p).read().replace('"../../5MW_Baseline', '"' + abs_base)
        open(p, 'w').write(t)
    # --- main fst
    p = os.path.join(d, CASE + '.fst'); t = open(p).read()
    t = set_val(t, 'TMax', TMAX)
    t = set_val(t, 'InflowFile', '"InflowWind.dat"')
    open(p, 'w').write(t)
    # --- InflowWind (TurbSim full field)
    t = open(abs_base + '/NRELOffshrBsline5MW_InflowWind_Steady8mps.dat').read()
    t = set_val(t, 'WindType', 3)
    t = set_val(t, 'FileName_BTS', '"wind.bts"')
    t = set_val(t, 'TowerFile', 'True')
    t = set_val(t, 'PropagationDir', f'{wdir:.2f}')   # wind-rotor misalignment (waves fixed at 0 deg: QTF data only at 0 deg)
    open(os.path.join(d, 'InflowWind.dat'), 'w').write(t)
    # --- TurbSim
    ts = open(ROOT + '/openfast_templates/TurbSim_template.inp').read()
    ts = set_val(ts, 'RandSeed1', seed)
    ts = set_val(ts, 'WrADFF', 'True'); ts = set_val(ts, 'WrADTWR', 'True'); ts = set_val(ts, 'WrBLFF', 'False')
    ts = set_val(ts, 'NumGrid_Z', 13); ts = set_val(ts, 'NumGrid_Y', 13)
    ts = set_val(ts, 'TimeStep', 0.05); ts = set_val(ts, 'AnalysisTime', TMAX + 10)
    ts = set_val(ts, 'GridHeight', 150); ts = set_val(ts, 'GridWidth', 150)
    ts = set_val(ts, 'IECturbc', f'{TI:.2f}')
    ts = set_val(ts, 'URef', f'{U:.3f}')
    ts = set_val(ts, 'PLExp', 0.14)
    open(os.path.join(d, 'wind.inp'), 'w').write(ts)
    # --- SeaState: JONSWAP irregular waves, wave elev at platform and an up-wave buoy
    p = os.path.join(d, 'SeaState.dat'); t = open(p).read()
    t = set_val(t, 'WaveMod', 2)
    t = set_val(t, 'WaveTMax', 1000)
    t = set_val(t, 'X_HalfWidth', 130); t = set_val(t, 'NX', 13)
    t = set_val(t, 'WaveDT', 0.25)
    t = set_val(t, 'WaveHs', f'{Hs:.3f}'); t = set_val(t, 'WaveTp', f'{Tp:.3f}')
    t = set_val(t, 'WaveDir', 0)
    t = set_val(t, 'WvLowCOff', 0.15); t = set_val(t, 'WvHiCOff', 2.5)
    t = re.sub(r'^\s*\S+(\s+WaveSeed\(1\))', f'   {seed + 7}\\1', t, flags=re.M)
    bx, by = -120.0, 0.0   # up-wave buoy 120 m ahead of platform
    t = re.sub(r'^.*WaveElevxi', f'  0, {bx:.2f}    WaveElevxi', t, flags=re.M)
    t = re.sub(r'^.*WaveElevyi', f'  0, {by:.2f}    WaveElevyi', t, flags=re.M)
    t = set_outlist(t, ['Wave1Elev', 'Wave2Elev'], r'OUTPUT CHANNELS')
    open(p, 'w').write(t)
    # --- MoorDyn: one line type per line
    p = os.path.join(d, 'NRELOffshrBsline5MW_OC4DeepCwindSemi_MoorDyn.dat'); t = open(p).read()
    lt = ''
    for i, dl in enumerate(delta):
        s = 1 - dl
        lt += f'main{i+1}   {D0*np.sqrt(s):.5f}   {M0*s:.4f}   {EA0*s:.5E}   -1.0   0   2.0   0.8   0.4   0.25\n'
    t = re.sub(r'^main\s+0\.0766.*\n', lt, t, flags=re.M)
    for i in range(3):
        t = re.sub(rf'^({i+1}\s+)main(\s+)', rf'\1main{i+1}\2', t, flags=re.M)
    open(p, 'w').write(t)
    # --- outputs
    p = os.path.join(d, 'NRELOffshrBsline5MW_OC4DeepCwindSemi_ElastoDyn.dat'); t = open(p).read()
    t = set_outlist(t, ED_CH, r'^\s*OutList'); open(p, 'w').write(t)
    p = os.path.join(d, 'NRELOffshrBsline5MW_OC3Hywind_AeroDyn.dat'); t = open(p).read()
    t = set_outlist(t, ['RtFldFxh', 'RtFldCt', 'RtVAvgxh'], r'^\s*OutList\s'); open(p, 'w').write(t)
    p = os.path.join(d, 'NRELOffshrBsline5MW_OC4DeepCwindSemi_HydroDyn.dat'); t = open(p).read()
    t = set_outlist(t, ['HydroFxi', 'HydroFyi', 'HydroMzi'], r'OUTPUT CHANNELS'); open(p, 'w').write(t)
    json.dump(dict(name=name, U=U, TI=TI, Hs=Hs, Tp=Tp, wdir=wdir, delta=list(delta), seed=seed),
              open(os.path.join(d, 'meta.json'), 'w'))
    return d


def design(seed=2026):
    """Campaign: Latin-hypercube environmental sampling + damage states.
    Splits: train/val (in-distribution), test_id (held-out), test_ood
    (unseen severities, 2-line damage, extreme sea states)."""
    rng = np.random.default_rng(seed)
    cases = []

    def env(n, Ulo=4.5, Uhi=22.0):
        u = (rng.permutation(n) + rng.random(n)) / n
        U = Ulo + (Uhi - Ulo) * u
        Hs = np.clip(0.9 + 0.17 * U + 0.012 * U**2 + rng.normal(0, 0.35, n), 0.8, 7.5)   # site-like Hs(U)
        Tp = np.clip(4.2 + 0.30 * U + 0.7 * np.sqrt(Hs) + rng.normal(0, 0.8, n), 6.0, 15.0)
        TI = np.clip(100 * 0.14 * (0.75 * U + 5.6) / U * 0.75 + rng.normal(0, 1.0, n), 5, 25)  # ~IEC class C NTM
        wdir = rng.uniform(-10, 10, n)
        return U, TI, Hs, Tp, wdir

    def damage(n, lo=0.0, hi=0.35, p_healthy=0.25):
        D = np.zeros((n, 3))
        for k in range(n):
            if rng.random() < p_healthy:
                continue
            D[k, rng.integers(3)] = rng.uniform(max(lo, 0.03), hi)
        return D

    def add(split, n, Ulo=4.5, Uhi=22.0, D=None):
        U, TI, Hs, Tp, wd = env(n, Ulo, Uhi)
        if D is None:
            D = damage(n)
        for k in range(n):
            cases.append(dict(name=f'{split}_{len([c for c in cases if c["split"]==split]):03d}', split=split,
                              U=float(U[k]), TI=float(TI[k]), Hs=float(Hs[k]), Tp=float(Tp[k]), wdir=float(wd[k]),
                              delta=[float(x) for x in D[k]], seed=int(rng.integers(1, 2**31 - 1))))

    add('train', 110)
    add('test', 30)
    # OOD: severities beyond training range, two damaged lines, extreme sea states
    D = np.zeros((24, 3))
    for k in range(8):
        D[k, rng.integers(3)] = rng.uniform(0.38, 0.5)
    for k in range(8, 16):
        a, b = rng.choice(3, 2, replace=False); D[k, a] = rng.uniform(0.1, 0.3); D[k, b] = rng.uniform(0.1, 0.3)
    for k in range(16, 24):
        if rng.random() > 0.25:
            D[k, rng.integers(3)] = rng.uniform(0.05, 0.35)
    U, TI, Hs, Tp, wd = env(24)
    U[16:], Hs[16:] = rng.uniform(20, 24, 8), rng.uniform(6.5, 8.5, 8)
    Tp[16:] = np.clip(9.0 + 0.6 * Hs[16:], 12, 15)
    for k in range(24):
        cases.append(dict(name=f'ood_{k:03d}', split='ood', U=float(U[k]), TI=float(TI[k]), Hs=float(Hs[k]),
                          Tp=float(Tp[k]), wdir=float(wd[k]), delta=[float(x) for x in D[k]],
                          seed=int(rng.integers(1, 2**31 - 1))))
    return cases


if __name__ == '__main__':
    cases = design()
    os.makedirs(RUNS, exist_ok=True)
    json.dump(cases, open(ROOT + '/design.json', 'w'), indent=1)
    which = sys.argv[1:] or [c['name'] for c in cases]
    for c in cases:
        if c['name'] in which:
            make_case(c['name'], c['U'], c['TI'], c['Hs'], c['Tp'], c['wdir'], c['delta'], c['seed'])
    print(len(cases), 'cases;', {s: sum(c['split'] == s for c in cases) for s in ['train', 'test', 'ood']})
