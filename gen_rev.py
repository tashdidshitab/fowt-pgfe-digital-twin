"""Revision campaign: cases in which the simulated 'reality' differs from the twin.
Builds on gen_cases.make_case and then rewrites MoorDyn / ElastoDyn inputs.

Every case stores in meta.json the full physical description used by MoorDyn
(line segments, anchor/fairlead coordinates, property scales, friction, mass).
"""
from paths import ROOT, RTEST
import os, json, re, numpy as np, sys
import gen_cases as G

RUNS = G.RUNS
D0, M0, EA0, L0 = 0.0766, 113.35, 7.536e8, 835.35
ANCH = np.array([[418.8, 725.383, -200.0], [-837.6, 0.0, -200.0], [418.8, -725.383, -200.0]])
FAIR = np.array([[20.434, 35.393, -14.0], [-40.868, 0.0, -14.0], [20.434, -35.393, -14.0]])
RHO = 1025.0


def moordyn_text(segs, anch, fair, ea_s=1.0, m_s=1.0, cd_s=1.0, ca_s=1.0, mu=None, nseg_per_m=20 / 835.35):
    """segs[i] = list of (length, delta) from anchor to fairlead for line i."""
    types, pts, lines = [], [], []
    tid = {}

    def ltype(dl):
        key = round(dl, 5)
        if key not in tid:
            s = 1 - dl
            # nominal diameter, mass and EA scale with the remaining section; property
            # scales ea_s / m_s model mis-specified nominal properties of the real chain
            tid[key] = f'lt{len(tid)+1}'
            types.append(f'{tid[key]}   {D0*np.sqrt(s):.5f}   {M0*s*m_s:.4f}   {EA0*s*ea_s:.5E}   -1.0   0   {2.0*cd_s:.3f}   {0.8*ca_s:.3f}   0.4   0.25')
        return tid[key]
    # point numbering: 1-3 anchors, 4-6 fairleads, 7+ free points
    for i in range(3):
        pts.append(f'{i+1}   Fixed   {anch[i,0]:.4f}   {anch[i,1]:.4f}   {anch[i,2]:.3f}   0   0   0   0')
    for i in range(3):
        pts.append(f'{i+4}   Vessel   {fair[i,0]:.4f}   {fair[i,1]:.4f}   {fair[i,2]:.4f}   0   0   0   0')
    fair_lines, other_lines = [], []
    nxt = 7
    for i in range(3):
        L = sum(s[0] for s in segs[i])
        a, f = anch[i], fair[i]
        prev = i + 1; s_acc = 0.0
        for k, (ln, dl) in enumerate(segs[i]):
            last = k == len(segs[i]) - 1
            if last:
                end = i + 4
            else:
                s_acc += ln
                # rough initial position: laid on seabed up to ~ (L-600) m, then straight to fairlead
                frac = s_acc / L
                p = a + frac * (f - a)
                z = -200.0 if s_acc < L - 600 else a[2] + (f[2] - a[2]) * (s_acc - (L - 600)) / 600
                pts.append(f'{nxt}   Free   {p[0]:.3f}   {p[1]:.3f}   {z:.3f}   0   0   0   0')
                end = nxt; nxt += 1
            rec = (ltype(dl), prev, end, ln, max(4, int(round(ln * nseg_per_m))))
            (fair_lines if last else other_lines).append(rec)
            prev = end
    lines_all = fair_lines + other_lines      # line IDs 1-3 = fairlead segments (FairTen1-3)
    for j, (t, a_, b_, ln, ns) in enumerate(lines_all):
        lines.append(f'{j+1}   {t}   {a_}   {b_}   {ln:.3f}   {ns}   -')
    opts = ['0.002    dtM', '3.0e6    kbot', '3.0e5    cbot', '2.0      dtIC', '120.0    TmaxIC', '4.0      CdScaleIC', '0.01     threshIC']
    if mu is not None:
        opts += [f'{mu[0]:.3f}    mu_kT', f'{mu[1]:.3f}    mu_kA', '1.0    mc', '200    cv']
    return '\n'.join([
        '--------------------- MoorDyn Input File ------------------------------------',
        'Mooring system for OC4-DeepCwind Semi (revision campaign)',
        'FALSE    Echo      - echo the input file data (flag)',
        '----------------------- LINE TYPES ------------------------------------------',
        'Name     Diam      MassDen       EA    BA/-zeta    EI    Cd      Ca     CdAx   CaAx',
        '(-)       (m)      (kg/m)        (N)    (N-s/-)    (-)   (-)     (-)    (-)    (-)',
        *types,
        '---------------------- POINTS --------------------------------',
        'ID     Attachment   X          Y         Z      M      V       CdA   CA',
        '(-)    (-)         (m)        (m)       (m)    (kg)   (m^3)   (m^2)  (-)',
        *pts,
        '---------------------- LINES --------------------------------------',
        'ID      LineType   AttachA   AttachB  UnstrLen  NumSegs   Outputs',
        '(-)       (-)       (-)       (-)         (m)      (-)         (-)',
        *lines,
        '---------------------- SOLVER OPTIONS ---------------------------------------',
        *opts,
        '------------------------ OUTPUTS --------------------------------------------',
        'FairTen1', 'FairTen2', 'FairTen3', 'AnchTen1', 'AnchTen2', 'AnchTen3',
        'Point4FX', 'Point4FY', 'Point4FZ', 'Point5FX', 'Point5FY', 'Point5FZ', 'Point6FX', 'Point6FY', 'Point6FZ', 'END',
        '------------------------- need this line --------------------------------------', ''])


def build(c):
    d = G.make_case(c['name'], c['U'], c['TI'], c['Hs'], c['Tp'], c['wdir'], c['delta'], c['seed'])
    mods = c.get('mods', {})
    wp = os.path.join(d, 'wind.inp'); wt = open(wp).read()
    wt = G.set_val(wt, 'GridHeight', 170); wt = G.set_val(wt, 'GridWidth', 170); open(wp, 'w').write(wt)
    anch = ANCH.copy(); fair = FAIR.copy()
    for i in range(3):
        r = np.linalg.norm(ANCH[i, :2]); anch[i, :2] *= (r + mods.get('anch_dr', [0, 0, 0])[i]) / r
        rf = np.linalg.norm(FAIR[i, :2]); fair[i, :2] *= (rf + mods.get('fair_dr', [0, 0, 0])[i]) / rf
        fair[i, 2] += mods.get('fair_dz', [0, 0, 0])[i]
    dL = mods.get('dL', [0, 0, 0])
    segs = mods.get('segs') or [[(L0 + dL[i], c['delta'][i])] for i in range(3)]
    txt = moordyn_text(segs, anch, fair, mods.get('ea_s', 1.0), mods.get('m_s', 1.0), mods.get('cd_s', 1.0),
                       mods.get('ca_s', 1.0), mods.get('mu'))
    open(os.path.join(d, 'NRELOffshrBsline5MW_OC4DeepCwindSemi_MoorDyn.dat'), 'w').write(txt)
    if 'ptfm_m_s' in mods:
        p = os.path.join(d, 'NRELOffshrBsline5MW_OC4DeepCwindSemi_ElastoDyn.dat'); t = open(p).read()
        m0 = float(re.search(r'^\s*(\S+)\s+PtfmMass', t, re.M).group(1))
        t = G.set_val(t, 'PtfmMass', f'{m0*mods["ptfm_m_s"]:.1f}'); open(p, 'w').write(t)
    if mods.get('init_above_rated'):
        p = os.path.join(d, 'NRELOffshrBsline5MW_OC4DeepCwindSemi_ElastoDyn.dat'); t = open(p).read()
        for b in (1, 2, 3):
            t = re.sub(rf'^(\s*)\S+(\s+BlPitch\({b}\))', r'\g<1>15\2', t, flags=re.M)
        t = G.set_val(t, 'RotSpeed', 12.1); open(p, 'w').write(t)
    meta = json.load(open(os.path.join(d, 'meta.json')))
    meta.update(split=c['split'], group=c.get('group', ''), mods=mods,
                anch=anch.tolist(), fair=fair.tolist(), segs=[[list(s) for s in sg] for sg in segs])
    json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'))
    return d


def design(seed=3030):
    rng = np.random.default_rng(seed)
    cases = []

    def env(n, Ulo=4.5, Uhi=22.0):
        u = (rng.permutation(n) + rng.random(n)) / n
        U = Ulo + (Uhi - Ulo) * u
        Hs = np.clip(0.9 + 0.17 * U + 0.012 * U**2 + rng.normal(0, 0.35, n), 0.8, 7.5)
        Tp = np.clip(4.2 + 0.30 * U + 0.7 * np.sqrt(Hs) + rng.normal(0, 0.8, n), 6.0, 15.0)
        TI = np.clip(100 * 0.14 * (0.75 * U + 5.6) / U * 0.75 + rng.normal(0, 1.0, n), 5, 25)
        wdir = rng.uniform(-10, 10, n)
        return U, TI, Hs, Tp, wdir

    def one(n, lo=0.05, hi=0.35):
        D = np.zeros((n, 3))
        for k in range(n):
            D[k, rng.integers(3)] = rng.uniform(lo, hi)
        return D

    def add(split, group, D, mods=None, wdir=None, envs=None):
        n = len(D)
        U, TI, Hs, Tp, wd = envs if envs is not None else env(n)
        if wdir is not None:
            wd = wdir
        for k in range(n):
            m = mods[k] if isinstance(mods, list) else (mods or {})
            cases.append(dict(name=f'{split}_{len([c for c in cases if c["split"]==split]):03d}', split=split, group=group,
                              U=float(U[k]), TI=float(TI[k]), Hs=float(Hs[k]), Tp=float(Tp[k]), wdir=float(wd[k]),
                              delta=[float(x) for x in D[k]], seed=int(rng.integers(1, 2**31 - 1)), mods=m))

    # H: additional healthy training simulations (healthy-only environmental model)
    add('htrain', 'healthy-train', np.zeros((20, 3)))
    # A: independent evaluation runs (new turbulence/wave seeds)
    add('rev', 'A1-intact', np.zeros((22, 3)))
    D = np.zeros((16, 3)); pairs = [(0, 1), (0, 2), (1, 2)]
    for k in range(16):
        a, b = pairs[k % 3]; D[k, a] = rng.uniform(0.05, 0.35); D[k, b] = rng.uniform(0.05, 0.35)
    add('rev', 'A2-two-line', D)
    design0 = json.load(open(ROOT + '/design.json'))
    src = [c for c in design0 if c['split'] == 'test' and max(c['delta']) > 0][:8]
    envs = tuple(np.array([c[k] for c in src]) for k in ['U', 'TI', 'Hs', 'Tp', 'wdir'])
    add('rev', 'A3-reseed', np.array([c['delta'] for c in src]), envs=envs,
        mods=[{'reseed_of': c['name']} for c in src])
    D = one(8); D[:2] = 0
    add('rev', 'A4-misalign', D, wdir=rng.choice([-1, 1], 8) * rng.uniform(15, 40, 8))
    D = np.zeros((2, 3)); D[0, 1] = 0.55; D[1, 0] = 0.65
    add('rev', 'A5-grid-edge', D)
    # B: physical mismatch / nuisance changes (twin keeps nominal model)
    def grp(name, mods_list, n_intact):
        n = len(mods_list); D = one(n); D[:n_intact] = 0
        add('mm', name, D, mods=mods_list)
    grp('B1-EA-0.9', [{'ea_s': 0.9}] * 4, 2); grp('B1-EA-1.1', [{'ea_s': 1.1}] * 4, 2)
    grp('B2-mass-0.97', [{'m_s': 0.97}] * 4, 2); grp('B2-mass-1.03', [{'m_s': 1.03}] * 4, 2)
    grp('B3-drag', [{'cd_s': 1.5, 'ca_s': 1.25}] * 4, 2); grp('B3-friction', [{'mu': [1.0, 0.7]}] * 4, 2)
    grp('B4-fair-uniform', [{'fair_dr': [0.5] * 3, 'fair_dz': [0.3] * 3}] * 4, 2)
    grp('B4-fair-random', [{'fair_dr': rng.normal(0, 0.5, 3).round(3).tolist(), 'fair_dz': rng.normal(0, 0.3, 3).round(3).tolist()} for _ in range(4)], 2)
    for s in [1.005, 0.995, 1.01, 0.99]:
        grp(f'B5-ptfm-mass-{s}', [{'ptfm_m_s': s}] * 2, 1)
    # B6 localized degradation (delta vector = 0; damage described by segments)
    loc = []
    for k in range(8):
        i = int(rng.integers(3)); dl = [0.3, 0.5][k % 2]
        if k < 4:   # fairlead region, last 100 m
            sg = [(L0 - 100.0, 0.0), (100.0, dl)]
        else:       # touchdown region, 170-320 m from anchor
            sg = [(170.0, 0.0), (150.0, dl), (L0 - 320.0, 0.0)]
        segs = [[(L0, 0.0)] for _ in range(3)]; segs[i] = sg
        loc.append({'segs': segs, 'loc_line': i, 'loc_delta': dl, 'loc_where': 'fairlead' if k < 4 else 'touchdown'})
    add('mm', 'B6-localized', np.zeros((8, 3)), mods=loc)
    # C: installations with truly perturbed anchors / line lengths in MoorDyn
    inst = [('I1', 2.0, 0.5, None), ('I2', 2.0, 0.5, None), ('I3', 5.0, 0.5, None), ('I4', 5.0, 1.0, None), ('I5', 2.0, 0.5, 1.005)]
    for nm, sr, sl, pm in inst:
        dr = rng.normal(0, sr, 3).round(3).tolist(); dl_ = rng.normal(0, sl, 3).round(3).tolist()
        m = {'anch_dr': dr, 'dL': dl_, 'install': nm}
        if pm:
            m['ptfm_m_s'] = pm
        D = np.zeros((8, 3)); D[5:] = one(3)
        add('inst', f'C-{nm}', D, mods=[dict(m, role='commission' if k < 4 else 'test') for k in range(8)])
    return cases


if __name__ == '__main__':
    cases = design()
    json.dump(cases, open(ROOT + '/design_rev.json', 'w'), indent=1)
    which = sys.argv[1:]
    for c in cases:
        if not which or c['name'] in which:
            build(c)
    from collections import Counter
    print(len(cases), Counter(c['group'] for c in cases))
