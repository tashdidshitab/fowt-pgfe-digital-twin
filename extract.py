"""Extract the 400-s analysis window of each finished run to a compact npz."""
from paths import ROOT, RTEST
import os, json, glob, numpy as np
from openfast_toolbox.io import FASTOutputFile

RUNS = ROOT + '/runs'
OUT = ROOT + '/data'
T0 = 120.0
CH = {'surge': 'PtfmSurge_[m]', 'sway': 'PtfmSway_[m]', 'heave': 'PtfmHeave_[m]',
      'roll': 'PtfmRoll_[deg]', 'pitch': 'PtfmPitch_[deg]', 'yaw': 'PtfmYaw_[deg]',
      'ax': 'PtfmTAxt_[m/s^2]', 'ay': 'PtfmTAyt_[m/s^2]', 'az': 'PtfmTAzt_[m/s^2]',
      'p': 'PtfmRVxt_[deg/s]', 'q': 'PtfmRVyt_[deg/s]', 'r': 'PtfmRVzt_[deg/s]',
      'nac_ax': 'NcIMUTAxs_[m/s^2]', 'nac_ay': 'NcIMUTAys_[m/s^2]',
      'rotspd': 'RotSpeed_[rpm]', 'pitchb': 'BldPitch1_[deg]', 'genpwr': 'GenPwr_[kW]', 'gentq': 'GenTq_[kN-m]',
      'u_hub': 'Wind1VelX_[m/s]', 'v_hub': 'Wind1VelY_[m/s]',
      'eta_ptfm': 'Wave1Elev_[m]', 'eta_buoy': 'Wave2Elev_[m]',
      'thrust': 'RtFldFxh_[N]', 'twr_my': 'TwrBsMyt_[kN-m]', 'twr_mx': 'TwrBsMxt_[kN-m]', 'twr_fx': 'TwrBsFxt_[kN]',
      'T1': 'FAIRTEN1_[N]', 'T2': 'FAIRTEN2_[N]', 'T3': 'FAIRTEN3_[N]',
      'A1': 'ANCHTEN1_[N]', 'A2': 'ANCHTEN2_[N]', 'A3': 'ANCHTEN3_[N]',
      'hydro_fx': 'HydroFxi_[N]', 'hydro_fy': 'HydroFyi_[N]', 'hydro_mz': 'HydroMzi_[N-m]'}


def extract(name, force=False):
    d = os.path.join(RUNS, name)
    out = os.path.join(OUT, name + '.npz')
    if os.path.exists(out) and not force:
        return out
    log = open(os.path.join(d, 'of.log')).read() if os.path.exists(os.path.join(d, 'of.log')) else ''
    if 'terminated normally' not in log:
        return None
    df = FASTOutputFile(os.path.join(d, '5MW_OC4Semi_WSt_WavesWN.outb')).toDataFrame()
    df = df[df['Time_[s]'] >= T0 - 1e-6]
    meta = json.load(open(os.path.join(d, 'meta.json')))
    arrs = {k: df[v].to_numpy(np.float32) for k, v in CH.items()}
    for j in (4, 5, 6):          # optional fairlead force vectors (revision runs)
        for ax in 'XYZ':
            col = f'POINT{j}F{ax}_[N]'
            if col in df:
                arrs[f'P{j}F{ax}'] = df[col].to_numpy(np.float32)
    arrs['t'] = df['Time_[s]'].to_numpy(np.float32)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(out, meta=json.dumps(meta), **arrs)
    return out


def load(name):
    z = np.load(os.path.join(OUT, name + '.npz'))
    D = {k: z[k] for k in z.files if k != 'meta'}
    D['meta'] = json.loads(str(z['meta']))
    return D


def available():
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(OUT + '/*.npz'))
    return names


if __name__ == '__main__':
    design = json.load(open(ROOT + '/design.json')) + json.load(open(ROOT + '/design_rev.json'))
    n = 0
    for c in design:
        if extract(c['name']):
            n += 1
    print('extracted', n)
