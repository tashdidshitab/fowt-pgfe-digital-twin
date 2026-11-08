"""Sparse, noisy sensor model of a commercially realistic FOWT monitoring setup.
Truth channels come from OpenFAST; measurements add noise/bias and are
down-sampled. Nothing on the mooring lines or tower is instrumented."""
import numpy as np

FS_OUT = 20.0   # OpenFAST output rate
FS = 4.0        # sensor rate used by the twin

# 1-sigma noise levels (scale=1)
NOISE = dict(
    surge=0.10, sway=0.10, heave=0.05,          # RTK-GNSS (m)
    yaw=0.15,                                   # dual-antenna GNSS heading (deg)
    roll=0.02, pitch=0.02,                      # MRU (deg)
    ax=0.01, ay=0.01, az=0.01,                  # MRU accelerations (m/s^2)
    p=0.01, q=0.01, r=0.01,                     # MRU rates (deg/s)
    nac_ax=0.02,                                # nacelle accelerometer (m/s^2)
    rotspd=0.05, pitchb=0.05, genpwr=20.0,      # SCADA (rpm, deg, kW)
    u_hub=0.4, v_hub=0.4,                       # nacelle anemometer/vane (m/s)
    eta_buoy=0.03,                              # wave buoy (m)
)
# multiplicative calibration bias (1-sigma), drawn once per window
BIAS = dict(u_hub=0.03, genpwr=0.01)

# per-window constant offsets (1-sigma at offset level 1): residual of tide /
# draft correction for heave, MRU mounting misalignment + ballast trim drift
# for roll/pitch, GNSS antenna lever-arm / heading calibration
OFFSET = dict(surge=0.05, sway=0.05, heave=0.05, roll=0.05, pitch=0.05, yaw=0.1)

SETS = {
    # name: channels available
    'S1_GNSS+SCADA':      ['surge', 'sway', 'heave', 'yaw', 'rotspd', 'pitchb', 'genpwr', 'u_hub', 'v_hub'],
    'S2_GNSS+MRU+SCADA':  ['surge', 'sway', 'heave', 'yaw', 'roll', 'pitch', 'ax', 'ay', 'az', 'p', 'q', 'r', 'nac_ax', 'rotspd', 'pitchb', 'genpwr', 'u_hub', 'v_hub'],
    'S3_full_sparse':     ['surge', 'sway', 'heave', 'yaw', 'roll', 'pitch', 'ax', 'ay', 'az', 'p', 'q', 'r', 'nac_ax', 'rotspd', 'pitchb', 'genpwr', 'u_hub', 'v_hub', 'eta_buoy'],
}
DEFAULT = 'S3_full_sparse'


def measure(D, chans=None, noise_scale=1.0, seed=0, t0=None, dur=None, offset=0.0):
    """Return dict of measured channels at FS Hz for the window [t0, t0+dur)."""
    chans = chans or SETS[DEFAULT]
    rng = np.random.default_rng(seed)
    t = D['t']
    m = np.ones(len(t), bool)
    if t0 is not None:
        m = (t >= t0) & (t < t0 + dur)
    step = int(round(FS_OUT / FS))
    out = {'t': t[m][::step].astype(float)}
    for c in chans:
        x = D[c][m].astype(float)
        if c in BIAS:
            x = x * (1 + noise_scale * BIAS[c] * rng.standard_normal())
        # anti-alias: average over the decimation block before sampling
        n = len(x) // step * step
        x = x[:n].reshape(-1, step).mean(1)
        x = x + noise_scale * NOISE[c] * rng.standard_normal(len(x))
        if c in OFFSET:
            x = x + offset * OFFSET[c] * rng.standard_normal()
        out[c] = x[:len(out['t'])]
    out['t'] = out['t'][:len(out[chans[0]])]
    return out


def pose(S):
    """6-DOF pose array (N,6) in m / rad from measured channels (roll/pitch
    default to 0 if the MRU is absent)."""
    z = np.zeros(len(S['t']))
    return np.c_[S['surge'], S['sway'], S['heave'], np.radians(S.get('roll', z)),
                 np.radians(S.get('pitch', z)), np.radians(S['yaw'])]
