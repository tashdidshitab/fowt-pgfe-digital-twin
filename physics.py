"""Physics core of the digital twin.
 - Tabulated quasi-static elastic catenary (H, V) as a function of horizontal
   span l, vertical height h and per-line degradation delta, so that line loads
   for whole time histories and many candidate degradation states are cheap.
 - 6-DOF mooring load from measured platform pose.
"""
from paths import ROOT, RTEST
import os, numpy as np
from scipy.interpolate import RegularGridInterpolator
from catenary import solve_line, line_props, L0, ANCH, FAIR, rotmat

TAB = ROOT + '/cat_table.npz'
# twin-model anchor position error (m, per line, radial) to emulate model-form
# error: as-installed anchor radius / unstretched length differ from design
ANCH_PERT = np.zeros(3)
LG = np.arange(730.0, 836.0, 1.0)
HG = np.arange(168.0, 204.0, 1.0)
DG = np.round(np.arange(0.0, 0.6001, 0.025), 4)


def build_table():
    Hs = np.full((len(DG), len(LG), len(HG)), np.nan); Vs = Hs.copy()
    for a, d in enumerate(DG):
        EA, w = line_props(d)
        for c, h in enumerate(HG):
            g = None
            for b, l in enumerate(LG):
                try:
                    H, V, g = solve_line(l, h, L0, EA, w, g)
                    r = np.array([H, V])
                    if np.all(np.isfinite(r)):
                        Hs[a, b, c], Vs[a, b, c] = H, V
                except Exception:
                    g = None
    np.savez(TAB, H=Hs, V=Vs, L=LG, Hh=HG, D=DG)


class CatTable:
    def __init__(self):
        if not os.path.exists(TAB):
            build_table()
        z = np.load(TAB)
        pts = (z['D'], z['L'], z['Hh'])
        self.fH = RegularGridInterpolator(pts, np.log(z['H']), bounds_error=False, fill_value=None)
        self.fV = RegularGridInterpolator(pts, np.log(z['V']), bounds_error=False, fill_value=None)

    def HV(self, delta, l, h):
        P = np.stack(np.broadcast_arrays(delta, l, h), -1)
        return np.exp(self.fH(P)), np.exp(self.fV(P))


_CT = None


def table():
    global _CT
    if _CT is None:
        _CT = CatTable()
    return _CT


def fairlead_kin(q):
    """q: (N,6) [surge,sway,heave (m), roll,pitch,yaw (rad)] -> fairlead
    positions (3,N,3) and lever arms (3,N,3) in inertial frame."""
    r, p, y = q[:, 3], q[:, 4], q[:, 5]
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = cy * cp; R[:, 0, 1] = cy * sp * sr - sy * cr; R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp; R[:, 1, 1] = sy * sp * sr + cy * cr; R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp;     R[:, 2, 1] = cp * sr;                R[:, 2, 2] = cp * cr
    arm = np.einsum('nij,kj->kni', R, FAIR)
    pos = q[None, :, :3] + arm
    return pos, arm


def line_loads(q, delta_i, i):
    """Quasi-static load of line i on the platform for pose history q (N,6)
    and degradation delta_i (scalar or (N,)). Returns F (N,6) about the
    platform reference point and fairlead tension (N,)."""
    pos, arm = fairlead_kin(q)
    a = ANCH[i, :2] * (1 + ANCH_PERT[i] / np.linalg.norm(ANCH[i, :2]))
    d = pos[i, :, :2] - a
    l = np.linalg.norm(d, axis=1)
    h = pos[i, :, 2] - ANCH[i, 2]
    H, V = table().HV(delta_i, l, h)
    u = d / l[:, None]
    f = np.stack([-H * u[:, 0], -H * u[:, 1], -V], 1)
    m = np.cross(arm[i], f)
    return np.concatenate([f, m], 1), np.hypot(H, V)


if __name__ == '__main__':
    import time
    t = time.time(); build_table(); print('table built in %.1fs' % (time.time() - t))
    z = np.load(TAB); print('nan fraction', np.isnan(z['H']).mean())
    # check vs direct solver
    from catenary import Mooring
    rng = np.random.default_rng(0)
    q = np.c_[rng.uniform(-15, 15, (50, 2)), rng.uniform(-3, 3, 50), np.radians(rng.uniform(-4, 4, (50, 3)))]
    err = []
    for dl in [0.0, 0.17, 0.33, 0.5]:
        for k in range(50):
            F, T = Mooring(delta=(dl, dl, dl)).forces(q[k])
            Ft = sum(line_loads(q[k:k+1], dl, i)[0][0] for i in range(3))
            Tt = np.array([line_loads(q[k:k+1], dl, i)[1][0] for i in range(3)])
            err.append(np.abs(Tt - T).max() / T.max())
    print('max rel tension error table vs direct: %.2e' % np.max(err))
