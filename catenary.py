"""Quasi-static elastic catenary with seabed contact (no friction) for the
OC4-DeepCwind semisubmersible mooring system. Used as the physics core of the
digital twin: given platform pose and per-line degradation, returns fairlead
tensions and the 6x6 mooring restoring force."""
import numpy as np
from scipy.optimize import fsolve

G = 9.80665
RHO = 1025.0
D0 = 0.0766          # nominal line diameter (m)
M0 = 113.35          # nominal mass per length (kg/m)
EA0 = 7.536e8        # nominal axial stiffness (N)
L0 = 835.35          # unstretched length (m)
ANCH = np.array([[418.8, 725.383, -200.0], [-837.6, 0.0, -200.0], [418.8, -725.383, -200.0]])
FAIR = np.array([[20.434, 35.393, -14.0], [-40.868, 0.0, -14.0], [20.434, -35.393, -14.0]])


def line_props(delta):
    """Degradation delta = fraction of load-bearing cross-section lost
    (corrosion/wear). EA and dry mass scale with (1-delta); displaced volume
    scales with diameter^2 too, so wet weight scales with (1-delta)."""
    s = 1.0 - delta
    EA = EA0 * s
    w = (M0 - RHO * np.pi * D0**2 / 4) * G * s
    return EA, w


def _residual(x, l, h, L, EA, w):
    H, V = np.exp(x)  # positivity
    if V < w * L:  # part of line on seabed
        LB = L - V / w
        r_l = LB + H / w * np.arcsinh(V / H) + H * L / EA - l
        r_h = H / w * (np.sqrt(1 + (V / H) ** 2) - 1) + V**2 / (2 * EA * w) - h
    else:
        Va = V - w * L
        r_l = H / w * (np.arcsinh(V / H) - np.arcsinh(Va / H)) + H * L / EA - l
        r_h = H / w * (np.sqrt(1 + (V / H) ** 2) - np.sqrt(1 + (Va / H) ** 2)) + (V * L - w * L**2 / 2) / EA - h
    return [r_l / l, r_h / h]


def solve_line(l, h, L, EA, w, guess=None):
    if guess is None:
        guess = np.log([max(1e3, 1e5), w * L * 0.6])
    sol, info, ier, _ = fsolve(_residual, guess, args=(l, h, L, EA, w), full_output=True, xtol=1e-12)
    H, V = np.exp(sol)
    return H, V, sol


def rotmat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Mooring:
    def __init__(self, delta=(0, 0, 0), dL=(0, 0, 0)):
        self.delta = np.asarray(delta, float)
        self.dL = np.asarray(dL, float)
        self._g = [None] * 3

    def forces(self, q):
        """q = [surge, sway, heave, roll, pitch, yaw] (m, rad). Returns
        (6-vector mooring load on platform about ref point, fairlead tensions)."""
        R = rotmat(*q[3:])
        F = np.zeros(6)
        T = np.zeros(3)
        for i in range(3):
            EA, w = line_props(self.delta[i])
            rf = R @ FAIR[i]
            pf = q[:3] + rf
            d = pf[:2] - ANCH[i, :2]
            l = np.linalg.norm(d)
            h = pf[2] - ANCH[i, 2]
            H, V, self._g[i] = solve_line(l, h, L0 + self.dL[i], EA, w, self._g[i])
            u = d / l
            f = np.array([-H * u[0], -H * u[1], -V])
            F[:3] += f
            F[3:] += np.cross(rf, f)
            T[i] = np.hypot(H, V)
        return F, T

    def stiffness(self, q, eps=None):
        if eps is None:
            eps = np.array([0.05, 0.05, 0.05, 1e-3, 1e-3, 1e-3])
        K = np.zeros((6, 6))
        for j in range(6):
            dq = np.zeros(6); dq[j] = eps[j]
            Fp, _ = self.forces(q + dq)
            Fm, _ = self.forces(q - dq)
            K[:, j] = -(Fp - Fm) / (2 * eps[j])
        return K


if __name__ == "__main__":
    q0 = np.zeros(6)
    m = Mooring()
    F, T = m.forces(q0)
    print("healthy: fairlead tensions (kN)", np.round(T / 1e3, 1), "net F", np.round(F / 1e3, 1))
    print("Kxx, Kyy, Kzz, Kpp(pitch), Kyaw:", [f"{v:.4g}" for v in np.diag(m.stiffness(q0))[[0, 1, 2, 4, 5]]])
    for dlt in [0.1, 0.2, 0.3, 0.4]:
        md = Mooring(delta=(0, dlt, 0))
        F, T = md.forces(q0)
        K = md.stiffness(q0)
        print(f"line2 delta={dlt}: T(kN)={np.round(T/1e3,1)}  Kxx={K[0,0]:.4g} Kyaw={K[5,5]:.4g} Fx0={F[0]/1e3:.1f}kN")
