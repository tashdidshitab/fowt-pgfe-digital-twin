from paths import ROOT, RTEST
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = dict(pg='#2a78d6', gbm='#eb6834', rf='#1baf7a', grey='#8a8984', ink='#0b0b0b', ink2='#52514e', fill='#eef4fc', fill2='#f4f3f0')
fig, ax = plt.subplots(figsize=(10.5, 4.4)); ax.set_xlim(0, 105); ax.set_ylim(0, 44); ax.axis('off')
plt.rcParams['font.family'] = 'DejaVu Sans'


def box(x, y, w, h, title, body, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec=ec, lw=1.2))
    ax.text(x + 1, y + h - 1.4, title, fontsize=7.9, weight='bold', color=C['ink'], va='top')
    ax.text(x + 1, y + h - 4.6, body, fontsize=7.4, color=C['ink2'], va='top', linespacing=1.35)


def arrow(x0, y0, x1, y1, col=C['ink2']):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=10, color=col, lw=1.2))


box(0.5, 24, 19, 18, 'Asset (OpenFAST)', 'NREL 5-MW on OC4\nDeepCwind semi-sub\nTurbSim wind, JONSWAP\nwaves, 2nd-order drift\nMoorDyn lines with\ncorrosion/wear δ_i', C['fill2'], C['grey'])
box(0.5, 1, 19, 19, 'Sparse sensors', 'RTK-GNSS: surge, sway,\nheave, heading\nMRU: roll, pitch, rates\nSCADA: rotor speed,\npitch, power, nacelle\nwind; wave buoy 120 m\nNo tension / strain gauges', C['fill2'], C['grey'])
arrow(10, 24, 10, 20.8)
box(25, 24, 25, 18, '(1) Environmental load', 'hydrostatics F_hs(q̄) (design\nmodel) + ridge on pose-free\nSCADA/buoy features\n(thrust, Hs, Tm, drift proxies)\ntrained on INTACT sims only;\nresidual covariance from CV', C['fill'], C['pg'])
box(25, 1, 25, 19, '(2) Physics core', 'Elastic catenary with seabed\ncontact, tabulated on\n(span, height, δ)\nF̄_moor(q(t); δ) averaged\nover measured pose history\n(mean load vs MoorDyn: Sec. 4.1)', C['fill'], C['pg'])
arrow(20, 11, 24.4, 11); arrow(20, 15, 24.4, 30)
box(55.5, 12, 23, 22, '(3) PG-FE inversion', 'Window-mean equilibrium (6 DOF)\nΣ_i F̄_i(δ_i) + F̄_env ≈ 0\nGaussian likelihood ×\nspike-and-slab prior (p0 = 0.9)\nExact grid posterior over\n(δ1, δ2, δ3) → mean, CIs,\nP(δ_i > 0.05), localisation', C['fill'], C['pg'])
arrow(50.6, 33, 55, 28); arrow(50.6, 10, 55, 18)
box(55.5, 1, 22, 9, '(0) Commissioning cal.', 'intact windows → per-line\nanchor radius (± uncert.)', C['fill2'], C['grey'])
arrow(66.5, 10.5, 66.5, 11.5)
box(83, 29, 21.5, 13, 'Virtual sensing, fatigue', 'Fairlead tensions from pose\n+ δ̂ (QS + dynamic residual)\ncausal: δ̂ of previous window\nRainflow DEL (m = 3, 4)', C['fill2'], C['grey'])
box(83, 14.5, 21.5, 12, 'Motion forecasting', 'Buoy preview 120 m up-wave\n2-20 s ahead: pitch,\nsurge, heave', C['fill2'], C['grey'])
box(83, 1, 21.5, 11, 'Health state output', 'δ̂_i ± CI per line\nalarm + localisation\nfatigue DEL per window', C['fill2'], C['grey'])
arrow(78.2, 27, 82.6, 34); arrow(78.2, 20, 82.6, 20); arrow(78.2, 15, 82.6, 7)
import os; os.makedirs(ROOT + '/figures', exist_ok=True)
fig.savefig(ROOT + '/figures/Figure_1.png', dpi=300, bbox_inches='tight')
fig.savefig(ROOT + '/figures/Figure_1.pdf', bbox_inches='tight'); print('ok')
