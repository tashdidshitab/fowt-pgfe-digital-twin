"""Full-fidelity replicas of selected campaign cases (same wind/wave seeds) for checking the
reduced-fidelity settings across sea states and damage states."""
from paths import ROOT, RTEST
import json, sys, gen_cases as G
G.TEMPLATE = ROOT + '/openfast_templates/full'
G.RUNS = ROOT + '/runs_ff'
cases = {c['name']: c for c in json.load(open(ROOT + '/design.json'))}
PICK = ['train_001', 'test_010', 'ood_009', 'ood_019']
if __name__ == '__main__':
    for n in PICK:
        c = cases[n]
        print(n, round(c['U'], 1), round(c['Hs'], 1), [round(x, 2) for x in c['delta']])
        G.make_case(n, c['U'], c['TI'], c['Hs'], c['Tp'], c['wdir'], c['delta'], c['seed'])
