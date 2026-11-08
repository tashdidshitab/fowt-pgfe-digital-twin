# Supplementary material: physics-guided digital twin for floating offshore wind mooring integrity

Code, simulation inputs and results supporting the paper:

> T.S. Ahmed, *Physics-guided digital twin for floating offshore wind turbine mooring integrity: per-line degradation estimation from sparse sensors under coupled wind–wave loading* (submitted to *Ocean Engineering*).

The repository contains everything needed to regenerate the simulation campaign, the experiments, the figures and the numbers reported in the paper. The paper itself is not included.

The digital twin estimates the corrosion/wear section loss of each mooring line of a floating wind turbine from GNSS, motion-reference-unit and SCADA data, without sensors on the lines. Its physics-guided fault estimator (PG-FE) inverts the window-mean 6-DOF equilibrium between a tabulated elastic catenary and an environmental load trained on intact-system simulations only. The study uses 328 coupled OpenFAST v5.0.0 simulations of the NREL 5-MW turbine on the OC4-DeepCwind semisubmersible.

## Contents

```
paths.py                 project locations (override with env vars FOWT_ROOT, OPENFAST_RTEST)
design.json              main campaign (164 runs): environment, per-line δ, TurbSim/wave seeds
design_rev.json          revision campaign (164 runs): healthy-train, evaluation, mismatch, installations
campaign*.log            run logs of both campaigns
openfast_templates/      OpenFAST input templates: fast_ua (campaign settings), full (full-fidelity
                         replicas), TurbSim template
gen_cases.py             case generator (main campaign) + campaign design
gen_rev.py               revision campaign design and generator (mismatch, installations, localised damage)
gen_ff.py                full-fidelity replicas
static/make_static.py    static OpenFAST tests -> hydrostatic restoring fit (static/hydrostatic_fit.json)
run_campaign.sh, run_batch.sh   runners for the main campaign (TurbSim + OpenFAST, 2 in parallel)
run_queue.sh             resumable runner for the revision campaign
extract.py               OpenFAST .outb -> 400-s analysis windows in data/*.npz
catenary.py, physics.py  elastic catenary with seabed contact; tabulated physics core (cat_table.npz)
sensors.py               sparse/noisy sensor model (configurations S1-S3, noise, offsets)
dt.py, rev_core.py       PG-FE (environmental model, Bayesian inversion, calibration, weight nuisance),
                         physics baselines (WLS, L1-WLS, EKF), tuned data-driven baselines
rev_exp.py               all experiments (E1-E9) -> results/rev_*.json
vsense.py, forecast.py   virtual sensing / fatigue; motion forecasting with wave preview
exp_fault.py, exp_vs.py, exp_fc.py, make_figs.py   first-version pipeline (helpers reused by rev_exp.py;
                         results/fc_main.json comes from exp_fc.py)
make_figs_rev.py, fig1.py   figures -> figures/
build_numbers.py         all reported numbers -> results/numbers.tex, results/numbers_rev.json
results/                 all experiment outputs (JSON)
data/                    extracted simulation windows: download separately (see data/README.md)
```

## Reproducing

### 1. Environment
```bash
conda env create -f environment.yml && conda activate fowt-pgfe
git clone --depth 1 --branch v5.0.0 https://github.com/OpenFAST/r-test.git
export FOWT_ROOT=$PWD OPENFAST_RTEST=$PWD/r-test
```
Compile the Bladed-style controller used by the OC4 model:
```bash
cd $OPENFAST_RTEST/glue-codes/openfast/5MW_Baseline/ServoData
gfortran -O2 -shared -fPIC -ffree-line-length-none -o DISCON_OC3Hywind.dll DISCON_OC3/DISCON_OC3Hywind.F90
cd $FOWT_ROOT
```

### 2. Figures and numbers from the stored results (no simulation needed)
```bash
python build_numbers.py
python fig1.py
python make_figs_rev.py installs:6 vs:8 fc:9 parity:3 uq:4 mismatch:5 robust:7
```
`make_figs_rev.py campaign:2` also needs `data/`.

### 3. Re-running the experiments
Download the extracted windows into `data/` (see `data/README.md`), then:
```bash
export OMP_NUM_THREADS=1   # important: avoids thread over-subscription in scikit-learn
python rev_exp.py e1_main e2_uq e1b_sequential e3_mismatch e4_install e5_lc e5_noise e5_win e6_causal e7_vs e8_forecast
```
`e9_fidelity` additionally needs the raw OpenFAST outputs of the four full-fidelity replicas and their reduced-fidelity counterparts.

### 4. Re-running the simulations (about 15 CPU-hours on 2 cores)
```bash
./run_campaign.sh                    # main campaign: builds runs/ from design.json, runs TurbSim + OpenFAST (resumable)
python -c "import json; print('\n'.join(c['name'] for c in json.load(open('design_rev.json'))))" > todo_rev.txt
./run_queue.sh                       # revision campaign + 4 full-fidelity replicas (resumable; re-launch until done)
python static/make_static.py         # static tests -> hydrostatic restoring fit
python extract.py                    # -> data/*.npz
```

## Notes
- Wave heading is fixed at 0° because the second-order WAMIT database of the OC4 platform exists only for that heading. Directional variation is introduced through wind–wave misalignment.
- Installation runs `inst_000`–`inst_015` and `inst_021`–`inst_031` were simulated with a 150 m TurbSim grid. The grid was widened to 170 m (the current code) after runs with an 11 m anchor offset left the grid.
- `cat_table.npz` (catenary table) is regenerated automatically by `physics.py` if deleted.
- Random seeds: campaign design seeds 2026 (main) and 3030 (revision). Per-run TurbSim and wave seeds are stored in the design files. Sensor-noise seeds are derived deterministically from run name and window (CRC32).

## License
No license has been chosen yet. Add one (e.g. MIT for code, CC-BY-4.0 for data) before making the repository public.

## Note on openfast_toolbox
With NumPy 2.x, `openfast_toolbox` 3.5.1 failed to read some binary `.outb` files. In
`openfast_toolbox/io/fast_output_file.py`, change the line
`data = (data - ColOff) / ColScl` to `data = (data - np.ravel(ColOff)) / np.ravel(ColScl)`.
