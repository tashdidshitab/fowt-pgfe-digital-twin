#!/bin/bash
: "${FOWT_ROOT:=$(cd "$(dirname "$0")" && pwd)}"; export FOWT_ROOT   # openfast, turbsim and python must be on PATH
cd "$FOWT_ROOT"; P=python; export OMP_NUM_THREADS=1
$P extract.py > /dev/null
( for e in e1_main e2_uq e1b_sequential e3_mismatch; do $P rev_exp.py $e > logs_rev_$e.txt 2>&1; done ) &
( for e in e6_causal e7_vs e8_forecast e5_robust e9_fidelity; do $P rev_exp.py $e > logs_rev_$e.txt 2>&1; done ) &
wait; echo EXPS-DONE
