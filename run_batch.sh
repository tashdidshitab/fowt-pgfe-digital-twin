#!/bin/bash
# Run a fixed list of cases (args), 2 at a time; skips finished ones.
: "${FOWT_ROOT:=$(cd "$(dirname "$0")" && pwd)}"; export FOWT_ROOT   # openfast, turbsim and python must be on PATH
cd "$FOWT_ROOT"
python gen_cases.py "$@" > /dev/null
run_one() {
  d=$FOWT_ROOT/runs/$1
  if grep -q "terminated normally" $d/of.log 2>/dev/null; then return; fi
  cd $d; turbsim wind.inp > ts.log 2>&1; s=$(date +%s)
  openfast 5MW_OC4Semi_WSt_WavesWN.fst > of.log 2>&1
  st=$(grep -q "terminated normally" of.log && echo OK || echo FAIL)
  rm -f wind.bts wind.twr
  echo "$(date +%H:%M:%S) $1 $st $(( $(date +%s)-s ))s" >> $FOWT_ROOT/campaign.log
}
export -f run_one
printf "%s\n" "$@" | xargs -P 2 -I{} bash -c 'run_one {}'
