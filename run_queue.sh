#!/bin/bash
# Resumable queue: runs cases listed in todo_rev.txt (rev cases) and todo_ff.txt (full-fidelity
# replicas), 2 at a time. Starts no new run after DEADLINE (s since start) so the job ends cleanly.
: "${FOWT_ROOT:=$(cd "$(dirname "$0")" && pwd)}"; export FOWT_ROOT   # openfast, turbsim and python must be on PATH
cd "$FOWT_ROOT"
export T0=$(date +%s) DEADLINE=${DEADLINE:-6300}
python -c "
import gen_rev as R, json, os
cs={c['name']:c for c in json.load(open('design_rev.json'))}
for n in open('todo_rev.txt').read().split():
    if not os.path.exists(f'runs/{n}/of.log') or 'terminated normally' not in open(f'runs/{n}/of.log').read():
        R.build(cs[n])
"
python gen_ff.py > /dev/null
run_one() {
  n=$1; d=$FOWT_ROOT/runs/$n; [ -d $FOWT_ROOT/runs_ff/${n#ff:} ] && [[ $n == ff:* ]] && d=$FOWT_ROOT/runs_ff/${n#ff:}
  if grep -q "terminated normally" $d/of.log 2>/dev/null; then return; fi
  if [ $(( $(date +%s) - T0 )) -gt $DEADLINE ]; then return; fi
  cd $d; turbsim wind.inp > ts.log 2>&1; s=$(date +%s)
  openfast 5MW_OC4Semi_WSt_WavesWN.fst > of.log 2>&1
  st=$(grep -q "terminated normally" of.log && echo OK || echo FAIL)
  rm -f wind.bts wind.twr
  echo "$(date +%H:%M:%S) $n $st $(( $(date +%s)-s ))s" >> $FOWT_ROOT/campaign_rev.log
}
export -f run_one
( cat todo_rev.txt; echo; for n in train_001 test_010 ood_009 ood_019; do echo ff:$n; done ) | grep -v '^$' | xargs -P 2 -I{} bash -c 'run_one {}'
echo "QUEUE-END $(date +%H:%M:%S)" >> $FOWT_ROOT/campaign_rev.log
