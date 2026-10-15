#!/bin/bash
# pooled grouped 5-fold CV over all six dumps
cd "$(dirname "$0")"
TR="train=f_yc.txt:N,f_yt.txt:N,f_ycrm.txt:R,f_ytrm.txt:R,f_ys.txt:L,f_yo.txt:L"
NOLIB="drop=lib,lib_m_max,rk_lib,q_maxlib,q_lib2,q_nlib80"
run() { # name args...
  n=$1; shift
  for k in 0 1 2 3 4; do
    ssh -i ~/.claude/keys/ort_build_key yomei@192.168.6.14 "cd /d C:\prog\casmi\lmo && C:\prog\casmi\fwdmodel\lmart\lmart3.exe train $TR $* valfrac=0.2 fold=$k/5 out=cvy_m.txt thr=4 oof=oof_${n}.txt > cvy_${n}_$k.log 2>&1 & findstr FOLDRR cvy_${n}_$k.log & findstr /c:\"best iteration\" cvy_${n}_$k.log" | sed "s/^/[$n $k] /"
  done
}
for a in ${MIXES:-0.1 0.3 0.5 0.7}; do
  n=$(python -c "print(round((1-$a)/2,3))")
  M="mix=N:$n,R:$n,L:$a"
  for v in ${VARS:-none hand nolib}; do
    X=""; [ $v = hand ] && X="base=hand:1"; [ $v = nolib ] && X="$NOLIB"; [ $v = handnolib ] && X="base=hand:1 $NOLIB"
    run ${v}_$a $M $X $EXTRA
  done
done
[ -n "$CTRL" ] && run perm_none_0.5 mix=N:0.25,R:0.25,L:0.5 permute=1
echo CVDONE
