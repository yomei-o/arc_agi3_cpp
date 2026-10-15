@echo off
rem ev.bat <tag> <params> <coef> <QM_PA> <QM_DIR>
cd /d C:\prog\casmi
set QM_COEF=%3
set QM_PA=%4
set QM_DIR=%5
for %%q in (cqry tqry) do C:\prog\casmi\fwdmodel\qm\phys_dag_qm.exe eval C:\prog\casmi %%q pool730 10 %2 "" fwd_mlp_rank_xlib.bin 0.25 "" rev_x3072.bin 1 xlib 8 32 8 > qm\evd_%1_%%q.log 2>&1
findstr /c:"MRR@25" /c:"ablation" /c:"w=0.25  wr=1     MRR" /c:"wl=8   wa=8 " qm\evd_%1_cqry.log qm\evd_%1_tqry.log
