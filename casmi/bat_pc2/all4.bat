@echo off
rem fourth pass: old channel B (pcv=1) + pool popularity, normal and answer-removed, on the popularity-matched holdout
call C:\prog\casmi\fwdmodel\pc2\run.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hcBP0 ",pcv=1,ppop=0.15,ppgate=0.9"
call C:\prog\casmi\fwdmodel\pc2\run.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hcBP1 ",rm=1,pcv=1,ppop=0.15,ppgate=0.9"
echo ALLDONE
