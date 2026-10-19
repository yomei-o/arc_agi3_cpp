@echo off
rem extra point: hbqry rm=1 (enveda-180-like, PubChem popularity 1.30 = lower than hcqry 2.09)
call C:\prog\casmi\fwdmodel\probe\run.bat hbqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin pr_hb1 ",rm=1"
echo ALLDONE
