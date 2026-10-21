@echo off
rem lane Y: v30 production config dumps (phys_v30pc pcrev=tan), hbsqry (timsTOF sibling)
call C:\prog\casmi\fwdmodel\pc5\run5.bat hbsqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hbs30_0 phys_v30pc 2 ",pcrev=tan"
echo ALLDONE
