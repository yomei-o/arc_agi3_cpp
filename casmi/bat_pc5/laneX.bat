@echo off
rem lane X: v30 production config dumps (phys_v30pc pcrev=tan) for the Boltzmann-merge prototype
call C:\prog\casmi\fwdmodel\pc5\run5.bat haqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin ha30_1 phys_v30pc 2 ",rm=1,pcrev=tan"
echo ALLDONE
