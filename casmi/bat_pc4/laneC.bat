@echo off
rem lane C: v30 (pcrev=tan) on hasqry (sibling holdout, answer in the library)
call C:\prog\casmi\fwdmodel\pc4\run4.bat hasqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin has30_0 phys_v30pc 2 ",pcrev=tan"
echo ALLDONE
