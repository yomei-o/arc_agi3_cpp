@echo off
rem lane E: v30 pcrev=tan, answer removed, PubChem list ranked by z(rev)+pcw pop only (pcrank=rev)
call C:\prog\casmi\fwdmodel\pc4\run4.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hc30r_1 phys_v30pc 2 ",rm=1,pcrev=tan,pcrank=rev"
echo ALLDONE
