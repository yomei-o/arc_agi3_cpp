@echo off
rem third pass: the old channel (pcv=1, configuration B of pcslot_STATUS: score insertion) on the new base, for comparison;
rem then the chosen v2 configuration straight from C++ (verifies pc2_grid.py's simulation)
call C:\prog\casmi\fwdmodel\pc2\run.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hcB0 ",pcv=1"
call C:\prog\casmi\fwdmodel\pc2\run.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hcB1 ",rm=1,pcv=1"
call C:\prog\casmi\fwdmodel\pc2\run.bat haqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin haB0 ",pcv=1"
call C:\prog\casmi\fwdmodel\pc2\run.bat haqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin haB1 ",rm=1,pcv=1"
call C:\prog\casmi\fwdmodel\pc2\run.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hcV0 ",pcrank=fused,relhi=6,relp=3,sp=6,ppop=0.15,ppgate=0.9"
call C:\prog\casmi\fwdmodel\pc2\run.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hcV1 ",rm=1,pcrank=fused,relhi=6,relp=3,sp=6,ppop=0.15,ppgate=0.9"
echo ALLDONE
