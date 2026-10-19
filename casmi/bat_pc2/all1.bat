@echo off
rem first measurement pass: v27a config + PubChem v2 dumps on the denovo holdouts (normal and rm=1) and one sibling holdout
call C:\prog\casmi\fwdmodel\pc2\run.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hc0 ""
call C:\prog\casmi\fwdmodel\pc2\run.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hc1 ",rm=1"
call C:\prog\casmi\fwdmodel\pc2\run.bat haqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin ha0 ""
call C:\prog\casmi\fwdmodel\pc2\run.bat haqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin ha1 ",rm=1"
call C:\prog\casmi\fwdmodel\pc2\run.bat tqry xlib fwd_mlp_rank_xlib.bin rev2_vh.bin t0 ""
call C:\prog\casmi\fwdmodel\pc2\run.bat tqry xlib fwd_mlp_rank_xlib.bin rev2_vh.bin t1 ",rm=1"
call C:\prog\casmi\fwdmodel\pc2\run.bat cqry xlib fwd_mlp_rank_xlib.bin rev2_vh.bin c0 ""
call C:\prog\casmi\fwdmodel\pc2\run.bat cqry xlib fwd_mlp_rank_xlib.bin rev2_vh.bin c1 ",rm=1"
call C:\prog\casmi\fwdmodel\pc2\run.bat hasqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin has0 ""
echo ALLDONE
