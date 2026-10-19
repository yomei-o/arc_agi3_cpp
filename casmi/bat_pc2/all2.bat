@echo off
rem second pass: PubChem list ordered by z(hand fusion) + pcw pop (pcrank=fused)
call C:\prog\casmi\fwdmodel\pc2\run.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hcF0 ",pcrank=fused"
call C:\prog\casmi\fwdmodel\pc2\run.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hcF1 ",rm=1,pcrank=fused"
call C:\prog\casmi\fwdmodel\pc2\run.bat haqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin haF0 ",pcrank=fused"
call C:\prog\casmi\fwdmodel\pc2\run.bat haqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin haF1 ",rm=1,pcrank=fused"
call C:\prog\casmi\fwdmodel\pc2\run.bat tqry xlib fwd_mlp_rank_xlib.bin rev2_vh.bin tF0 ",pcrank=fused"
call C:\prog\casmi\fwdmodel\pc2\run.bat tqry xlib fwd_mlp_rank_xlib.bin rev2_vh.bin tF1 ",rm=1,pcrank=fused"
call C:\prog\casmi\fwdmodel\pc2\run.bat cqry xlib fwd_mlp_rank_xlib.bin rev2_vh.bin cF0 ",pcrank=fused"
call C:\prog\casmi\fwdmodel\pc2\run.bat cqry xlib fwd_mlp_rank_xlib.bin rev2_vh.bin cF1 ",rm=1,pcrank=fused"
echo ALLDONE
