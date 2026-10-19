@echo off
call C:\prog\casmi\fwdmodel\lm29\run.bat hasqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin has0 "" "" 5 lo
call C:\prog\casmi\fwdmodel\lm29\run.bat haqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin ha0 "" "" 5 lo
call C:\prog\casmi\fwdmodel\lm29\run.bat haqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin ha1 ",rm=1" "" 5 lo
call C:\prog\casmi\fwdmodel\lm29\run.bat hxqry zlib fwd_mlp_rank_ylib.bin rev2_z.bin hx0 "" "" 5 lo
echo ALLDONE A
