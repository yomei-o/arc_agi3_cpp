@echo off
call C:\prog\casmi\fwdmodel\lm29\run.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hc0 "" "" 5 hi
call C:\prog\casmi\fwdmodel\lm29\run.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hc1 ",rm=1" "" 5 hi
call C:\prog\casmi\fwdmodel\lm29\run.bat hbsqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hbs0 "" "" 5 hi
call C:\prog\casmi\fwdmodel\lm29\run.bat hbqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hb0 "" "" 5 hi
call C:\prog\casmi\fwdmodel\lm29\run.bat hbqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hb1 ",rm=1" "" 5 hi
echo ALLDONE B
