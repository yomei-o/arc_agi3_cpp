@echo off
rem gs3.bat : phys_v36gs runs for hxqry / hbqry / hbsqry with the pool36 options (lane1-3) + gsdump, 2 threads each, sequential
set R=C:\prog\casmi\pretrain\run37gs.bat
set O=,pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2,alloc=boltz,bT1=2,bT2=0.5,bpi=0.2,ba=0.5,cnnmodel=cnn_a_c.bin
call %R% hxqry zlib fwd_mlp_rank_ylib.bin rev2_z.bin p36x_0 2 "%O%,cfdump=pretrain\p37\out\p36x_0_cf.txt,gsdump=pretrain\p37\out\p36x_0_gs.txt"
call %R% hbqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin p36b_0 2 "%O%,cfdump=pretrain\p37\out\p36b_0_cf.txt,gsdump=pretrain\p37\out\p36b_0_gs.txt"
call %R% hbsqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin p36bs_0 2 "%O%,cfdump=pretrain\p37\out\p36bs_0_cf.txt,gsdump=pretrain\p37\out\p36bs_0_gs.txt"
echo ALLDONE
