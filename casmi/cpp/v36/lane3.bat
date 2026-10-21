@echo off
rem pool36 lane 3: hbsqry then hcqry rm=1
call C:\prog\casmi\v36\run36.bat hbsqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin p36bs_0 2 ",pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2,alloc=boltz,bT1=2,bT2=0.5,bpi=0.2,ba=0.5,cnnmodel=cnn_a_c.bin,cfdump=v36\out\p36bs_0_cf.txt"
call C:\prog\casmi\v36\run36.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin p36c_1 2 ",rm=1,pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2,alloc=boltz,bT1=2,bT2=0.5,bpi=0.2,ba=0.5,cnnmodel=cnn_a_c.bin,cfdump=v36\out\p36c_1_cf.txt"
echo ALLDONE
