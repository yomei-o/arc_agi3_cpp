@echo off
rem pool36 shuffled-CNN control (cnnshuf=7: query q gets query q+7 CNN p), dumps only
call C:\prog\casmi\v36\run36.bat hxqry zlib fwd_mlp_rank_ylib.bin rev2_z.bin p36x_0_shuf 2 ",pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2,alloc=boltz,bT1=2,bT2=0.5,bpi=0.2,ba=0.5,cnnmodel=cnn_a_c.bin,cnnshuf=7,cfdump=v36\out\p36x_0_shuf_cf.txt"
echo ALLDONE
