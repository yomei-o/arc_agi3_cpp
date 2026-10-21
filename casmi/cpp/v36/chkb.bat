@echo off
rem pool36: C++ check of the offline simulation, hbqry with cnnpool=2
call C:\prog\casmi\v36\run36.bat hbqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin p36b_cnn2 2 ",pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2,alloc=boltz,bT1=2,bT2=0.5,bpi=0.2,ba=0.5,cnnmodel=cnn_a_c.bin,cnnpool=2"
echo ALLDONE
