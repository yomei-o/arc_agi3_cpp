@echo off
rem rm1_hc.bat : hcqry with the truth REMOVED from the pool (rm=1; the answer can only come from generation or the
rem PubChem channel), v37 production configuration with tdump (tune38\run38.bat, holdout-safe wlib/ylib models),
rem for calibrating the PubChem promotion / Boltzmann merge offline (tune38_core W["pi"], W["a"]) against the normal
rem hcqry dump t38c. Output C:\prog\casmi\tune38\out\t38c_rm1*.
setlocal
set O=,pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2,alloc=boltz,bT1=3,bT2=0.5,bpi=0.1,ba=0.5,cnnmodel=cnn_a_c.bin,cnnpool=2,cnninst=0,ftmodel=ft8_ep5.bin,ftpool=2,rm=1
call C:\prog\casmi\tune38\run38.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_w.bin t38c_rm1 6 h2o\own_ep5_hcqry_fp.txt "%O%"
echo ALLDONE rm1_hc
