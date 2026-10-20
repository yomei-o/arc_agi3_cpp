@echo off
rem lane G: production configuration smoke test on the visible test set (train library, fwd_mlp_rank_v2, rev2_all_v4k,
rem pcfp = the production model rev3_all_fp3c16k_big.bin) - load / timing check only (no truth; mrr.py fails harmlessly)
call C:\prog\casmi\fwdmodel\pc4\run4.bat test train fwd_mlp_rank_v2.bin rev2_all_v4k.bin test32 phys_v32pc 2 ",pcrev=tan,pcfp=rev3_all_fp3c16k_big.bin"
echo ALLDONE
