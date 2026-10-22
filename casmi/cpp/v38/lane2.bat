@echo off
rem speed lane 2 (2 threads): hasqry, same options; phys_v37pc reference then phys_v38pc (caches on)
set O=,pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2,alloc=boltz,bT1=3,bT2=0.5,bpi=0.1,ba=0.5,cnnmodel=cnn_a_c.bin,cnnpool=2,ftmodel=ft8_ep5.bin,ftpool=2
call C:\prog\casmi\v37\run37.bat hasqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin s37_ft2 2 "%O%,cfdump=v37\out\s37_ft2_cf.txt"
call C:\prog\casmi\v38\run38.bat hasqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin s38_on 2 "%O%,cfdump=v38\out\s38_on_cf.txt"
for %%e in (.csv _ins.csv _det.txt _pp.txt _cf.txt) do (
  fc /b C:\prog\casmi\v37\out\s37_ft2%%e C:\prog\casmi\v38\out\s38_on%%e > nul && echo SAME s38_on%%e || echo DIFF s38_on%%e
)
echo ALLDONE
