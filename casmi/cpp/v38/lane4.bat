@echo off
rem speed lane 4 (2 threads): final phys_v38pc on hasqry with pccachemax=1000000 (exercises the full-cache path) vs phys_v37pc s37_ft2
set O=,pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2,alloc=boltz,bT1=3,bT2=0.5,bpi=0.1,ba=0.5,cnnmodel=cnn_a_c.bin,cnnpool=2,ftmodel=ft8_ep5.bin,ftpool=2
call C:\prog\casmi\v38\run38b.bat hasqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin s38_cap 2 "%O%,cfdump=v38\out\s38_cap_cf.txt,pccachemax=1000000"
for %%e in (.csv _ins.csv _det.txt _pp.txt _cf.txt) do (
  fc /b C:\prog\casmi\v37\out\s37_ft2%%e C:\prog\casmi\v38\out\s38_cap%%e > nul && echo SAME s38_cap%%e || echo DIFF s38_cap%%e
)
echo ALLDONE
