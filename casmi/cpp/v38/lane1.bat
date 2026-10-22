@echo off
rem speed lane 1 (4 threads): hbqry chk37b config; v38 with caches off (profile of the v37 path) then caches on.
rem reference = phys_v37pc output v37\out\c37b_ft2* (cpp/v37/chk37.bat b)
set O=,pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2,alloc=boltz,bT1=2,bT2=0.5,bpi=0.2,ba=0.5,cnnmodel=cnn_a_c.bin,cnnpool=2,ftmodel=ft8_ep5.bin,ftpool=2
call C:\prog\casmi\v38\run38.bat hbqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin b38_off 4 "%O%,ftdump=v38\out\b38_off_mol.bin,cfdump=v38\out\b38_off_cf.txt,poolcache=0,pccache=0"
call C:\prog\casmi\v38\run38.bat hbqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin b38_on 4 "%O%,ftdump=v38\out\b38_on_mol.bin,cfdump=v38\out\b38_on_cf.txt"
for %%t in (b38_off b38_on) do (
  for %%e in (.csv _ins.csv _det.txt _pp.txt _cf.txt _mol.bin) do (
    fc /b C:\prog\casmi\v37\out\c37b_ft2%%e C:\prog\casmi\v38\out\%%t%%e > nul && echo SAME %%t%%e || echo DIFF %%t%%e
  )
)
echo ALLDONE
