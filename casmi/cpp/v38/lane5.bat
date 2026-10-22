@echo off
rem speed lane 5 (4 threads): final source (v38\c, pccachemax 3M default, poolcache off) on hbqry vs phys_v37pc c37b_ft2
set O=,pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2,alloc=boltz,bT1=2,bT2=0.5,bpi=0.2,ba=0.5,cnnmodel=cnn_a_c.bin,cnnpool=2,ftmodel=ft8_ep5.bin,ftpool=2
call C:\prog\casmi\v38\run38c.bat hbqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin b38_c3m 4 "%O%,ftdump=v38\out\b38_c3m_mol.bin,cfdump=v38\out\b38_c3m_cf.txt"
for %%e in (.csv _ins.csv _det.txt _pp.txt _cf.txt _mol.bin) do (
  fc /b C:\prog\casmi\v37\out\c37b_ft2%%e C:\prog\casmi\v38\out\b38_c3m%%e > nul && echo SAME b38_c3m%%e || echo DIFF b38_c3m%%e
)
echo ALLDONE
