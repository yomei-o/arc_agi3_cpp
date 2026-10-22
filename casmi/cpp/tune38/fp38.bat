@echo off
rem fp38.bat : own FPNet (wlib-trained fpown_w_f16.bin = the e2e stand-in, holdout-safe for the h* sets) pool scores
rem   for hbqry / hbsqry (hxqry/hcqry/hasqry exist: h2o\own_ep5_<qry>_fp.txt) -> C:\prog\casmi\tune38\fp_<qry>.txt
setlocal
cd /d C:\prog\casmi
for %%q in (hbqry hbsqry) do (
  if not exist C:\prog\casmi\tune38\fp_%%q_mol.bin C:\prog\casmi\fpown\gg\build\fpnet_ggml.exe C:\prog\casmi\fpown\fpown_w_f16.bin C:\prog\casmi %%q C:\prog\casmi\tune38\fp_%%q 2
  if not exist C:\prog\casmi\tune38\fp_%%q.txt C:\prog\casmi\fpnet\kbuild\fpnet_pool.exe C:\prog\casmi %%q C:\prog\casmi\tune38\fp_%%q_mol.bin pool730 C:\prog\casmi\pool730_fp10226.bin 10 C:\prog\casmi\tune38\fp_%%q.txt
)
echo ALLDONE
