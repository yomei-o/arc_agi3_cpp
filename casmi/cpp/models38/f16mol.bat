@echo off
rem f16mol.bat : CPU (AVX2, 4 threads = Kaggle-like) molecule logits of cnn_f16 (CNN on fp3c16k bits, wlib) and cnn_a (timing
rem   reference) on hxqry/hbqry/hbsqry -> C:\prog\casmi\models38\cnnf16_<q>_mol.bin (pool38 tan: column with fp3c16k bits).
setlocal
cd /d C:\prog\casmi
set X=C:\prog\casmi\models38\cnnb\cnn_win_eval_cpu.exe
for %%q in (hxqry hbqry hbsqry) do (
  %X% dir=C:\prog\casmi qry=%%q models=cnn_f16.bin:fp3c16k molout=models38\cnnf16 threads=4
)
%X% dir=C:\prog\casmi qry=hxqry models=cnn_a.bin:v4k molout=models38\cnna threads=4
echo all done
