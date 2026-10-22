@echo off
rem fpn_eval.bat <stem> : FPNet-format export of C:\prog\casmi\<stem>.bin (TFM1, fpnet=1) and molecule logits on the timsTOF
rem   holdouts with the notebook's fpnet_ggml (2 threads) -> C:\prog\casmi\models38\fpn\<stem>_{hxqry,hbqry,hbsqry}_mol.bin
rem   (input of py/pool38.py dot: columns). Skips what exists. f32 export deleted after (only the f16 file is kept).
setlocal
cd /d C:\prog\casmi
set F=C:\prog\casmi\models38\fpn
set GG=C:\prog\casmi\fpown\gg\build\fpnet_ggml.exe
if not exist %F%\%~1_f16.bin C:\prog\casmi\fpown\build\tfm2fpnet.exe C:\prog\casmi\%~1.bin %F%\%~1_f32.bin %F%\%~1_f16.bin > %F%\%~1_export.log 2>&1
if exist %F%\%~1_f32.bin del %F%\%~1_f32.bin
for %%q in (hxqry hbqry hbsqry) do (
  if not exist %F%\%~1_%%q_mol.bin %GG% %F%\%~1_f16.bin C:\prog\casmi %%q %F%\%~1_%%q 2 > %F%\%~1_%%q.log 2>&1
)
echo eval %~1 done %date% %time%
