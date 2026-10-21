@echo off
rem window MRR of an FPNet-format weights file on hcqry and hxqry (holdout-safe only for wlib-trained weights)
rem usage: eval_win.bat <tag> <weights.bin (FPN1/FPN2)>     -> C:\prog\casmi\fpown\win_<tag>.txt
rem molecule logits via the deployed fpnet_ggml (C:\prog\casmi\fpown\gg), scored by win_eval with the 10226-bit window fingerprints;
rem rev2_w.bin (wlib, v4k) printed alongside as the clean reference
setlocal
cd /d C:\prog\casmi\fpown
set GG=C:\prog\casmi\fpown\gg\build\fpnet_ggml.exe
set WE=C:\prog\casmi\fpown\build\win_eval.exe
set OUT=C:\prog\casmi\fpown\win_%~1.txt
echo weights %~2 > %OUT%
for %%q in (hcqry hxqry) do (
  if not exist C:\prog\casmi\fpown\gg_%~1_%%q_mol.bin %GG% %~2 C:\prog\casmi %%q C:\prog\casmi\fpown\gg_%~1_%%q 4
  %WE% dir=C:\prog\casmi qry=%%q win=w_%%q "models=fpown\gg_%~1_%%q_mol.bin:fp10226,rev2_w.bin:v4k" threads=4 >> %OUT%
)
type %OUT%
