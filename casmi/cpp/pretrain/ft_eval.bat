@echo off
rem ft_eval.bat <model stem in C:\prog\casmi, e.g. ft8_ep2> : window yardstick of a fine-tuned TFM1 fp3c16k model on hcqry/hxqry/haqry,
rem alone and z-sum ensembled with rev3_fp3c16k_big (MLP) + cnn_a_c (CNN); shufm=2 = the same ensembles with the fine-tuned
rem member's predictions taken from query q+7 (control). Output C:\prog\casmi\pretrain\fte_<stem>_<qry>.log
setlocal
cd /d C:\prog\casmi
set P=C:\prog\casmi\pretrain
set M=%~1
for %%q in (hcqry hxqry haqry) do (
  if not exist %P%\%M%_%%q_mol.bin %P%\pt_train.exe mollogits model=%M%.bin qry=%%q out=pretrain\%M%_%%q_mol.bin > %P%\fte_mol_%M%_%%q.log 2>&1
  %P%\we2\cnn_win_eval2.exe dir=C:\prog\casmi qry=%%q win=w_%%q "models=rev3_fp3c16k_big.bin:fp3c16k,cnn_a_c.bin:v4k,pretrain\%M%_%%q_mol.bin:fp3c16k" "ensw=1,1,0;0.5,1,0;1,0,1;0,1,1;1,1,1;0.5,1,1;0.5,1,0.5;0.5,1,2" shufm=2 threads=3 > %P%\fte_%M%_%%q.log 2>&1
)
