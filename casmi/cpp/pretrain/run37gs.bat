@echo off
rem run37gs.bat <qry> <lib> <mlp> <rev2> <tag> <threads> <extra opts> : cpp/v36/run36.bat with phys_v36gs.exe (= phys_v36pc +
rem   gsdump=: every candidate's SMILES incl. the generated G<k> ones) and outputs in C:\prog\casmi\pretrain\p37\out\ (own dir).
rem   Used to give the generated candidates a real fine-tuned-transformer Tanimoto in py/pool37ft.py (pretrain_STATUS.md 8).
setlocal
cd /d C:\prog\casmi
C:\prog\casmi\pretrain\p37\phys_v36gs.exe eval C:\prog\casmi %~1 pool730 10 phys_size_fqry.txt pretrain\p37\out\%~5.csv %~3 0.25 "" %~4 1.0 %~2 8 32 24 "" 4 3 8000 "" "" 1 "gen=1,cap=300,prior=gen2o\gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=%~6,grid=0,libcap=1,ppopf=pc2\pool730_pop.bin,ppdet=pretrain\p37\out\%~5_pp.txt,pc=1,pcv=2,pcout=pretrain\p37\out\%~5_ins.csv,pcdet=pretrain\p37\out\%~5_det.txt,pcrank=fused,relhi=6,relp=3,sp=6,ppop=0.15,ppgate=0.9%~7" > pretrain\p37\out\%~5.log 2>&1
echo done %~5
