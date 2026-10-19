@echo off
rem holdout2: holdout-safe reverse models on ylib (alllib minus h2 queries / denovo molecules)
cd /d C:\prog\casmi\fwdmodel\rdkit-min\build
fp_prep.exe C:\prog\casmi ylib > C:\prog\casmi\h2\gpu.log 2>&1
cd /d C:\prog\casmi\fwdmodel\rev2
fp2_prep.exe apply C:\prog\casmi ylib v4k.bin >> C:\prog\casmi\h2\gpu.log 2>&1
cd /d C:\prog\casmi\fwdmodel\rev
rev_train_ens.exe train C:\prog\casmi ylib 15 512 0.0006 rev_y3072.bin 0.4 1.0 cqry 10 3072 31 >> C:\prog\casmi\h2\gpu.log 2>&1
cd /d C:\prog\casmi
C:\prog\casmi\fwdmodel\rev2\rev2_train.exe train lib=ylib fp=v4k out=rev2_y.bin epochs=12 hold=20 >> C:\prog\casmi\h2\gpu.log 2>&1
echo GPUDONE >> C:\prog\casmi\h2\gpu.log
