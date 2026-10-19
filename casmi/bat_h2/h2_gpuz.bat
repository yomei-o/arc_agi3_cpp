@echo off
rem holdout2: reverse models on zlib (= ylib minus every timsTOF spectrum of the xinstr holdout hxqry)
cd /d C:\prog\casmi\fwdmodel\rdkit-min\build
fp_prep.exe C:\prog\casmi zlib > C:\prog\casmi\h2\gpuz.log 2>&1
cd /d C:\prog\casmi\fwdmodel\rev2
fp2_prep.exe apply C:\prog\casmi zlib v4k.bin >> C:\prog\casmi\h2\gpuz.log 2>&1
cd /d C:\prog\casmi
C:\prog\casmi\fwdmodel\rev2\rev2_train.exe train lib=zlib fp=v4k out=rev2_z.bin epochs=12 hold=20 >> C:\prog\casmi\h2\gpuz.log 2>&1
cd /d C:\prog\casmi\fwdmodel\rev
rev_train_ens.exe train C:\prog\casmi zlib 15 512 0.0006 rev_z3072.bin 0.4 1.0 cqry 10 3072 31 >> C:\prog\casmi\h2\gpuz.log 2>&1
echo GPUZDONE >> C:\prog\casmi\h2\gpuz.log
