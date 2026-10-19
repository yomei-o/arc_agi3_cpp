@echo off
rem holdout2: holdout-safe rev v1 (3072 hidden, ECFP 2048) on ylib, same command as rev_x.bat
cd /d C:\prog\casmi\fwdmodel\rev
rev_train_ens.exe train C:\prog\casmi ylib 15 512 0.0006 rev_y3072.bin 0.4 1.0 cqry 10 3072 31 > C:\prog\casmi\h2\rev1.log 2>&1
echo REV1DONE >> C:\prog\casmi\h2\rev1.log
