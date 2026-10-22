@echo off
rem gridtest.bat : smoke test of the remote pool38 environment (hbqry, f5 only) - expected "cnn:2,f5:1 ... 0.8092 +0.0281 ... +0.0069"
cd /d C:\prog\casmi
set M=C:\prog\casmi\models38
set BITS=%M%\p36pool_fp10226.bin::pretrain\p36pool_mols.txt;%M%\g{s}_fp10226.bin::pretrain\g{s}_mols.txt
"C:\Program Files\Python312\python.exe" %M%\py\pool38.py %M%\D %M%\grid_test p36b_0:hbqry:b "cnn:2,f5:1" col=f5=dot:%M%\fpn\fpown_w_{q}_mol.bin#%BITS% "ref=cnn:2"
