@echo off
rem night_hn_cpu.bat : holdout-safe models for hnqry (enveda-np-examples, denovo) on nlib = wlib minus those 250 molecules.
rem CPU chain: library bits (v4k, fp3c16k, fp10226) -> REV2 (v4k) -> fp3 big (fp3c16k) -> forward MLP.
rem Launch detached: powershell -ExecutionPolicy Bypass -File C:\prog\casmi\night\launch.ps1 hn_cpu C:\prog\casmi\hn\night_hn_cpu.bat
setlocal
set D=C:\prog\casmi
set R=C:\prog\casmi\fwdmodel\rev3
set PY="C:\Program Files\Python312\python.exe"
set O=C:\prog\casmi\hn
set T=6
cd /d %D%
echo [%date% %time%] start cpu chain > %O%\cpu_progress.txt
if not exist %D%\nlib_head.bin (echo nlib_head.bin missing >> %O%\cpu_progress.txt & exit /b 1)

rem ---- 1. library bits in nlib_mols.txt order (same vocabularies as wlib)
if not exist %D%\nlib_v4k.bin %R%\fp3_prep.exe apply %D% nlib_mols.txt v4k.bin nlib_v4k.bin %T% > %O%\apply_nlib_v4k.log 2>&1
echo [%date% %time%] nlib_v4k done >> %O%\cpu_progress.txt
if not exist %D%\nlib_fp3c16k.bin %R%\fp3_prep.exe apply %D% nlib_mols.txt fp3c16k.bin nlib_fp3c16k.bin %T% > %O%\apply_nlib_fp3c16k.log 2>&1
echo [%date% %time%] nlib_fp3c16k done >> %O%\cpu_progress.txt
if not exist %D%\nlib_fp10226.bin %PY% %D%\fpown\fp10226.py mols %D%\nlib_mols.txt %D%\fpown\fp_bits.npy %D%\nlib_fp10226 4 > %O%\fp10226_nlib.log 2>&1
echo [%date% %time%] nlib_fp10226 done >> %O%\cpu_progress.txt
echo bits ok > %O%\bits.done

rem ---- 2. REV2 on nlib (as rev2_w: v4k, 12 epochs, hold 20)
if not exist %D%\rev2_n.bin %R%\rev3_train.exe train dir=%D% lib=nlib fp=v4k out=rev2_n.bin epochs=12 hold=20 eval=- threads=%T% > %O%\train_rev2_n.log 2>&1
echo [%date% %time%] rev2_n done >> %O%\cpu_progress.txt

rem ---- 3. fp3 big on nlib (as rev3_fp3c16k_big: 20 epochs, H=4096,4096)
if not exist %D%\rev3_n_fp3c16k_big.bin %R%\rev3_train.exe train dir=%D% lib=nlib fp=fp3c16k out=rev3_n_fp3c16k_big.bin epochs=20 H=4096,4096 hold=20 eval=- threads=%T% > %O%\train_rev3_n_big.log 2>&1
echo [%date% %time%] rev3_n_fp3c16k_big done >> %O%\cpu_progress.txt

rem ---- 4. forward MLP on nlib (as bat_h2\h2_mlp.bat for ylib)
cd /d C:\prog\casmi\fwdmodel\rdkit-min\build
if not exist %D%\fwd_mlp_nlib.bin train_forward_model_nn.exe %D% nlib 20000 32 30000 256 0.002 fwd_mlp_nlib.bin 0.05 > %O%\mlp1.log 2>&1
if not exist %D%\fwd_mlp_rank_nlib.bin train_forward_model_rank.exe %D% nlib fwd_mlp_nlib.bin 3000 20 5 0.1 5000 0.0002 fwd_mlp_rank_nlib.bin > %O%\mlp2.log 2>&1
echo [%date% %time%] mlp done >> %O%\cpu_progress.txt
echo [%date% %time%] cpu all done >> %O%\cpu_progress.txt
