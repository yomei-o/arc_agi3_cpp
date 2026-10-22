@echo off
rem fp10k.bat : FPNet 10226-bit fingerprints (cpp/fpown/fp10226.py) of every candidate in the p37 gsdump runs -> the FPNet
rem   pool term in py/pool38.py. Pool candidates: pretrain\p36pool_mols.txt (119,046), generated: pretrain\g{x,b,bs}_mols.txt.
rem   Output C:\prog\casmi\models38\{p36pool,gx,gb,gbs}_fp10226.bin (+ .ok). 2 workers.
setlocal
cd /d C:\prog\casmi
set PY="C:\Program Files\Python312\python.exe"
for %%m in (gx gb gbs p36pool) do (
  if not exist models38\%%m_fp10226.bin %PY% C:\prog\casmi\models38\fp10226.py mols pretrain\%%m_mols.txt fpown\fp_bits.npy models38\%%m_fp10226 2
  echo %%m done %time%
)
echo all done
