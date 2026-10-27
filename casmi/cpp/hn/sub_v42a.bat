@echo off
rem sub_v42a.bat : build-machine watcher for kernel v42a (auto_sub_remote.py), launched with night\launch.ps1 sub_v42a
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v42a 1 out_v42a "v42a (kernel v42a v1): v40 with the pool popularity re-order (0.15 x documentation) applied to every query (ppgate off; public lineage +0.010)" "z(fpnet, tanimoto)"
