@echo off
rem sub_v49.bat : build-machine watcher for kernel v49 (auto_sub_remote.py), launched with night\launch.ps1 sub_v49
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v49 1 out_v49 "v49 (kernel v49 v1): v40 + own C++ GLACIER/ICEBERG inference (host-approved public ms-pred weights) re-scoring same-formula groups of the pool list, reciprocal-rank base, ice 1.0 / gl 1.0 on a T4 + PubChem rank-1 promotion (relz > 3 or (S > 6 and pop >= 5), libSim < 0.9)" "PC PROMO done"
