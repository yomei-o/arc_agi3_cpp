@echo off
rem sub_v47.bat : build-machine watcher for kernel v47 (auto_sub_remote.py), launched with night\launch.ps1 sub_v47
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v47 1 out_v47 "v47 (kernel v47 v1): v40 + own C++ GLACIER/ICEBERG inference (host-approved public ms-pred weights) re-scoring same-formula groups of the pool list, reciprocal-rank base, ice 1.0 / gl 1.0 on a T4 (GPU fw_score, full ICE coverage)" "FW stage done"
