@echo off
rem sub_v48a.bat : build-machine watcher for kernel v48a (auto_sub_remote.py), launched with night\launch.ps1 sub_v48a
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v48a 1 out_v48a "v48a (kernel v48a v1): v40 + own C++ GLACIER/ICEBERG inference (host-approved public ms-pred weights) re-scoring same-formula groups of the pool list, reciprocal-rank base, ice 2.0 / gl 1.0 on a T4 (GPU fw_score, full ICE coverage)" "FW stage done"
