@echo off
rem sub_v48b.bat : build-machine watcher for kernel v48b (auto_sub_remote.py), launched with night\launch.ps1 sub_v48b
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v48b 1 out_v48b "v48b (kernel v48b v1): v40 + own C++ GLACIER/ICEBERG inference (host-approved public ms-pred weights) re-scoring same-formula groups of the pool list, reciprocal-rank base, ice 3.0 / gl 1.0 on a T4 (GPU fw_score, full ICE coverage)" "FW stage done"
