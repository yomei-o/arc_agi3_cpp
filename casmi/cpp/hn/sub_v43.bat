@echo off
rem sub_v43.bat : build-machine watcher for kernel v43 (auto_sub_remote.py), launched with night\launch.ps1 sub_v43
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v43 1 out_v43 "v43 (kernel v43 v1): v40 + own C++ GLACIER/ICEBERG inference (host-approved public ms-pred weights) re-scoring same-formula groups of the pool list, reciprocal-rank base, ice 0.5 / gl 1.0" "FW stage done"
