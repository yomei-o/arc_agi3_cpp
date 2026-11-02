@echo off
rem sub_v53b.bat : build-machine watcher for kernel v53b (auto_sub_remote.py), launched with night\launch.ps1 sub_v53b
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v53b 1 out_v53b "v53b (kernel v53b v1): v47 (own C++ GLACIER/ICEBERG, host-approved public ms-pred weights, same-formula group re-rank, ice 1.0 / gl 1.0, T4) + gated E2R2: library top-1 at rank 2 when best library score >= 1.0 (rf_STATUS holdout +0.008)" "FW stage done"
