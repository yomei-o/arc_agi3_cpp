@echo off
rem sub_v54.bat : build-machine watcher for kernel v54 (auto_sub_remote.py), launched with night\launch.ps1 sub_v54
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v54 1 out_v54 "v54 (kernel v54 v1): v47 (own C++ GLACIER/ICEBERG, host-approved public ms-pred weights, same-formula group re-rank, ice 1.0 / gl 1.0, T4) group base z(1/(rank+6)) (flatter than v53 1/(rank+3), LB 0.365) instead of z(1/(rank+1))" "FW stage done"
