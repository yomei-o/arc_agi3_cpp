@echo off
rem sub_v55.bat : build-machine watcher for kernel v55 (auto_sub_remote.py), launched with night\launch.ps1 sub_v55
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v55 1 out_v55 "v55 (kernel v55 v1): v47 (own C++ GLACIER/ICEBERG, host-approved public ms-pred weights, same-formula group re-rank, ice 1.0 / gl 1.0, T4) group base z(1/(rank+10)) (v54 1/(rank+6) LB 0.368) instead of z(1/(rank+1))" "FW stage done"
