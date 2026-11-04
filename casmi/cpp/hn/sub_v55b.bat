@echo off
rem sub_v55b.bat : build-machine watcher for kernel v55b (auto_sub_remote.py), launched with night\launch.ps1 sub_v55b
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v55b 1 out_v55b "v55b (kernel v55b v1): v47 (own C++ GLACIER/ICEBERG, host-approved public ms-pred weights, same-formula group re-rank, ice 1.0 / gl 1.0, T4) group base z(1/(rank+20)) (v54 1/(rank+6) LB 0.368) instead of z(1/(rank+1))" "FW stage done"
