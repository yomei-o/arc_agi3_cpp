@echo off
rem sub_v52.bat : build-machine watcher for kernel v52 (auto_sub_remote.py), launched with night\launch.ps1 sub_v52
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v52 1 out_v52 "v52 (kernel v52 v1): v40 + own C++ GLACIER/ICEBERG (host-approved public ms-pred weights) same-formula group re-rank, ice 1.0 / gl 1.0, T4; group base = z(ranker score + 0.15 pop) x 1.0 instead of z(1/(rank+1))" "FW stage done"
