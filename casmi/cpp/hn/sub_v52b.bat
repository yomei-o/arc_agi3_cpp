@echo off
rem sub_v52b.bat : build-machine watcher for kernel v52b (auto_sub_remote.py), launched with night\launch.ps1 sub_v52b
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v52b 1 out_v52b "v52b (kernel v52b v1): v40 + own C++ GLACIER/ICEBERG (host-approved public ms-pred weights) same-formula group re-rank, ice 1.0 / gl 1.0, T4; group base = z(ranker score + 0.15 pop) x 2.0 instead of z(1/(rank+1))" "FW stage done"
