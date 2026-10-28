@echo off
rem sub_v45.bat : build-machine watcher for kernel v45 (auto_sub_remote.py), launched with night\launch.ps1 sub_v45
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v45 1 out_v45 "v45 (kernel v45 v1): v40 + own C++ GLACIER/ICEBERG inference (host-approved public ms-pred weights) re-scoring same-formula groups of the pool list, reciprocal-rank base, ice 0.5 / gl 1.0, ICE budget 18000s (v43 skipped 26%% of ICE predictions at 10800s)" "FW stage done"
