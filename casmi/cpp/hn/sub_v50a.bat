@echo off
rem sub_v50a.bat : build-machine watcher for kernel v50a (auto_sub_remote.py), launched with night\launch.ps1 sub_v50a
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v50a 1 out_v50a "v50a (kernel v50a v1): v40 + own C++ GLACIER/ICEBERG inference (host-approved public ms-pred weights) re-scoring same-formula groups of the pool list, reciprocal-rank base, ice 1.0 / gl 0.5 on a T4 (GPU fw_score, full ICE coverage)" "FW stage done"
