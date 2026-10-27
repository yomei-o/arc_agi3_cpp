@echo off
rem sub_v44.bat : build-machine watcher for kernel v44 (auto_sub_remote.py), launched with night\launch.ps1 sub_v44
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v44 1 out_v44 "v44 (kernel v44 v1): v40 + own C++ GLACIER inference (host-approved public ms-pred weights) GLACIER as a fused-score term: z(score + 0.15 pop) + 0.5 z(gl), [M+H]+ only (iceberg_STATUS add_w0.5_mh)" "FW stage done"
