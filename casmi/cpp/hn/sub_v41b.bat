@echo off
rem sub_v41b.bat : build-machine watcher for kernel v41b (auto_sub_remote.py), launched with night\launch.ps1 sub_v41b
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v41b 1 out_v41b "v41b (kernel v41b v1): v40 + ICEBERG 0.5 / GLACIER 1.0 same-formula re-scoring of the pool list (host-approved public ms-pred forward models), reciprocal-rank base scores; e2e hc +0.009 hx 0" "ICE/GL stage done"
