@echo off
rem sub_v46b.bat : build-machine watcher for kernel v46b (auto_sub_remote.py), launched with night\launch.ps1 sub_v46b
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v46b 1 out_v46b "v46b (kernel v46b v1): v40 + own C++ GLACIER/ICEBERG inference (host-approved public ms-pred weights) as fused-score terms on a T4 (nvcc build; v46 fell back to the CPU, GPU check fixed): z(score + 0.15 pop) + 0.5 z(gl) + 0.5 z(ice); e2e hcqry +0.022 hxqry +0.006" "FW stage done"
