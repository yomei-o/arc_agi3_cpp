@echo off
rem sub_v40.bat : build-machine watcher for kernel v40 (auto_sub_remote.py), launched with night\launch.ps1 sub_v40
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\auto_sub_remote.py casmi26-cpp-library-search-v40 1 out_v40 "v40 (kernel v40 v1): v36nofp with the own FPNet pool term in continuous-Tanimoto form (p=sigmoid(logit)) instead of the logit dot product; e2e hc 0.5804->0.5823, hx 0.8908->0.8918" "z(fpnet, tanimoto)"
