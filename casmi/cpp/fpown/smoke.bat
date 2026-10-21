@echo off
rem smoke test: 3-minute training on wlib, export, ggml forward vs our trainer forward on the first 5 hcqry molecules
cd /d C:\prog\casmi\fpown
set B=C:\prog\casmi\fpown\build
set PY="C:\Program Files\Python312\python.exe"
if not "%~1"=="skiptrain" %B%\tfm_train.exe train dir=C:\prog\casmi lib=wlib fp=fp10226 fpnet=1 out=fpown_smoke eval=none B=64 stopAfter=3 hold=20 threads=6 logEvery=100
%B%\tfm2fpnet.exe C:\prog\casmi\fpown_smoke_ck.bin C:\prog\casmi\fpown\smoke_f32.bin C:\prog\casmi\fpown\smoke_f16.bin
%PY% C:\prog\casmi\fpown\to_f16.py C:\prog\casmi\fpown\smoke_f32.bin C:\prog\casmi\fpown\smoke_f16_py.bin
fc /b C:\prog\casmi\fpown\smoke_f16.bin C:\prog\casmi\fpown\smoke_f16_py.bin > nul && echo F16 IDENTICAL TO to_f16.py || echo F16 DIFFERS FROM to_f16.py
%B%\tfm_train.exe dumpz dir=C:\prog\casmi model=fpown_smoke_ck.bin q=hcqry maxMols=5 out=fpown\smoke_dumpz.bin fpnet=1 tf32=0
C:\prog\casmi\fpown\gg\build\fpnet_ggml.exe C:\prog\casmi\fpown\smoke_f32.bin C:\prog\casmi hcqry C:\prog\casmi\fpown\gg32 4 5 1
C:\prog\casmi\fpown\gg\build\fpnet_ggml.exe C:\prog\casmi\fpown\smoke_f16.bin C:\prog\casmi hcqry C:\prog\casmi\fpown\gg16 4 5 1
echo == f32 file (ggml) vs trainer
%PY% cmp_spec.py gg32_spec.bin smoke_dumpz.bin
echo == f16 file (ggml) vs trainer
%PY% cmp_spec.py gg16_spec.bin smoke_dumpz.bin
echo smoke done
