@echo off
rem build_gpu.bat : fw_score with the cuBLAS GEMM path (gpu_linear.cu) and the device-resident ICEBERG GGNN
rem (gpu_ggnn.cu) on the build machine (MSVC + CUDA toolkit). Also builds ice_check_gpu.exe (5th arg "gpu").
setlocal
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
set CU=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set RM=C:\prog\casmi\fwdmodel\rdkit-min
set RB=%RM%\build
set G=C:\prog\casmi\glacier
cd /d %G%
"%CU%\bin\nvcc.exe" -O3 -c gpu_linear.cu -o gpu_linear.obj -Xcompiler /MD
echo nvcc rc %errorlevel%
"%CU%\bin\nvcc.exe" -O3 -arch=sm_86 -c gpu_ggnn.cu -o gpu_ggnn.obj -Xcompiler /MD -I%G%
echo nvcc ggnn rc %errorlevel%
set INC=/I%RM% /I%RM%\GraphMol /IC:\prog\vcpkg\installed\x64-windows\include /I%G% /I"%CU%\include"
set LIBS=%RB%\SmilesParse.lib %RB%\CIPLabeler.lib %RB%\GraphMol.lib %RB%\Geometry.lib %RB%\DataStructs.lib %RB%\RDGeneral.lib
cl /nologo /std:c++20 /O2 /EHsc /MD /DNDEBUG /DFW_GPU /DYY_NO_UNISTD_H /bigobj /wd4819 /arch:AVX2 /fp:fast %INC% fw_score.cpp gpu_linear.obj gpu_ggnn.obj %G%\pc\GasteigerCharges.cpp %G%\pc\GasteigerParams.cpp %LIBS% /Fe:fw_score_gpu.exe /link /LIBPATH:C:\prog\vcpkg\installed\x64-windows\lib /LIBPATH:"%CU%\lib\x64" cublas.lib cudart.lib
echo build rc %errorlevel%
if "%ICECHECK%"=="0" goto :eof
cl /nologo /std:c++20 /O2 /EHsc /MD /DNDEBUG /DFW_GPU /DYY_NO_UNISTD_H /bigobj /wd4819 %INC% ice_check.cpp gpu_linear.obj gpu_ggnn.obj %G%\pc\GasteigerCharges.cpp %G%\pc\GasteigerParams.cpp %LIBS% /Fe:ice_check_gpu.exe /link /LIBPATH:C:\prog\vcpkg\installed\x64-windows\lib /LIBPATH:"%CU%\lib\x64" cublas.lib cudart.lib
echo build ice_check_gpu rc %errorlevel%
