@echo off
for /f "usebackq delims=" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set VS=%%i
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\prog\casmi\fpnet_src\cpp
if not exist ggml xcopy /e /i /q C:\prog\llama.cpp\ggml ggml >nul
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release || exit /b 1
cmake --build build -j 8 || exit /b 1
