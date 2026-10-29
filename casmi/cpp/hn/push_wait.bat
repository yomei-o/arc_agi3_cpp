@echo off
rem push_wait.bat <kp dir name> <watcher bat>: retry "kaggle kernels push" every 5 min until a GPU session is free
rem (max 2 batch GPU sessions), then run the watcher bat in this same process.
cd /d C:\prog\casmi\night
:retry
kaggle kernels push -p %1 > push_%1.txt 2>&1
type push_%1.txt
findstr /c:"successfully pushed" push_%1.txt >nul && goto ok
timeout /t 300 /nobreak >nul
goto retry
:ok
echo %date% %time% pushed %1
call %2
