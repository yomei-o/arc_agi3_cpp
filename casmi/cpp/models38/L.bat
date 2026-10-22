@echo off
rem L.bat "<job> <job> ..." : launch C:\prog\casmi\models38\<job>.bat detached (night\launch.ps1, name m38_<job>)
for %%n in (%~1) do powershell -ExecutionPolicy Bypass -File C:\prog\casmi\night\launch.ps1 m38_%%n C:\prog\casmi\models38\%%n.bat
