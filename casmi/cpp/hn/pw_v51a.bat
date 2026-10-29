@echo off
rem v51a: push when a GPU session is free, then wait for the 09:05 JST submission reset, then the normal watcher.
call C:\prog\casmi\night\push_wait.bat kp_v51a C:\prog\casmi\night\nop.bat
powershell -ExecutionPolicy Bypass -File C:\prog\casmi\night\wait_0905.ps1
call C:\prog\casmi\night\sub_v51a.bat
