@echo off
rem (9,143,60) after the 6-round GPU fit: 40 rounding stages x 1500 L-BFGS iterations, then lambda/gauge/build.
call C:\prog\casmi\night\day2_pipe.bat fit_9_143_60_a.npz c60a 40 1500
