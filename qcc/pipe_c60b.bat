@echo off
rem (9,143,60) after L-BFGS + 150k Adam: 40 rounding stages x 1500 L-BFGS iterations, then lambda/gauge/build.
call C:\prog\casmi\night\day2_pipe.bat adam_60a.npz c60b 40 1500
