@echo off
rem stage.bat : copy the p37 gsdump base runs (v35 + attpool 2, cnn = cnn_a_c) and the query mols files into models38\D (pool38 input)
cd /d C:\prog\casmi
for %%t in (p36x_0 p36b_0 p36bs_0) do for %%e in (_cf.txt _det.txt _pp.txt .csv _ins.csv) do copy /y pretrain\p37\out\%%t%%e models38\D\ > nul
for %%q in (hxqry hbqry hbsqry) do copy /y %%q_mols.txt models38\D\ > nul
dir /b models38\D
