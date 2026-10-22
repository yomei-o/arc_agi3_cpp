@echo off
rem fpn_grid.bat : pool-term grid of the own FPNet epochs (models38_STATUS.md). f5 = fpown_w.bin (5-epoch wlib run, the recipe of
rem   production fpnet_own_f16.bin), o<N> = its epoch N (fpown_w_ep<N>), e<N> = fpown12_w_ep<N> (12-epoch run). Term form =
rem   production: + 1.0 z(logit dot) on top of + 2 z(cnn_a); t<x> = the same model in the continuous-Tanimoto form.
rem   Reference = cnn 2 + f5 1. Controls: <col>~ = query q scored with q+7's logits, @perm = permuted within the query.
rem   Output: stdout (fpn_watch.bat writes it to models38\fpn_grid.txt). Missing epochs are skipped.
setlocal
cd /d C:\prog\casmi
set M=C:\prog\casmi\models38
set F=%M%\fpn
set BITS=%M%\p36pool_fp10226.bin::pretrain\p36pool_mols.txt;%M%\g{s}_fp10226.bin::pretrain\g{s}_mols.txt
set C=col=f5=dot:%F%\fpown_w_{q}_mol.bin#%BITS% col=t5=tan:%F%\fpown_w_{q}_mol.bin#%BITS%
for /l %%e in (1,1,4) do call set C=%%C%% col=o%%e=dot:%F%\fpown_w_ep%%e_{q}_mol.bin#%BITS%
for /l %%e in (1,1,12) do call set C=%%C%% col=e%%e=dot:%F%\fpown12_w_ep%%e_{q}_mol.bin#%BITS%
for %%e in (8 10 12) do call set C=%%C%% col=t%%e=tan:%F%\fpown12_w_ep%%e_{q}_mol.bin#%BITS%
set G=cnn:2;cnn:2,f5:0.5;cnn:2,f5:1.5;cnn:2,f5~:1;cnn:2,f5:1,@perm=f5;cnn:2,t5:1;cnn:2,t5:2
for /l %%e in (1,1,4) do call set G=%%G%%;cnn:2,o%%e:1
for /l %%e in (1,1,12) do call set G=%%G%%;cnn:2,e%%e:1
set G=%G%;cnn:2,e8:0.5;cnn:2,e8:1.5;cnn:2,e10:1.5;cnn:2,e12:0.5;cnn:2,e12:1.5;cnn:2,t8:1;cnn:2,t10:1;cnn:2,t12:1;cnn:2,t12:2
set G=%G%;cnn:2,e8~:1;cnn:2,e8:1,@perm=e8;cnn:2,e12~:1;cnn:2,e12:1,@perm=e12;cnn:2,e10~:1
"C:\Program Files\Python312\python.exe" %M%\py\pool38.py %M%\D %M%\grid_out p36x_0:hxqry:x,p36b_0:hbqry:b,p36bs_0:hbsqry:bs "%G%" %C% "ref=cnn:2,f5:1"
