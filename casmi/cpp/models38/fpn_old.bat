@echo off
rem fpn_old.bat (launch name m38_fpn_old) : pool-term inputs for the epochs 1-4 of the existing 5-epoch wlib run (fpown_w_ep<N>.bin;
rem   epoch 5 = fpown_w.bin is done by fpn_watch.bat) -> the pool metric's trend over epochs of the production recipe.
cd /d C:\prog\casmi
for %%e in (2 3 4 1) do call C:\prog\casmi\models38\fpn_eval.bat fpown_w_ep%%e
echo all done
