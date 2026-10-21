@echo off
rem smoke_post.bat : smoke test of the night_post.bat steps with the pilot model (pt_pilot.bin)
setlocal
cd /d C:\prog\casmi
set P=C:\prog\casmi\pretrain
%P%\pt_train.exe export model=pt_pilot.bin lib=wlib fp=fp3c16k HH=2048 out=ft_t_ck.bin > %P%\smoke_export.log 2>&1
%P%\tfm_train.exe train lib=wlib fp=fp3c16k out=ft_t resume=1 epochs=1 B=128 lr=3e-4 warm=2000 eval=none hold=20 ckptEvery=0 stopAfter=2 threads=3 > %P%\smoke_ft.log 2>&1
%P%\pt_train.exe mollogits model=ft_t_ck.bin qry=hcqry out=pretrain\ft_t_hcqry_mol.bin > %P%\smoke_mol.log 2>&1
C:\prog\casmi\fpown\build\win_eval.exe dir=C:\prog\casmi qry=hcqry win=w_hcqry "models=pretrain\ft_t_hcqry_mol.bin:fp3c16k" threads=3 > %P%\smoke_win.log 2>&1
%P%\pt_train.exe nbr model=pt_pilot.bin lib=wlib qrys=hcqry,hxqry tag=pilotc threads=3 > %P%\pilotc_nbr.log 2>&1
for %%q in (hcqry hxqry) do (
  %P%\pt_eval.exe qry=%%q lib=wlib vocabs=fp3c16k pop=1 threads=3 mpen=4 embnbr=ptnbr_%%q_pilotc.bin "refcfg=3:10:0.1:0" "sims=0,1,2" "ks=1,10,50" "Ts=0.001,0.003,0.01,0.03,0.1" lams=0 "ensw=0.5,1,2" > %P%\pilotc_eval_%%q.log 2>&1
)
echo smoke done
