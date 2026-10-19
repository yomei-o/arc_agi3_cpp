@echo off
rem LB-probe measurement: non-pool PubChem list on hcqry rm=1 (m_out), hcqry normal (sanity, ~0), haqry rm=1
call C:\prog\casmi\fwdmodel\probe\run.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin pr_hc1 ",rm=1"
call C:\prog\casmi\fwdmodel\probe\run.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin pr_hc0 ""
call C:\prog\casmi\fwdmodel\probe\run.bat haqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin pr_ha1 ",rm=1"
echo ALLDONE
