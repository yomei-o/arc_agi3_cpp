@echo off
rem final cqry/tqry evaluations, tie-averaged physics MRR (fusion lines are unaffected by ties)
set QM_TIES=avg
set C=C:\prog\casmi\qm\coef_fqry_full.txt
call C:\prog\casmi\qm\ev.bat refitT qm\p_refitT.txt %C% 0 0
call C:\prog\casmi\qm\ev.bat qF qm\p_gF.txt %C% 0 1
call C:\prog\casmi\qm\ev.bat paF qm\p_paF.txt %C% 1 0
call C:\prog\casmi\qm\evd.bat refitT qm\pd_refitT.txt %C% 0 0
call C:\prog\casmi\qm\evd.bat qF qm\pd_q.txt %C% 0 1
