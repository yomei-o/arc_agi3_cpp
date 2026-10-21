@echo off
rem lane Z: v30 production config dumps (phys_v30pc pcrev=tan), hxqry (xinstr, zlib + rev2_z)
call C:\prog\casmi\fwdmodel\pc5\run5.bat hxqry zlib fwd_mlp_rank_ylib.bin rev2_z.bin hx30_0 phys_v30pc 2 ",pcrev=tan"
echo ALLDONE
