@echo off
rem e2e of v41 (ICEBERG/GLACIER re-scoring) on the build machine: the public runners with the pre-made Windows site
rem (pure-python wheels + marker, system RDKit 2026.03 via --any-rdkit = plumbing test, predictions not panel-exact)
set ICE_PKG=C:\prog\casmi\ice\casmi26-iceberg
set GL_PKG=C:\prog\casmi\ice\casmi26-glacier
set ICE_SITE=C:\prog\casmi\ice\site
set ICE_EXTRA=--any-rdkit
set ICE_BUDGET=1500
set GL_BUDGET=1500
call C:\prog\casmi\v36\e2e36.bat casmi_match_v41.ipynb work_v41 F00
