@echo off
rem setup_ref.bat: reference venv C:\prog\casmi\iceberg\venv (system Python untouched).
rem DGL has no Windows wheels for torch >= 2.4 (data.dgl.ai torch-2.6 repo: linux only); the last Windows wheel on
rem PyPI is dgl 2.2.1 (CPU only) -> torch 2.3.1 CPU, numpy < 2. This venv is the CPU reference / timing environment.
setlocal
set PY=C:\prog\casmi\iceberg\venv\Scripts\python.exe
%PY% -m pip install -q --upgrade pip
%PY% -m pip install -q torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
%PY% -m pip install -q "numpy<2" dgl==2.2.1 rdkit==2025.3.6 pytorch-lightning==2.4.0 torchmetrics einops pandas pyarrow scipy scikit-learn h5py omegaconf pyyaml tqdm networkx LinSATNet
%PY% -m pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
%PY% -m pip install -q pygmtools
%PY% -m pip freeze > C:\prog\casmi\iceberg\venv_freeze.txt
%PY% -c "import torch, dgl, rdkit, torch_scatter, numpy; print('torch', torch.__version__, 'dgl', dgl.__version__, 'rdkit', rdkit.__version__, 'numpy', numpy.__version__)"
echo SETUPDONE
