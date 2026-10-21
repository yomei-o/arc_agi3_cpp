@echo off
rem setup_gpu.bat: GPU venv C:\prog\casmi\iceberg\venv_gpu for GLACIER (system Python untouched).
rem torch 2.6.0+cu124 as ms-pred's pyproject (cu124 extra); NO dgl (no Windows CUDA wheels): GLACIER's
rem predict path is dense (graphormer + LinSATNet), ice_score.py --dgl-stub registers an import-only stub.
setlocal
"C:\Program Files\Python312\python.exe" -m venv C:\prog\casmi\iceberg\venv_gpu
set PY=C:\prog\casmi\iceberg\venv_gpu\Scripts\python.exe
"%PY%" -m pip install -q --upgrade pip
"%PY%" -m pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"%PY%" -m pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
"%PY%" -m pip install -q rdkit==2025.3.6 pytorch-lightning==2.4.0 torchmetrics einops pandas scipy scikit-learn h5py omegaconf pyyaml tqdm networkx LinSATNet pygmtools multiprocess pathos matplotlib seaborn platformdirs cython pubchempy
"%PY%" -m pip freeze > C:\prog\casmi\iceberg\venv_gpu_freeze.txt
"%PY%" -c "import torch, rdkit, numpy; print('torch', torch.__version__, torch.cuda.is_available(), 'rdkit', rdkit.__version__, 'numpy', numpy.__version__)"
echo SETUPDONE
