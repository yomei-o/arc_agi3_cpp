"""ice_keys.py: molecular formula + InChIKey14 for a list of SMILES (stdin JSON list -> stdout JSON {smiles: [formula, ik14]}).

Run as a subprocess with the ICEBERG package's wheel site first on PYTHONPATH (RDKit there) so the notebook process
itself never needs RDKit. Unparseable SMILES get ["", ""].
"""
import json
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog("rdApp.*")
out = {}
for s in json.load(sys.stdin):
    try:
        m = Chem.MolFromSmiles(s)
        out[s] = [CalcMolFormula(m), Chem.MolToInchiKey(m)[:14]] if m is not None else ["", ""]
    except Exception:
        out[s] = ["", ""]
json.dump(out, sys.stdout)
