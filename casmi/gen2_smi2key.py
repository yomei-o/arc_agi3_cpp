# gen2_smi2key.py: convert a SMILES submission CSV (molecule_id,smiles;smiles;...)
# to InChIKey14 lists so mrr.py can score it (checks the submission path end to end).
#   python gen2_smi2key.py <in.csv> <out.csv>
import sys, csv
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
bad = 0
with open(sys.argv[1], encoding='utf8') as f, open(sys.argv[2], 'w', encoding='utf8', newline='') as g:
    r = csv.reader(f); w = csv.writer(g)
    w.writerow(next(r))
    for row in r:
        keys = []
        for s in (row[1].split(';') if len(row) > 1 and row[1] else []):
            m = Chem.MolFromSmiles(s)
            k = Chem.MolToInchiKey(m)[:14] if m is not None else ''
            if not k:
                bad += 1
            keys.append(k or 'BAD')
        w.writerow([row[0], ';'.join(keys)])
print('unparsable', bad)
