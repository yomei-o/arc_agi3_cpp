# pool730 -> packed 10226-bit FPNet fingerprints (chem.raw_fingerprint()[fp_bits], np.packbits), row i = line i of pool730_mols.txt
# usage: pool_fp.py <code_dir> <fp_bits.npy> <mols.txt> <out.bin> <workers>
import sys, numpy as np, time
from multiprocessing import Pool
code, bitsp, molp, outp, nw = sys.argv[1:6]
sys.path.insert(0, code)
def init():
    global chem, BITS
    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
    from casmi import chem as c; chem = c
    BITS = np.load(bitsp)
def fp(smi):
    r = chem.raw_fingerprint(smi)
    if r is None: return None
    return np.packbits(r[BITS]).tobytes()
if __name__ == "__main__":
    t0 = time.time()
    smis = [l.rstrip("\n").split("\t")[1] for l in open(molp, encoding="utf-8")]
    nb = len(np.load(bitsp)); nbytes = (nb + 7) // 8
    print("n", len(smis), "nbits", nb, "nbytes", nbytes, flush=True)
    bad = 0
    with Pool(int(nw), initializer=init) as P, open(outp, "wb") as f, open(outp + ".ok", "wb") as g:
        for i, r in enumerate(P.imap(fp, smis, chunksize=256)):
            if r is None: bad += 1; f.write(b"\0" * nbytes); g.write(b"\0")
            else: f.write(r); g.write(b"\1")
            if i % 100000 == 0: print(i, f"{time.time()-t0:.0f}s", flush=True)
    print("done bad", bad, f"{time.time()-t0:.0f}s")
