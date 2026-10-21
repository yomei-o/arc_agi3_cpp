# Our own 10226-bit fingerprint (the FPNet output bits) with RDKit, written from the bit definition
# (fpnet_own_STATUS.md): raw = ECFP4 4096 | ECFP6 4096 | FCFP4 2048 | RDKit path<=6 2048 | AtomPair 2048 |
# TopTorsion 2048 | MACCS 167 (16551 bits), then the 10226 selected positions (fp_bits.npy, an index list
# that DEFINES which bits the inference file's output head stands for; it is data, not code/weights).
#
#   fp10226.py mols  <mols.txt> <fp_bits.npy> <out_stem> <workers>
#        mols.txt lines "<name>\t<smiles>" (or "<uid>\t<smiles>"); writes
#        <out_stem>.bin   sparse Fps (u32 n, u32 nbits, per row u16 count + u16 bits[count]) = tfm_train / win_eval format
#        <out_stem>.ok    1 byte per row (0 = SMILES did not parse -> empty row)
#   fp10226.py verify <pool730_mols.txt> <pool730_fp10226.bin> <fp_bits.npy> <n=300>
#        recompute n random pool rows and compare bit-for-bit with the packed pool file (1279 B/row)
import sys, struct, time
import numpy as np
from multiprocessing import Pool

GEN = None
BITS = None


def init(bitsp):
    global GEN, BITS
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator as g
    RDLogger.DisableLog("rdApp.*")
    GEN = [g.GetMorganGenerator(radius=2, fpSize=4096),
           g.GetMorganGenerator(radius=3, fpSize=4096),
           g.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=g.GetMorganFeatureAtomInvGen()),
           g.GetRDKitFPGenerator(fpSize=2048, maxPath=6),
           g.GetAtomPairGenerator(fpSize=2048),
           g.GetTopologicalTorsionGenerator(fpSize=2048)]
    BITS = np.load(bitsp)


def raw(smi):
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        parts = [gg.GetFingerprintAsNumPy(m).astype(np.uint8) for gg in GEN]
        parts.append(np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.uint8))
        return np.concatenate(parts)
    except Exception:
        return None


def sel(smi):
    r = raw(smi)
    if r is None:
        return None
    return np.flatnonzero(r[BITS]).astype("<u2")


def cmd_mols(molp, bitsp, out, nw):
    t0 = time.time()
    smis = []
    for l in open(molp, encoding="utf-8"):
        p = l.rstrip("\r\n").split("\t")
        smis.append(p[1] if len(p) > 1 else "")
    nb = len(np.load(bitsp))
    print("rows", len(smis), "nbits", nb, flush=True)
    bad = 0
    with Pool(nw, initializer=init, initargs=(bitsp,)) as P, open(out + ".bin", "wb") as f, open(out + ".ok", "wb") as g:
        f.write(struct.pack("<II", len(smis), nb))
        for i, r in enumerate(P.imap(sel, smis, chunksize=512)):
            if r is None:
                bad += 1; f.write(struct.pack("<H", 0)); g.write(b"\0")
            else:
                f.write(struct.pack("<H", len(r))); f.write(r.tobytes()); g.write(b"\1")
            if i % 200000 == 0:
                print(i, f"{time.time() - t0:.0f}s", flush=True)
    print("done bad", bad, f"{time.time() - t0:.0f}s", flush=True)


def cmd_verify(molp, poolp, bitsp, n):
    init(bitsp)
    smis = [l.rstrip("\r\n").split("\t")[1] for l in open(molp, encoding="utf-8")]
    nb = len(BITS); nbytes = (nb + 7) // 8
    fp = np.memmap(poolp, np.uint8, "r").reshape(-1, nbytes)
    assert fp.shape[0] == len(smis), (fp.shape, len(smis))
    rng = np.random.default_rng(2610)
    idx = rng.choice(len(smis), n, replace=False)
    eq = diff = none = 0
    for i in idx:
        r = raw(smis[i])
        if r is None:
            none += 1; continue
        a = np.packbits(r[BITS])
        if np.array_equal(a, fp[i]): eq += 1
        else: diff += 1; print("diff row", i, int((np.unpackbits(a) != np.unpackbits(fp[i])).sum()), smis[i])
    print(f"verify {n} random pool rows: equal {eq} diff {diff} unparsable {none}")


if __name__ == "__main__":
    if sys.argv[1] == "mols":
        cmd_mols(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif sys.argv[1] == "verify":
        cmd_verify(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 300)
