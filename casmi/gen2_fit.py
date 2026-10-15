# gen2_fit.py: fit fusion + generation weights with fuse_fit on phys_gen
# feature dumps, write a weights file phys_gen reads (8 base + 6 gen).
#   python gen2_fit.py <fuse_fit.exe> <train:w,...> <eval,...> <out_weights.txt> [lambda=0.001] [disabled=7,8,9,10,11,12,13]
# inLib (7) and the COCONUT metadata (8-13) stay disabled (inLib lost 0.041 on the LB).
import sys, subprocess, re
exe, train, ev, outp = sys.argv[1:5]
lam = sys.argv[5] if len(sys.argv) > 5 else '0.001'
dis = sys.argv[6] if len(sys.argv) > 6 else '7,8,9,10,11,12,13'
r = subprocess.run([exe, train, ev, lam, dis, '-', 'rrrrrr'], capture_output=True, text=True)
print(r.stdout)
if r.returncode:
    print(r.stderr)
    sys.exit(1)
line = [l for l in r.stdout.splitlines() if l.startswith('weights:')][0]
w = [float(x) for x in re.findall(r'\s(-?[0-9.]+(?:e-?\d+)?)(?=\s|$)', line)]
assert len(w) == 20, (len(w), line)
with open(outp, 'w') as f:
    f.write('\n'.join(f'{x:.6g}' for x in w[:8] + w[14:20]) + '\n')
print('wrote', outp, w[:8] + w[14:20])
