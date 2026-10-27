"""ice_hp.py <ckpt.pt>: hyper-parameters and state_dict shapes of an ICEBERG checkpoint (gen.pt / inten.pt)."""
import sys

import torch

c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(sorted(c.keys()))
hp = c.get("hyper_parameters", {})
print({k: v for k, v in hp.items() if not str(k).startswith("_")})
sd = c["state_dict"]
print("params", sum(v.numel() for v in sd.values()))
for k, v in sd.items():
    if ".layers." in k and ".layers.0." not in k:
        continue
    print(k, tuple(v.shape))
