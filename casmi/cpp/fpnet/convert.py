# fpnet_full1.pt -> raw f32 binary for fpnet_ggml.cpp
# format: "FPN1" | i32 nbits,d,layers,heads,n_tensors | per tensor: i32 namelen, name, i32 ndim, i32 dims[ndim] (torch order), f32 data
import sys, struct, torch, numpy as np
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = ck["model"]
with open(sys.argv[2], "wb") as f:
    f.write(b"FPN1"); f.write(struct.pack("<5i", ck["nbits"], ck["d"], ck["layers"], ck["heads"], len(sd)))
    for k, v in sd.items():
        a = v.detach().float().contiguous().numpy()
        kb = k.encode(); f.write(struct.pack("<i", len(kb))); f.write(kb)
        f.write(struct.pack("<i", a.ndim)); f.write(struct.pack(f"<{a.ndim}i", *a.shape)); f.write(a.astype("<f4").tobytes())
print("ok", len(sd))
