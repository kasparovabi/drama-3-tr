import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from safetensors.torch import save_file

from engine import fp8_block, load_speaker

source, target = sys.argv[1], sys.argv[2]
out = {}
for i, p in enumerate(torch.load(source, mmap=True)):
    w13, s13 = fp8_block(p["w13"])
    w2, s2 = fp8_block(p["w2"])
    for name, t in (("selected", p["selected"]), ("gate", p["gate"].float()), ("bias", p["bias"].float()),
                    ("w13", w13), ("s13", s13), ("w2", w2), ("s2", s2)):
        out[f"layers.{i}.{name}"] = t.contiguous()
save_file(out, target, metadata={"format": "pt", "base_model": "fishaudio/drama-3", "quant": "fp8_e4m3 block 128x128"})
back = load_speaker(target)
orig = torch.load(source, mmap=True)
for i in (0, 17, 39):
    q, s = fp8_block(orig[i]["w13"])
    assert torch.equal(back[i]["w13"].view(torch.uint8), q.view(torch.uint8)) and torch.equal(back[i]["s13"], s)
    assert torch.equal(back[i]["gate"], orig[i]["gate"].float())
print("ok", len(back), "layers")
