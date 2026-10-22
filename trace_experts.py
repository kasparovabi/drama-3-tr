import json
import os
import random

os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "1"

import soundfile as sf
import torch

from engine import ExpertStore, OnDemandExperts, PrefixCache, attach_speaker, build_model, synthesize
from prepare_data import encode, load_codec

N = int(os.getenv("N", "300"))
BANK = os.getenv("BANK", "ref/voice")
DIRECTIONS = ["ciddi", "hüzünlü", "şaşkın", "endişeli", "alaycı", "sert", "samimi", "heyecanla", "fısıldayarak", "kıkırdayarak", ""]
REFS = {"ciddi": "ciddi", "hüzünlü": "huzunlu", "şaşkın": "saskin", "endişeli": "endiseli", "alaycı": "alayci", "sert": "sert", "samimi": "samimi"}

seen = [set() for _ in range(40)]
slots_for = ExpertStore.slots_for


def traced(self, layer, experts):
    seen[layer].update(experts)
    return slots_for(self, layer, experts)


ExpertStore.slots_for = traced
random.seed(0)
data = json.load(open("ref/freya_tr_eval.json", encoding="utf-8"))
texts = [d["text"] if isinstance(d, dict) else d for d in data]
random.shuffle(texts)
model, store = build_model(300, disk=True)
attach_speaker(model, "speaker.safetensors")
OnDemandExperts.batch = 2
codec = load_codec()
refs = {}
counts = torch.zeros(40, 256, dtype=torch.int32)
current = None
jobs = sorted(((random.choice(DIRECTIONS), t) for t in texts[:N]), key=lambda j: REFS.get(j[0], "ana"))
for i, (d, text) in enumerate(jobs):
    ref = REFS.get(d, "ana")
    ref = ref if os.path.exists(f"{BANK}/{ref}.wav") else "ana"
    if ref not in refs:
        name = f"{BANK}/{ref}"
        audio, sr = sf.read(name + ".wav", dtype="float32")
        codes = encode(codec, audio if audio.ndim == 1 else audio.mean(1), sr).long()
        rt = open(name + ".txt", encoding="utf-8").read().strip()
        refs[ref] = (codes, rt, PrefixCache(model, rt, codes))
    codes, rt, prefix = refs[ref]
    if current != ref:
        prefix.rows = None
        current = ref
    synthesize(model, prefix, codec, f"[{d}] {text}" if d else text, codes, rt, temperature=1.0, top_p=0.9, max_frames=1, strength=2)
    for layer in range(40):
        for e in seen[layer]:
            counts[layer, e] += 1
        seen[layer].clear()
    if i % 25 == 0:
        print(i, flush=True)
torch.save({"counts": counts, "n": N}, "results/trace_experts.pt")
