import os

os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "1"

import soundfile as sf
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from engine import OnDemandExperts, PrefixCache, attach_speaker, build_model, hot_experts, synthesize
from prepare_data import encode, load_codec

model, store = build_model(300, disk=True)
store.pin_hot(hot_experts(int(os.getenv("HOT", "6000"))))
attach_speaker(model, "speaker.safetensors")
OnDemandExperts.batch = 2
codec = load_codec()
REF = os.getenv("REF", "ref/tikir_long")
audio, sr = sf.read(REF + ".wav", dtype="float32")
codes = encode(codec, audio if audio.ndim == 1 else audio.mean(1), sr).long()
text = open(REF + ".txt", encoding="utf-8").read().strip()
prefix = PrefixCache(model, text, codes)
line = "Bu hafta gündemde yine aynı tartışma var, herkes kendi payına düşeni yapmak zorunda."
for _ in range(2):
    wave, info = synthesize(model, prefix, codec, line, codes, text, temperature=1.0, top_p=0.9, strength=2, max_frames=400)
print(info, flush=True)

g = model.cfg_graphs
backbone, tower = model.backbone, model.audio_decoder
pos = int(g.pos)
kv = next(b for b in g.buckets if b >= pos + 2)
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(50):
    g.graphs[kv].replay()
end.record()
torch.cuda.synchronize()
print(f"graph step {start.elapsed_time(end) / 50:.2f} ms", flush=True)


@torch.inference_mode()
def pieces():
    with record_function("logits"):
        logits = backbone.compute_logits(g.hidden).float()
        guided = logits[1:] + g.strength * (logits[:1] - logits[1:])
    with record_function("tower"):
        sample = tower.decode_step(guided, g.hidden[:1], g.pos, g.frame, g.history,
                                   temperature=g.temperature, top_k=g.top_k, top_p=g.top_p, check_finite=False)
    with record_function("backbone"):
        e = tower.embed_backbone(sample, backbone.embed(sample[:, 0]))
        backbone(inputs_embeds=e[None].expand(2, -1, -1).contiguous(), input_pos=g.pos + 1)


for _ in range(3):
    pieces()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        pieces()
    torch.cuda.synchronize()
for e in prof.key_averages():
    if e.key in ("logits", "tower", "backbone"):
        print(f"{e.key:9s} gpu {e.device_time_total / 20 / 1000:.2f} ms  cpu {e.cpu_time_total / 20 / 1000:.2f} ms")
print(prof.key_averages().table(sort_by="device_time_total", row_limit=25))
