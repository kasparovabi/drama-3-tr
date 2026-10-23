import os

os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "1"

import soundfile as sf
import torch

from engine import OnDemandExperts, PrefixCache, attach_speaker, build_model, hot_experts, read_prompt_cfg, strip_directions, synthesize
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt
from prepare_data import encode, load_codec

model, store = build_model(300, disk=True)
store.pin_hot(hot_experts(6000))
attach_speaker(model, "speaker.safetensors")
OnDemandExperts.batch = 2
codec = load_codec()
audio, sr = sf.read("ref/yeni_ses/ana.wav", dtype="float32")
codes = encode(codec, audio if audio.ndim == 1 else audio.mean(1), sr).long()
ref_text = open("ref/yeni_ses/ana.txt", encoding="utf-8").read().strip()
prefix = PrefixCache(model, ref_text, codes)
opts = dict(temperature=1.0, top_p=0.9, strength=2.0, max_frames=400)
LINES = ["[ciddi] Fransa Senatosu da sporda dini simgeleri yasaklayan tasarıyı kabul etti.",
         "[endişeli ama kararlı] Korkarım bu gidişle başını örten kadınlar dışarıda kalacak."]

synthesize(model, prefix, codec, LINES[0], codes, ref_text, **opts)
rows, turns = model.cfg_session["rows"], model.cfg_session["turns"]
prompt = build_drama3_prompt(LINES[1], model.tokenizer, model.config, [ref_text], [codes], turns)
assert torch.equal(prompt[: len(rows)], rows)
n = len(prompt)
fA, fB = prompt.cuda(), strip_directions(prompt, model.tokenizer, prefix.length).cuda()
pos = torch.arange(n, device="cuda")


def logp(start):
    with torch.inference_mode():
        h = read_prompt_cfg(model, prefix, fA, fB, pos, n, start)
        return torch.log_softmax(model.backbone.compute_logits(h).float()[:, model.audio_decoder.main_candidates], -1)


cont, full, again = logp(len(rows)), logp(None), logp(None)
kl = lambda p, q: float((p.exp() * (p - q)).sum(-1).max())
print(f"satir {n}, surdurulen {len(rows)}, KL surdurme-bastan {kl(full, cont):.2e}, KL bastan-bastan {kl(full, again):.2e}")
print("en olasi kod ayni", bool((cont.argmax(-1) == full.argmax(-1)).all()))
assert kl(full, cont) < 0.2, "konusmaci ile tam model farki 0,07 civari, cok buyukse surdurme bozuk"
print("OK")
