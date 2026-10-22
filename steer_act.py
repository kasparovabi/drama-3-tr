import json
import os

os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "1"

import soundfile as sf
import torch

from engine import OnDemandExperts, PrefixCache, attach_speaker, build_drama3_prompt, build_model, hot_experts, strip_directions, synthesize
from prepare_data import encode, load_codec

EMOTIONS = os.getenv("EMOTIONS", "hüzünlü,sert,fısıldayarak,heyecanla").split(",")
LAYERS = [int(x) for x in os.getenv("LAYERS", "8,16,24").split(",")]
ALPHAS = [float(x) for x in os.getenv("ALPHAS", "1,2").split(",")]
SENTENCES = [
    "Bu hafta gündemde yine aynı tartışma var.",
    "Dün akşam açıklanan veriler piyasayı sarstı.",
    "Bunu ilk duyduğumda gerçekten inanamadım.",
    "Sonuçta herkes kendi payına düşeni yapmak zorunda.",
]
OUT = os.getenv("OUT", "results/steer_act")
REF = os.getenv("REF", "ref/tikir_long")
os.makedirs(OUT, exist_ok=True)
data = json.load(open("ref/freya_tr_eval.json", encoding="utf-8"))[300:330]

model, store = build_model(300, disk=True)
store.pin_hot(hot_experts(int(os.getenv("HOT", "6000"))))
attach_speaker(model, "speaker.safetensors")
OnDemandExperts.batch = 2
codec = load_codec()
audio, sr = sf.read(REF + ".wav", dtype="float32")
codes = encode(codec, audio if audio.ndim == 1 else audio.mean(1), sr).long()
text = open(REF + ".txt", encoding="utf-8").read().strip()
prefix = PrefixCache(model, text, codes)
synthesize(model, prefix, codec, "Merhaba.", codes, text, temperature=1.0, top_p=0.9, max_frames=1, strength=2)
layers = model.backbone.model.layers
state = {"collect": False, "add": None}
grabs = {l: [] for l in LAYERS}


def hook(index):
    def fn(module, inputs, out):
        if out.shape[1] <= 1:
            return
        if state["collect"]:
            grabs[index].append(out.detach().float())
        if state["add"] is not None and index == state["add"][0]:
            return out + state["add"][1].to(out.dtype)
    return fn


for l in LAYERS:
    layers[l].register_forward_hook(hook(l))

P = prefix.length
vectors = {}
for emo in EMOTIONS:
    sums = {l: 0 for l in LAYERS}
    rows = 0
    for s in data:
        line = f"[{emo}] {s}"
        prompt = build_drama3_prompt(line, model.tokenizer, model.config, [text], [codes])
        same = (prompt[P:] == strip_directions(prompt, model.tokenizer, P)[P:]).all(-1).cuda()
        for l in LAYERS:
            grabs[l].clear()
        state["collect"] = True
        synthesize(model, prefix, codec, line, codes, text, temperature=1.0, top_p=0.9, max_frames=1, strength=2)
        state["collect"] = False
        for l in LAYERS:
            h = torch.cat(grabs[l], 1)
            sums[l] = sums[l] + (h[0] - h[1])[same].sum(0)
        rows += int(same.sum())
    vectors[emo] = {l: sums[l] / rows for l in LAYERS}
    print(emo, {l: round(float(vectors[emo][l].norm()), 2) for l in LAYERS}, flush=True)
torch.save({e: {l: v.cpu() for l, v in d.items()} for e, d in vectors.items()}, f"{OUT}/vectors.pt")


def say(line, name, add=None):
    for j in range(2):
        torch.manual_seed(j)
        state["add"] = add
        try:
            wave, info = synthesize(model, prefix, codec, line, codes, text, temperature=1.0, top_p=0.9, strength=2, max_frames=400)
        except RuntimeError:
            continue
        finally:
            state["add"] = None
        sf.write(f"{OUT}/{name}_{j}.wav", wave, codec.sample_rate)


for i, s in enumerate(SENTENCES):
    say(s, f"{i}_yok")
    for emo in EMOTIONS:
        say(f"[{emo}] {s}", f"{i}_{emo}_etiket")
        for l in LAYERS:
            for a in ALPHAS:
                say(s, f"{i}_{emo}_L{l}a{a:g}", add=(l, a * vectors[emo][l]))
    print(i, flush=True)
