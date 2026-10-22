import json
import os
import sys
import time

os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "1"

import soundfile as sf
import torch

import engine
from engine import ExpertStore, OnDemandExperts, PrefixCache, attach_speaker, build_model, synthesize
from prepare_data import encode, load_codec

STRENGTH = float(os.getenv("TTS_CFG", "2"))
SENTENCES = [
    "Bu hafta gündemde yine aynı tartışma var.",
    "Belediye yeni bir karar aldı ve kimse bundan haberdar değildi.",
    "Rakamlara bakınca durum hiç de anlatıldığı gibi değil.",
    "Peki bu kararın bedelini kim ödeyecek?",
    "Dün akşam açıklanan veriler piyasayı sarstı.",
    "Bunu ilk duyduğumda gerçekten inanamadım.",
]

stats = {"disk_s": 0.0, "loaded": 0, "read_s": 0.0}
load = ExpertStore._load_from_disk


def timed_load(self, layer, missing, I):
    t = time.perf_counter()
    load(self, layer, missing, I)
    stats["disk_s"] += time.perf_counter() - t
    stats["loaded"] += len(missing)


ExpertStore._load_from_disk = timed_load
read = engine.read_prompt_cfg


def timed_read(*a):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = read(*a)
    torch.cuda.synchronize()
    stats["read_s"] += time.perf_counter() - t
    return out


engine.read_prompt_cfg = timed_read

model, store = build_model(int(os.getenv("SLOTS", "300")), disk=True)
if int(os.getenv("HOT", "0")):
    t0 = time.perf_counter()
    store.pin_hot(engine.hot_experts(int(os.environ["HOT"])))
    print(f"pinned {len(store.hot)} experts in {time.perf_counter() - t0:.0f}s", flush=True)
fetch = store.disk.fetch


def timed_fetch(keys):
    t = time.perf_counter()
    out = fetch(keys)
    stats["fetch_s"] += time.perf_counter() - t
    stats["cold"] += len(keys)
    return out


store.disk.fetch = timed_fetch
stats.update(fetch_s=0.0, cold=0)
attach_speaker(model, "speaker.safetensors")
OnDemandExperts.batch = 2
codec = load_codec()
name = os.getenv("REF", "ref/tikir_long")
audio, sr = sf.read(name + ".wav", dtype="float32")
codes = encode(codec, audio if audio.ndim == 1 else audio.mean(1), sr).long()
text = open(name + ".txt", encoding="utf-8").read().strip()
prefix = PrefixCache(model, text, codes)
synthesize(model, prefix, codec, "Merhaba.", codes, text, temperature=1.0, top_p=0.9, strength=STRENGTH)
rows = []
for i, s in enumerate(SENTENCES):
    for k in stats:
        stats[k] = 0
    torch.manual_seed(i)
    wave, info = synthesize(model, prefix, codec, f"[ciddi] {s}", codes, text, temperature=1.0, top_p=0.9, strength=STRENGTH)
    frames_s = info["gen_sec"] - stats["read_s"]
    row = {**info, "wave_sum": round(float(abs(wave).sum()), 3), "read_s": round(stats["read_s"], 2), "disk_s": round(stats["disk_s"], 2), "loaded": stats["loaded"], "cold": stats["cold"], "fetch_s": round(stats["fetch_s"], 2),
           "frames_s": round(frames_s, 2), "ms_per_frame": round(1000 * frames_s / max(1, info["frames"]), 1), "rtf_x": round(info["sec"] / info["gen_sec"], 2)}
    rows.append(row)
    if os.getenv("SAVE"):
        os.makedirs(os.environ["SAVE"], exist_ok=True)
        sf.write(f"{os.environ['SAVE']}/{i}_{os.getenv('TAG', 'x')}_0.wav", wave, codec.sample_rate)
    print(row, flush=True)
json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "results/profile_short.json", "w"), indent=1)
