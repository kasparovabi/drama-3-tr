import json
import os
import sys
import time

os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "1"

import soundfile as sf
import torch

from engine import OnDemandExperts, PrefixCache, attach_speaker, build_model, hot_experts, synthesize
from prepare_data import encode, load_codec

STRENGTH = float(os.getenv("TTS_CFG", "2"))
TAKES = int(os.getenv("TAKES", "3"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/runaway"
BANK = os.getenv("BANK", "ref/voice")
SENTENCES = [
    "Bu hafta gündemde yine aynı tartışma var.",
    "Belediye yeni bir karar aldı ve kimse bundan haberdar değildi.",
    "Rakamlara bakınca durum hiç de anlatıldığı gibi değil.",
    "Dün akşam açıklanan veriler piyasayı sarstı.",
    "Bunu ilk duyduğumda gerçekten inanamadım.",
    "Peki bu kararın bedelini kim ödeyecek?",
    "Mahalledeki esnaf aylardır aynı sorundan şikâyet ediyor ama ses çıkaran yok.",
    "Sonuçta herkes kendi payına düşeni yapmak zorunda.",
]
DIRECTIONS = {
    "": "ana", "ciddi": "ciddi", "hüzünlü": "huzunlu", "şaşkın": "saskin", "endişeli": "endiseli",
    "alaycı": "alayci", "sert": "sert", "samimi": "samimi", "heyecanla": "ana", "fısıldayarak": "ana", "gülerek": "ana",
}

if os.getenv("DIRS"):
    DIRECTIONS = {d: "ana" for d in os.environ["DIRS"].split(",")}
os.makedirs(OUT, exist_ok=True)
model, store = build_model(300, disk=True)
if int(os.getenv("HOT", "0")):
    store.pin_hot(hot_experts(int(os.environ["HOT"])))
attach_speaker(model, "speaker.safetensors")
OnDemandExperts.batch = 2
codec = load_codec()
log = open(f"{OUT}/runs.jsonl", "a", encoding="utf-8")
done = set()
if os.path.exists(f"{OUT}/runs.jsonl"):
    done = {(r["direction"], r["i"], r["take"]) for r in map(json.loads, open(f"{OUT}/runs.jsonl", encoding="utf-8"))}

for ref in sorted(set(DIRECTIONS.values())):
    name = f"{BANK}/{ref}"
    audio, sr = sf.read(name + ".wav", dtype="float32")
    codes = encode(codec, audio if audio.ndim == 1 else audio.mean(1), sr).long()
    text = open(name + ".txt", encoding="utf-8").read().strip()
    prefix = PrefixCache(model, text, codes)
    for direction in [d for d, r in DIRECTIONS.items() if r == ref]:
        for i, sentence in enumerate(SENTENCES):
            for take in range(TAKES):
                if (direction, i, take) in done:
                    continue
                torch.manual_seed(1000 * i + take)
                line = f"[{direction}] {sentence}" if direction else sentence
                t0 = time.perf_counter()
                wave, info = synthesize(model, prefix, codec, line, codes, text, temperature=1.0, top_p=0.9, strength=STRENGTH)
                row = {"direction": direction, "ref": ref, "i": i, "take": take, "chars": len(sentence), "wall": round(time.perf_counter() - t0, 1), **info}
                if info["stop"] != "eos" or info["sec"] > 3 + len(sentence) / 6:
                    sf.write(f"{OUT}/{direction or 'yok'}_{i}_{take}.wav", wave, codec.sample_rate)
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                print(row, flush=True)
