import json
import os
import time

os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "1"

import numpy as np
import soundfile as sf
import torch

from engine import OnDemandExperts, PrefixCache, StreamDecoder, attach_speaker, build_model, synthesize
from prepare_data import encode, load_codec

SENTENCES = [
    "Selam dostum, bugün nasılsın?",
    "[heyecanla] Harika bir haberim var, sonunda robot kolum çalıştı!",
    "[sıcak ve gizemli bir masalcı gibi] Bugün sana küçük bir hikâye anlatacağım. Bir zamanlar, denizin kıyısında küçük bir kasaba varmış. [gülerek] Kasabanın kedisi her sabah balıkçıların teknesine gizlice binermiş.",
]

model, store = build_model(300)
attach_speaker(model, "speaker.safetensors")
OnDemandExperts.batch = 2
codec = load_codec()
audio, sr = sf.read("ref/tikir_long.wav", dtype="float32")
ref_codes = encode(codec, audio if audio.ndim == 1 else audio.mean(1), sr).long()
ref_text = open("ref/tikir_long.txt", encoding="utf-8").read().strip()
prefix = PrefixCache(model, ref_text, ref_codes)
opts = dict(temperature=1.0, top_p=0.9, strength=4.0)
synthesize(model, prefix, codec, "Merhaba.", ref_codes, ref_text, **opts)

SR = codec.sample_rate
gap = np.zeros(SR, dtype=np.float32)
streamed, waited, report = [], [], []
for text in SENTENCES:
    pieces = []
    t0 = time.perf_counter()
    stream = StreamDecoder(codec, lambda x: pieces.append((time.perf_counter() - t0, x)))
    full, info = synthesize(model, prefix, codec, text, ref_codes, ref_text, stream=stream, **opts)
    done = time.perf_counter() - t0
    joined = np.concatenate([x for _, x in pieces])
    n = min(len(joined), len(full))
    snr = 10 * np.log10((full[:n] ** 2).sum() / max(((full[:n] - joined[:n]) ** 2).sum(), 1e-12))
    live, clock, stalls = [np.zeros(int(pieces[0][0] * SR), dtype=np.float32)], pieces[0][0], 0.0
    for arrival, x in pieces:
        if arrival > clock:
            live.append(np.zeros(int((arrival - clock) * SR), dtype=np.float32))
            stalls += arrival - clock
            clock = arrival
        live.append(x)
        clock += len(x) / SR
    streamed += live + [gap]
    waited += [np.zeros(int(done * SR), dtype=np.float32), full, gap]
    report.append({"text": text[:50], "audio_sec": round(len(full) / SR, 2), "first_audio_sec": round(pieces[0][0], 2),
                   "full_wait_sec": round(done, 2), "stall_sec": round(stalls, 2), "pieces": len(pieces),
                   "snr_db": round(float(snr), 1), "length_diff": len(joined) - len(full)})
    print(report[-1], flush=True)

os.makedirs("listen", exist_ok=True)
sf.write("listen/streamed.wav", np.concatenate(streamed), SR)
sf.write("listen/not_streamed.wav", np.concatenate(waited), SR)
json.dump(report, open("results/streaming_demo.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
