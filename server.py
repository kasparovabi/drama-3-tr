import io
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "1"

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from engine import OnDemandExperts, PrefixCache, attach_speaker, build_model, hot_experts, synthesize
from prepare_data import encode, load_codec

PORT = int(os.getenv("TTS_PORT", "8775"))
STRENGTH = float(os.getenv("TTS_CFG", "4"))
SPEAKER = os.getenv("TTS_SPEAKER", "speaker.safetensors")
REF = os.getenv("TTS_REF", "ref/tikir_long")
ANCHOR = os.getenv("TTS_ANCHOR", REF)
CAMPPLUS = os.getenv("TTS_CAMPPLUS")
HOT = int(os.getenv("TTS_HOT", "0"))
TAGS = {
    "[whispering]": "[fısıldayarak]", "[scoffs]": "[küçümseyerek]", "[sighs]": "[iç çekerek]",
    "[excited]": "[heyecanla]", "[laughs]": "[gülerek]", "[chuckles]": "[kıkırdayarak]",
}

model, store = build_model(300, disk=True)
if HOT:
    store.pin_hot(hot_experts(HOT))
attach_speaker(model, SPEAKER)
OnDemandExperts.batch = 2
codec = load_codec()
lock = threading.Lock()
refs, current = {}, [None]


def reference(name):
    if not re.fullmatch(r"[\w/]+", name) or ".." in name:
        raise ValueError(f"bad ref {name!r}")
    if name not in refs:
        audio, sr = sf.read(name + ".wav", dtype="float32")
        codes = encode(codec, audio if audio.ndim == 1 else audio.mean(1), sr).long()
        text = open(name + ".txt", encoding="utf-8").read().strip()
        refs[name] = (codes, text, PrefixCache(model, text, codes))
    return refs[name]


def load_embedder():
    import glob
    import sys

    import torchaudio.compliance.kaldi as kaldi

    sys.path.insert(0, CAMPPLUS)
    from modules.campplus.DTDNN import CAMPPlus

    net = CAMPPlus(feat_dim=80, embedding_size=192)
    net.load_state_dict(torch.load(glob.glob(f"{CAMPPLUS}/checkpoints/models--funasr--campplus/snapshots/*/campplus_cn_common.bin")[0], map_location="cpu"))
    net.eval()

    def embed(wave, sr):
        x = torch.from_numpy(resample_poly(wave, 16000, sr).astype(np.float32))[None]
        f = kaldi.fbank(x, num_mel_bins=80, dither=0, sample_frequency=16000)
        with torch.no_grad():
            return torch.nn.functional.normalize(net((f - f.mean(0, keepdim=True))[None])[0], dim=-1)

    audio, sr = sf.read(ANCHOR + ".wav", dtype="float32")
    anchor = embed(audio if audio.ndim == 1 else audio.mean(1), sr)
    return lambda wave: float(embed(wave, codec.sample_rate) @ anchor)


similarity = load_embedder() if CAMPPLUS else None


def speak(text, strength=STRENGTH, ref=REF, best_of=1, temperature=1.0, top_p=0.9, follow=False):
    for en, tr in TAGS.items():
        text = text.replace(en, tr)
    takes = []
    with lock:
        ref_codes, ref_text, prefix = reference(ref)
        if current[0] != ref:
            prefix.rows = model.cfg_session = None
            current[0] = ref
        tries = best_of if similarity and not follow else 1
        for n in range(tries + 1):
            if n == tries and any(ok for ok, _, _ in takes):
                break
            wave, info = synthesize(model, prefix, codec, text, ref_codes, ref_text, temperature=temperature, top_p=top_p,
                                    strength=strength, max_frames=frame_cap(text), follow=follow)
            takes.append((info["stop"] == "eos", similarity(wave) if similarity else 0.0, wave))
    print(f"[tts] {info} {' '.join(f'{s:.2f}{"" if ok else "!"}' for ok, s, _ in takes)} {text[:60]}", flush=True)
    return max(takes, key=lambda t: t[:2])[2]


def frame_cap(text):
    return int(21.5 * (4 + len(text) / 5))


speak("Merhaba.")
model.cfg_session = None
print(f"[tts] READY http://127.0.0.1:{PORT}/tts GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            text = str(req.get("text", ""))
            body = b""
            if any(c.isalnum() for c in text):
                wave = speak(text, float(req.get("cfg", STRENGTH)), str(req.get("ref") or REF), min(4, max(1, int(req.get("best_of", 1)))),
                             float(req.get("temperature", 1.0)), float(req.get("top_p", 0.9)), bool(req.get("follow")))
                if req.get("format") == "wav":
                    buf = io.BytesIO()
                    sf.write(buf, wave, codec.sample_rate, format="WAV", subtype="PCM_16")
                    body = buf.getvalue()
                else:
                    body = (np.clip(resample_poly(wave, 160, 441), -1, 1) * 32767).astype(np.int16).tobytes()
            self.send_response(200)
        except Exception as e:
            print(f"[tts] error {e!r}", flush=True)
            body = b""
            self.send_response(500)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
