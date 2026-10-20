import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "1"

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from engine import OnDemandExperts, PrefixCache, attach_speaker, build_model, synthesize
from prepare_data import encode, load_codec

PORT = int(os.getenv("TTS_PORT", "8775"))
STRENGTH = float(os.getenv("TTS_CFG", "4"))
SPEAKER = os.getenv("TTS_SPEAKER", "speaker.safetensors")
REF = os.getenv("TTS_REF", "ref/tikir_long")
TAGS = {
    "[whispering]": "[fısıldayarak]", "[scoffs]": "[küçümseyerek]", "[sighs]": "[iç çekerek]",
    "[excited]": "[heyecanla]", "[laughs]": "[gülerek]", "[chuckles]": "[kıkırdayarak]",
}

model, store = build_model(300, disk=True)
attach_speaker(model, SPEAKER)
OnDemandExperts.batch = 2
codec = load_codec()
audio, sr = sf.read(REF + ".wav", dtype="float32")
ref_codes = encode(codec, audio if audio.ndim == 1 else audio.mean(1), sr).long()
ref_text = open(REF + ".txt", encoding="utf-8").read().strip()
prefix = PrefixCache(model, ref_text, ref_codes)
lock = threading.Lock()


def speak(text, strength=STRENGTH):
    for en, tr in TAGS.items():
        text = text.replace(en, tr)
    with lock:
        wave, info = synthesize(model, prefix, codec, text, ref_codes, ref_text, temperature=1.0, top_p=0.9, strength=strength)
    print(f"[tts] {info} {text[:60]}", flush=True)
    return wave


speak("Merhaba.")
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
                wave = speak(text, float(req.get("cfg", STRENGTH)))
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
