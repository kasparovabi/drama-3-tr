import json
import os
import sys

import librosa
import soundfile as sf
import soxr
from faster_whisper import WhisperModel
from jiwer import cer, wer

sys.path.insert(0, "data/freya_tr_eval")
from benchmark import normalize_tr, transcribe

# librosa.resample hangs in this environment, this is the same call as its default (soxr_hq)
librosa.resample = lambda y, orig_sr, target_sr: soxr.resample(y, orig_sr, target_sr, quality="HQ")

WHISPER = os.environ.get("WHISPER_MODEL", "large-v3")

root = sys.argv[1] if len(sys.argv) > 1 else "out/freya_tr_eval"
items = json.load(open(f"{root}/list.json", encoding="utf-8"))
asr = WhisperModel(WHISPER, device="cuda", compute_type="float16")
refs, hyps, rows = [], [], []
for k in items:
    wav, sr = sf.read(k["wav"], dtype="float32")
    ref, hyp = normalize_tr(k["text"]), normalize_tr(transcribe(asr, wav, sr))
    refs.append(ref)
    hyps.append(hyp)
    rows.append({"wav": k["wav"], "ref": ref, "hyp": hyp, "wer": round(wer(ref, hyp), 3), "stop": k["stop"]})
result = {
    "sentences": len(refs),
    "wer": round(wer(refs, hyps), 4),
    "cer": round(cer(refs, hyps), 4),
    "rtf": round(sum(k["gen_sec"] for k in items) / sum(k["sec"] for k in items), 3),
    "no_eos": sum(k["stop"] != "eos" for k in items),
    "sentences_with_errors": sum(r["wer"] > 0 for r in rows),
    "rows": rows,
}
os.makedirs("results", exist_ok=True)
json.dump(result, open(f"results/{root.rstrip('/').split('/')[-1]}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print({k: v for k, v in result.items() if k != "rows"})
for r in sorted(rows, key=lambda r: -r["wer"])[:15]:
    print(r["wer"], "|", r["ref"], "|", r["hyp"])
