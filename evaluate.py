import json
import re
import sys
import unicodedata

from faster_whisper import WhisperModel


def normalize(text):
    text = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", text)
    text = text.replace("I", "ı").replace("İ", "i").lower()
    text = "".join(c for c in unicodedata.normalize("NFC", text) if c.isalnum() or c.isspace())
    return " ".join(text.split())


def distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main(list_path, out):
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
    result = []
    for row in json.load(open(list_path, encoding="utf-8")):
        segments, _ = model.transcribe(row["wav"], language="tr", beam_size=5)
        heard = " ".join(s.text.strip() for s in segments)
        r, h = normalize(row["text"]), normalize(heard)
        result.append({
            **row,
            "heard": heard,
            "cer": distance(r, h) / max(1, len(r)),
            "wer": distance(r.split(), h.split()) / max(1, len(r.split())),
        })
        print(json.dumps(result[-1], ensure_ascii=False), flush=True)
    json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
