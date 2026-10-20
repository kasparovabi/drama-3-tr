import json, sys
from faster_whisper import WhisperModel
m = WhisperModel(sys.argv[2], device="cuda", compute_type="int8_float16")
segments, _ = m.transcribe(sys.argv[1], language="tr", word_timestamps=True, beam_size=5)
out = [{"start": s.start, "end": s.end, "text": s.text.strip(), "words": [(w.start, w.end, w.word) for w in s.words]} for s in segments]
json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for s in out: print(f"{s['start']:.1f}-{s['end']:.1f} {s['text']}")
