# drama-3-tr

Run Fish Audio Drama 3 in Turkish on a single 16 GB GPU, close to real time for short sentences and almost twice as fast for paragraphs, with its inline directions and emotion intact. The full model reads the prompt, a distilled 32-expert speaker generates the audio, and experts are streamed from the NVMe drive so the 41.5 GB checkpoint never has to sit in RAM.

Weights and full results are on Hugging Face at [kasparovabi/drama-3-tr-speaker](https://huggingface.co/kasparovabi/drama-3-tr-speaker).

**Built with Fish Audio.** This project uses [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3) and [fish-speech](https://github.com/fishaudio/fish-speech). It is not affiliated with or endorsed by Fish Audio.

## Results in one table

| | Value |
|---|---|
| Freya-TR-Eval WER and CER (495 sentences, Freya's protocol) | 2.68% and 0.70% without guidance, 2.78% and 0.67% with guidance in disk mode |
| Teacher-forced KL to unmodified Drama 3 (200 held-out utterances) | 0.066 |
| Speed on RTX 5070 Ti, experts in RAM | 1.2x real time for short sentences, 1.7 to 1.9x for paragraphs |
| Speed with experts read from disk, guidance on | 0.86x real time for short sentences |
| First audio with streaming decode, experts in RAM | 1.15 s (short sentence) |
| VRAM and system RAM | 14.2 GiB, and 4.4 GB of RAM while generating (9.3 GB peak while loading) |

## Setup

Tested on Windows 11, Python 3.13, torch 2.14.1+cu132 and triton-windows 3.8.

```bash
git clone https://github.com/kasparovabi/drama-3-tr && cd drama-3-tr
git clone https://github.com/fishaudio/fish-speech && git -C fish-speech checkout 5415669
hf download fishaudio/drama-3 --local-dir models/drama-3
hf download kasparovabi/drama-3-tr-speaker speaker.safetensors --local-dir .
pip install torch triton-windows safetensors tokenizers soundfile omegaconf hydra-core scipy psutil einops einx loguru descript-audio-codec
```

## Generate speech

`sentences.json` is a JSON list of strings. Put directions in square brackets, in Turkish.

```bash
python engine.py --speaker speaker.safetensors --graph --slots 300 --disk \
  --ref my_voice.wav --ref_text my_voice.txt --sentences sentences.json --out out
```

`--cfg` sets the direction guidance strength (default 4, 0 turns it off) and `--seed` makes a run reproducible. `server.py` serves the same engine over HTTP (POST `/tts` with `{"text": ...}`, 16 kHz PCM back, or `"format": "wav"` for 44.1 kHz WAV). `streaming_demo.py` shows streaming playback, where audio is decoded while frames are still being generated.

## Evaluate on Freya-TR-Eval

Put `eval/benchmark.py` from the [FreyaTTS repository](https://github.com/freyavoiceai/FreyaTTS) and `freya_tr_eval.jsonl` from the [freya-tr-eval dataset](https://huggingface.co/datasets/freyavoice/freya-tr-eval) in `data/freya_tr_eval`, generate the 495 sentences with `--sentences ref/freya_tr_eval.json --out out/freya_tr_eval`, then score them.

```bash
pip install faster-whisper jiwer librosa soxr
python freya_score.py out/freya_tr_eval
```

## Train your own speaker

```bash
python prepare_data.py --archive <ISSAI TSC archive> --count 4000 --out data/tsc4000.pt
python exp3_distill.py --data data/tsc4000.pt --out results/distill.json \
  --k 32 --arms trajectory --train 3800 --steps 600 --save trajectory_32
python release/convert_weights.py results/speaker_trajectory_32.pt speaker.safetensors
```

Distilling 32 experts per layer on 3,800 Turkish utterances takes about 85 minutes on one RTX 5070 Ti. The same recipe should work for other languages with their own speech data, though we only tried Turkish.

## Files

| File | What it does |
|---|---|
| `engine.py` | Inference engine (reader, speaker, prefix cache, CUDA graphs, direction guidance, streaming decode) |
| `checkpoint.py` | Read-only checkpoint access and single-layer loading for experiments |
| `disk.py` | Unbuffered NVMe reader for experts |
| `prepare_data.py` | Codec loading and ISSAI TSC encoding |
| `exp1_routing.py`, `exp2_split.py` | Routing measurements and the reader/speaker split experiment |
| `exp3_distill.py` | Layer-by-layer speaker distillation (trajectory anchoring) |
| `evaluate.py`, `freya_score.py` | Whisper scoring on our own sets, and on Freya-TR-Eval with Freya's protocol |
| `server.py`, `streaming_demo.py` | HTTP server and streaming playback demo |
| `prefix_cache_test.py` | Checks that the cached reference prefix gives the same hidden state as a full read |
| `release/convert_weights.py` | Converts a trained speaker to FP8 safetensors |

## Related work

1. SlimWise (arXiv 2609.34117) separates prefill and decode experts for MoE language models. Our reader and speaker split is the same idea, found independently for TTS.
2. GPTAQ (ICML 2025) uses the same asymmetric objective as our trajectory anchoring, for quantization.
3. OpenMoE (ICML 2024) reported token-determined routing, which we also observe on Turkish text.
4. SMG-CFG studied guidance for emotion control in autoregressive TTS.
5. Freya-TR-Eval and its scoring script come from FreyaTTS (arXiv 2607.09530).

## License

Code in this repository is Apache 2.0 (see `LICENSE`). The speaker weights on Hugging Face are a Derivative Work of Drama 3 and fall under the Fish Audio Community License v1.0, which also governs any use of Drama 3 itself. Do not use it to imitate a real person's voice without consent.

Author Ahmet Kazankaya.
