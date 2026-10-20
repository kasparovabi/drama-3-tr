import argparse
import io
import sys
import tarfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "fish-speech"))


def load_codec(device="cuda"):
    cfg = OmegaConf.load(str(ROOT / "fish-speech/fish_speech/configs/modded_dac_vq.yaml"))
    codec = instantiate(cfg)
    state = torch.load(
        ROOT / "models/drama-3/codec.pth", map_location="cpu", mmap=True, weights_only=True
    )
    state = state.get("state_dict", state)
    if any("generator" in k for k in state):
        state = {k.replace("generator.", ""): v for k, v in state.items() if "generator." in k}
    rebuilt = {
        f"{name}.{b}" if name else b
        for name, m in codec.named_modules()
        for b in m._non_persistent_buffers_set
    }
    codec.load_state_dict({k: v for k, v in state.items() if k not in rebuilt}, strict=True)
    codec.eval()
    for m in list(codec.modules()):
        if is_parametrized(m, "weight"):
            remove_parametrizations(m, "weight", leave_parametrized=True)
        elif hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)
    for m in codec.modules():
        if "causal_mask" in m._buffers:
            # shortcut: the 32768x32768 mask takes 1 GB, sequences longer than 8192 positions raise IndexError, grow it if long audio is needed
            m.causal_mask = m.causal_mask[:8192, :8192].clone()
    return codec.to(device=device, dtype=torch.bfloat16)


@torch.inference_mode()
def encode(codec, audio, sr):
    target = codec.sample_rate
    if sr != target:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target)
    x = torch.from_numpy(audio.astype(np.float32)).to(codec.device, torch.bfloat16)[None, None]
    return codec.encode(x, torch.tensor([x.shape[-1]], device=codec.device))[0][0].cpu()


def iter_tsc(archive, split, count):
    pending = {}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if f"/{split}/" not in member.name or not member.isfile():
                continue
            uid, ext = member.name.rsplit("/", 1)[1].rsplit(".", 1)
            data = tar.extractfile(member).read()
            item = pending.setdefault(uid, {})
            item[ext] = data
            if {"wav", "txt"} <= item.keys():
                del pending[uid]
                yield uid, item["txt"].decode("utf-8").strip(), item["wav"]
                count -= 1
                if count == 0:
                    return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True)
    p.add_argument("--split", default="Train")
    p.add_argument("--count", type=int, default=400)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    codec = load_codec()
    records = []
    for uid, text, wav in iter_tsc(a.archive, a.split, a.count):
        audio, sr = sf.read(io.BytesIO(wav), dtype="float32")
        codes = encode(codec, audio, sr)
        records.append({"id": uid, "text": text, "codes": codes.to(torch.int16)})
        if len(records) % 50 == 0:
            print(len(records), codes.shape, flush=True)
    torch.save(records, a.out)
    print("done", len(records), sum(r["codes"].shape[1] for r in records), "frames")


if __name__ == "__main__":
    main()
