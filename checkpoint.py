import json
import math
import mmap
import re
import struct
import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "fish-speech"))

from fish_speech.models.text2semantic.fish_omni.backbone import DecoderLayer  # noqa: E402
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig  # noqa: E402
from fish_speech.models.text2semantic.fish_omni.layers import LayerContext  # noqa: E402

CKPT = ROOT / "models" / "drama-3"
PREFIX = "model.language_model."
EXPERT_KEY = re.compile(
    r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)"
)


def config():
    with open(CKPT / "config.json", encoding="utf-8") as f:
        return FishOmniConfig.from_dict(json.load(f))


DTYPES = {"F32": torch.float32, "BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn}


class Shard:
    """Read-only mapping of a safetensors file. On Windows safe_open's copy-on-write mapping commits the whole file size."""

    def __init__(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
            self.map = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self.base = 8 + n

    def get_tensor(self, key):
        h = self.header[key]
        start, end = h["data_offsets"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            t = torch.frombuffer(self.map, dtype=torch.uint8, offset=self.base + start, count=end - start)
        return t.view(DTYPES[h["dtype"]]).reshape(h["shape"])


class Weights:
    def __init__(self, ckpt=CKPT):
        self.ckpt = Path(ckpt)
        with open(self.ckpt / "model.safetensors.index.json", encoding="utf-8") as f:
            self.index = json.load(f)["weight_map"]
        self.files = {}

    def get(self, key):
        shard = self.index[key]
        if shard not in self.files:
            self.files[shard] = Shard(self.ckpt / shard)
        return self.files[shard].get_tensor(key)

    def keys(self, prefix):
        return [k for k in self.index if k.startswith(prefix)]


@torch.no_grad()
def load_layer(cfg, i, weights, device="cuda"):
    with torch.device("meta"):
        layer = DecoderLayer(cfg.text, i, fp8=True, dtype=torch.bfloat16)
    layer.to_empty(device=device)
    targets = dict(layer.named_parameters())
    targets.update(layer.named_buffers())
    experts = layer.mlp.experts
    prefix = f"{PREFIX}layers.{i}."
    filled = set()
    for key in weights.keys(prefix):
        name = key[len(prefix) :]
        tensor = weights.get(key)
        match = EXPERT_KEY.fullmatch(name)
        if match is None:
            targets[name].copy_(tensor)
            filled.add(name)
            continue
        expert, proj, kind = int(match[1]), match[2], match[3]
        suffix = "scale_inv" if kind == "weight_scale_inv" else "weight"
        if proj == "down_proj":
            target = getattr(experts, f"w2_{suffix}")[expert]
        else:
            full = getattr(experts, f"w13_{suffix}")[expert]
            half = full.shape[0] // 2
            target = full[:half] if proj == "gate_proj" else full[half:]
        target.copy_(tensor)
        filled.add((expert, proj, kind))
    expected = len(targets) - 4 + experts.num_experts * 6
    assert len(filled) == expected, (i, len(filled), expected)
    return layer.eval()


def layer_context(cfg, length, device="cuda"):
    rotary = cfg.text.rotary_dim
    exponents = torch.arange(0, rotary, 2, dtype=torch.float32, device=device)
    freq = 1.0 / (cfg.text.rope_theta ** (exponents / rotary))
    pos = torch.arange(length, device=device)
    angle = pos.float()[:, None] * freq[None, :]
    angle = torch.cat((angle, angle), dim=-1)
    return LayerContext(
        input_pos=pos,
        cos=angle.cos(),
        sin=angle.sin(),
        kv_len=length,
        continue_state=False,
        use_cache=False,
    )


class InputEmbed:
    """Backbone input. Text rows use the token embedding, audio rows (token + c1) / sqrt(2)."""

    def __init__(self, cfg, weights):
        self.cfg = cfg
        self.text = weights.get(PREFIX + "embed_tokens.weight")
        self.code = weights.get("audio_decoder.codebook_embeddings.weight")
        self.vocab = cfg.audio_decoder.vocab_size

    def __call__(self, rows):
        main = rows[:, 0]
        emb = self.text[main].float()
        audio = (main >= self.cfg.semantic_start_token_id) & (
            main <= self.cfg.semantic_end_token_id
        )
        extra = self.code[rows[:, 2] + self.vocab].float()
        mixed = (emb + extra) / math.sqrt(2)
        return torch.where(audio[:, None], mixed, emb).to(torch.bfloat16)


class Head:
    """Final norm plus the lm_head rows for eos and the semantic band only."""

    def __init__(self, cfg, weights, device="cuda"):
        self.candidates = torch.cat(
            (
                torch.tensor([cfg.eos_token_id]),
                torch.arange(cfg.semantic_start_token_id, cfg.semantic_end_token_id + 1),
            )
        )
        self.weight = weights.get("lm_head.weight")[self.candidates].to(device)
        self.norm = weights.get(PREFIX + "norm.weight").to(device).float()
        self.eps = cfg.text.rms_norm_eps
        self.index = torch.full((cfg.text.vocab_size,), -1, dtype=torch.long)
        self.index[self.candidates] = torch.arange(len(self.candidates))

    def __call__(self, hidden):
        x = hidden.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = (x * (1.0 + self.norm)).to(torch.bfloat16)
        return (x @ self.weight.T).float()
