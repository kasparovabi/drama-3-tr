import argparse
import json
import os
import time
from collections import OrderedDict

os.environ.setdefault("FISH_DRAMA3_CUDA_GRAPH", "0")

import soundfile as sf
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from checkpoint import CKPT, PREFIX, Weights, config
from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone, backbone_name
from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_moe_forward
from fish_speech.models.text2semantic.fish_omni.generate import RAS_WINDOW_FRAMES, Drama3DecodeGraphs, build_drama3_prompt, ensure_drama3_caches
from fish_speech.models.text2semantic.fish_omni.model import AUDIO_DECODER_PREFIXES, FishOmniModel
from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder, load_audio_decoder_weights

BF = torch.bfloat16
PROJ = ("gate_proj", "up_proj", "down_proj")


class ExpertStore:
    """Loads experts on demand and keeps them in LRU slots on the GPU."""

    def __init__(self, cfg, weights, slots):
        I, H = cfg.text.moe_intermediate_size, cfg.text.hidden_size
        self.weights = weights
        self.w13 = torch.empty((slots, 2 * I, H), dtype=torch.uint8, device="cuda")
        self.w2 = torch.empty((slots, H, I), dtype=torch.uint8, device="cuda")
        self.s13 = torch.empty((slots, 2 * I // 128, H // 128), device="cuda")
        self.s2 = torch.empty((slots, H // 128, I // 128), device="cuda")
        self.lru = OrderedDict()
        self.free = list(range(slots))
        self.misses = self.hits = 0
        self.disk = None

    def _load(self, layer, missing):
        I = self.w13.shape[1] // 2
        if self.disk is not None and missing:
            return self._load_from_disk(layer, missing, I)
        for e, s in missing:
            prefix = f"{PREFIX}layers.{layer}.mlp.experts.{e}."
            get = lambda p, t: self.weights.get(f"{prefix}{p}.{t}")
            self.w13[s, :I].copy_(get("gate_proj", "weight").view(torch.uint8))
            self.w13[s, I:].copy_(get("up_proj", "weight").view(torch.uint8))
            self.w2[s].copy_(get("down_proj", "weight").view(torch.uint8))
            self.s13[s, : I // 128].copy_(get("gate_proj", "weight_scale_inv"))
            self.s13[s, I // 128 :].copy_(get("up_proj", "weight_scale_inv"))
            self.s2[s].copy_(get("down_proj", "weight_scale_inv"))

    def _load_from_disk(self, layer, missing, I):
        d = self.disk
        for b in range(0, len(missing), d.slots):
            batch = missing[b : b + d.slots]
            keys = [(layer, e) for e, _ in batch]
            offsets = d.fetch(keys)
            for (e, s), k, off in zip(batch, keys, offsets):
                root = d.where[k][0]
                v = lambda proj: d.view(off, proj)
                self.w13[s, :I].copy_(v("gate_proj").view(I, -1), non_blocking=True)
                self.w13[s, I:].copy_(v("up_proj").view(I, -1), non_blocking=True)
                self.w2[s].copy_(v("down_proj").view(-1, I), non_blocking=True)
                self.s13[s, : I // 128].copy_(d.scales[(root, "gate_proj")])
                self.s13[s, I // 128 :].copy_(d.scales[(root, "up_proj")])
                self.s2[s].copy_(d.scales[(root, "down_proj")])
            torch.cuda.current_stream().synchronize()

    def slots_for(self, layer, experts):
        result, missing = [], []
        for e in experts:
            key = (layer, e)
            if key in self.lru:
                self.lru.move_to_end(key)
                self.hits += 1
            else:
                self.misses += 1
                s = self.free.pop() if self.free else self.lru.popitem(last=False)[1]
                missing.append((e, s))
                self.lru[key] = s
            result.append(self.lru[key])
        self._load(layer, missing)
        return result


class OnDemandExperts(nn.Module):
    light = False
    batch = 1

    def __init__(self, layer, store, count):
        super().__init__()
        self.layer, self.store = layer, store
        self.slot_map = torch.full((count,), -1, dtype=torch.long, device="cuda")
        self.speaker = self.reader = None

    def forward(self, x, ids, weights):
        if self.speaker is not None and x.shape[0] <= OnDemandExperts.batch:
            return self.speaker(x)
        if self.reader is not None and OnDemandExperts.light:
            return self.reader(x)
        unique = ids.unique()
        self.slot_map[unique] = torch.tensor(self.store.slots_for(self.layer, unique.tolist()), device="cuda")
        s = self.store
        fp8 = torch.float8_e4m3fn
        return fp8_moe_forward(x, self.slot_map[ids], weights, s.w13.view(fp8), s.s13, s.w2.view(fp8), s.s2)


@torch.no_grad()
def build_model(slots=1200, disk=False):
    cfg = config()
    weights = Weights()
    with torch.device("meta"):
        backbone = FishOmniBackbone(cfg.text, fp8_experts=True, dtype=BF)
        tower = FishOmniAudioDecoder(cfg, dtype=BF)
    store = ExpertStore(cfg, weights, slots)
    if disk:
        from disk import DiskExperts

        store.disk = DiskExperts(256, 16)
    for i, layer in enumerate(backbone.model.layers):
        layer.mlp.experts = OnDemandExperts(i, store, cfg.text.num_experts)
    backbone.materialize(["cuda"])
    targets = dict(backbone.named_parameters())
    targets.update(backbone.named_buffers())
    filled = set()
    for key in weights.index:
        if ".mlp.experts." in key or key.startswith(AUDIO_DECODER_PREFIXES):
            continue
        name = backbone_name(key)
        targets[name].copy_(weights.get(key))
        filled.add(name)
    missing = [n for n in targets if n not in filled and "slot_map" not in n]
    assert not missing, missing[:5]
    tower.to_empty(device="cuda")
    load_audio_decoder_weights(
        tower,
        ((k, weights.get(k)) for k in weights.index if k.startswith(AUDIO_DECODER_PREFIXES)),
    )
    if disk:
        weights.files.clear()
        store.weights = None
    tok = Tokenizer.from_file(str(CKPT / "tokenizer.json"))
    return FishOmniModel(cfg, backbone, tower, tok).eval(), store


def fp8_block(w):
    *lead, r, c = w.shape
    b = w.float().view(*lead, r // 128, 128, c // 128, 128)
    scale = b.abs().amax(dim=(-3, -1)).clamp(min=1e-12) / 448.0
    q = (b / scale[..., :, None, :, None]).view(*lead, r, c).to(torch.float8_e4m3fn)
    return q, scale.float()


class Speaker:
    def __init__(self, p):
        self.gate = p["gate"].cuda().float()
        self.bias = p["bias"].cuda().float()
        if "s13" in p:
            self.w13, self.s13, self.w2, self.s2 = (p[k].cuda() for k in ("w13", "s13", "w2", "s2"))
        else:
            self.w13, self.s13 = fp8_block(p["w13"].cuda())
            self.w2, self.s2 = fp8_block(p["w2"].cuda())

    def __call__(self, x):
        probs = torch.softmax(x.float() @ self.gate.T, -1)
        w, ids = probs.topk(8, -1)
        w = (w / w.sum(-1, keepdim=True)).to(x.dtype)
        y = fp8_moe_forward(x, ids, w, self.w13, self.s13, self.w2, self.s2)
        return (y.float() + self.bias).to(x.dtype)


def load_speaker(path):
    if not str(path).endswith(".safetensors"):
        return torch.load(path, mmap=True)
    from safetensors.torch import load_file

    flat, layers = load_file(path), []
    for key, t in flat.items():
        i, name = key.split(".", 2)[1:]
        layers += [{} for _ in range(int(i) + 1 - len(layers))]
        layers[int(i)][name] = t
    return layers


def attach_speaker(model, path):
    for layer, p in zip(model.backbone.model.layers, load_speaker(path)):
        layer.mlp.experts.speaker = Speaker(p)
        if "reader" in p:
            layer.mlp.experts.reader = Speaker(p["reader"])


class PrefixCache:
    """Reads the system and reference rows, identical for every sentence, once and keeps the DeltaNet states.
    The attention KV of those rows stays in the model cache, so it must be cleared whenever the cache is rebuilt."""

    def __init__(self, model, ref_text, ref_codes):
        a, b = (build_drama3_prompt(t, model.tokenizer, model.config, [ref_text], [ref_codes]) for t in ("a", "b"))
        m = min(len(a), len(b))
        self.length = int((a[:m] != b[:m]).any(-1).nonzero()[0]) - 1
        self.layers = [l.linear_attn for l in model.backbone.model.layers if l.layer_type != "full_attention"]
        self.rows = self.states = None

    def restore_or_read(self, rows, read):
        P = self.length
        if self.rows is not None and torch.equal(rows[:P], self.rows):
            for m, (c, r) in zip(self.layers, self.states):
                m.conv_state.copy_(c)
                m.recurrent_state.copy_(r)
            return
        read(0, P)
        self.rows = rows[:P].clone()
        self.states = [(m.conv_state.clone(), m.recurrent_state.clone()) for m in self.layers]


def chunks(a, b, size=128):
    s = list(range(a, b, size)) + [b]
    # the speaker treats a one-row input as an audio frame, so no prefill chunk may be a single row
    if len(s) > 2 and s[-1] - s[-2] == 1:
        s[-2] -= 1
    return zip(s[:-1], s[1:])


def read_prompt(model, prefix, frames, pos, n):
    backbone, tower = model.backbone, model.audio_decoder

    def read(a, b):
        hidden = None
        for s, e in chunks(a, b):
            part = frames[s:e]
            hidden = backbone(inputs_embeds=tower.embed_backbone(part, backbone.embed(part[:, 0]))[None], input_pos=pos[s:e])[0]
            paired = min(e, n - 1)
            if paired > s:
                tower.prefill(hidden[: paired - s], pos[s:paired], frames[s:paired], frames[s + 1 : paired + 1])
        return hidden

    prefix.restore_or_read(frames, read)
    OnDemandExperts.light = True
    try:
        return read(prefix.length, n)[-1:]
    finally:
        OnDemandExperts.light = False


@torch.inference_mode()
def generate_frames(model, prefix, prompt, max_frames, temperature, top_k, top_p):
    backbone, tower = model.backbone, model.audio_decoder
    n = len(prompt)
    capacity = Drama3DecodeGraphs.graph_capacity(model)
    capturing = os.environ.get("FISH_DRAMA3_CUDA_GRAPH", "1") != "0" and model.decode_graphs is None
    if capturing or model.cache_capacity < capacity:
        prefix.rows = None
    graphs = Drama3DecodeGraphs.for_model(model, capacity)
    ensure_drama3_caches(model, capacity)
    frames = torch.zeros((n + max_frames, prompt.shape[1]), dtype=torch.long, device="cuda")
    frames[:n] = prompt.cuda()
    pos = torch.arange(n + max_frames, device="cuda")
    hidden = read_prompt(model, prefix, frames, pos, n)
    made, stop = 0, "length"
    if graphs is not None:
        graphs.start(hidden, n - 1, frames[:n], temperature, top_k, top_p)
    for step in range(max_frames):
        k = n - 1 + step
        if graphs is not None:
            sample, main = graphs.step(k)
        else:
            sample = tower.decode_step(
                backbone.compute_logits(hidden), hidden, pos[k : k + 1], frames[k : k + 1], frames[: k + 1, 0],
                temperature=temperature, top_k=top_k, top_p=top_p,
            )
            main = sample[0, 0].item()
        if main == model.config.eos_token_id:
            stop = "eos"
            break
        frames[k + 1] = sample[0]
        made += 1
        if graphs is None:
            hidden = backbone(inputs_embeds=tower.embed_backbone(sample, backbone.embed(sample[:, 0]))[None], input_pos=pos[k + 1 : k + 2])[0]
    return frames[n : n + made].cpu(), stop


def strip_directions(prompt, tok, start):
    """Replaces text tokens inside square brackets with spaces, so length and positions stay the same."""
    plain = prompt.clone()
    space, inside = tok.encode(" ").ids[0], False
    for i in range(start, len(prompt)):
        if prompt[i, 1:].any():
            continue
        piece = tok.id_to_token(int(prompt[i, 0])) or ""
        if "[" in piece:
            inside = True
        if inside:
            plain[i, 0] = space
        if "]" in piece:
            inside = False
    return plain


def read_prompt_cfg(model, prefix, fA, fB, pos, n):
    backbone, tower = model.backbone, model.audio_decoder

    def read(a, b):
        hidden = None
        for s, e in chunks(a, b):
            g = torch.stack([tower.embed_backbone(f[s:e], backbone.embed(f[s:e, 0])) for f in (fA, fB)])
            hidden = backbone(inputs_embeds=g, input_pos=pos[s:e])
            paired = min(e, n - 1)
            if paired > s:
                tower.prefill(hidden[0, : paired - s], pos[s:paired], fA[s:paired], fA[s + 1 : paired + 1])
        return hidden

    prefix.restore_or_read(fA, read)
    return read(prefix.length, n)[:, -1]


class CfgGraphs:
    """CUDA graphs of one decode step that advances the directed and the undirected sequence together."""

    def __init__(self, model, capacity):
        self.model = model
        H = model.config.text.hidden_size
        self.hidden = torch.zeros((2, H), dtype=model.backbone.dtype, device="cuda")
        self.next = torch.zeros_like(self.hidden)
        self.pos = torch.zeros(1, dtype=torch.long, device="cuda")
        self.frame = torch.zeros((1, 1 + model.config.audio_decoder.num_codebooks), dtype=torch.long, device="cuda")
        self.sample = torch.zeros_like(self.frame)
        self.history = torch.full((RAS_WINDOW_FRAMES,), -1, dtype=torch.long, device="cuda")
        self.strength = torch.zeros((), device="cuda")
        self.temperature = torch.zeros((), device="cuda")
        self.top_k = torch.zeros((), dtype=torch.long, device="cuda")
        self.top_p = torch.zeros((), device="cuda")
        self.pool = torch.cuda.graph_pool_handle()
        self.buckets = tuple(sorted({min(b, capacity) for b in (1024, 2048, 4096, capacity)}))
        self.graphs = {}
        for kv in self.buckets:
            self.capture(kv)

    def attentions(self):
        tower = self.model.audio_decoder
        return [l.transformer_layer.attention for l in (*tower.shared_layers, *tower.residual_layers)]

    def run(self, kv):
        backbone, tower = self.model.backbone, self.model.audio_decoder
        logits = backbone.compute_logits(self.hidden).float()
        guided = logits[1:] + self.strength * (logits[:1] - logits[1:])
        sample = tower.decode_step(
            guided, self.hidden[:1], self.pos, self.frame, self.history,
            temperature=self.temperature, top_k=self.top_k, top_p=self.top_p, check_finite=False,
        )
        self.sample.copy_(sample)
        g = tower.embed_backbone(sample, backbone.embed(sample[:, 0]))
        h = backbone(inputs_embeds=g[None].expand(2, -1, -1).contiguous(), input_pos=self.pos + 1, static_kv_len=kv)
        self.next.copy_(h[:, 0])

    def capture(self, kv):
        for a in self.attentions():
            a.static_kv_len = kv
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.run(kv)
            torch.cuda.current_stream().wait_stream(stream)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self.run(kv)
            self.graphs[kv] = g
        finally:
            for a in self.attentions():
                a.static_kv_len = None

    def start(self, hidden, pos, frames, strength, temperature, top_k, top_p):
        self.hidden.copy_(hidden)
        self.pos.fill_(pos)
        self.frame.copy_(frames[pos : pos + 1])
        window = frames[: pos + 1, 0][-RAS_WINDOW_FRAMES:]
        self.history.fill_(-1)
        self.history[RAS_WINDOW_FRAMES - len(window) :].copy_(window)
        self.strength.fill_(strength)
        self.temperature.fill_(temperature)
        self.top_k.fill_(top_k)
        self.top_p.fill_(top_p)

    def step(self, pos):
        self.graphs[next(b for b in self.buckets if b >= pos + 2)].replay()
        sample = self.sample.clone()
        self.hidden.copy_(self.next)
        self.frame.copy_(sample)
        self.history.copy_(torch.cat((self.history[1:], sample[0, :1])))
        self.pos.add_(1)
        return sample, int(sample[0, 0])


@torch.inference_mode()
def generate_frames_cfg(model, prefix, prompt, strength, max_frames, temperature, top_k, top_p, stream=None):
    backbone, tower = model.backbone, model.audio_decoder
    n = len(prompt)
    capacity = Drama3DecodeGraphs.graph_capacity(model)
    if backbone.max_batch_size != 2:
        backbone.setup_caches(max_batch_size=2, max_seq_len=capacity)
        tower.setup_caches(capacity)
        model.cache_capacity = capacity
        model.cfg_graphs = CfgGraphs(model, capacity) if os.environ.get("FISH_DRAMA3_CUDA_GRAPH") == "1" else None
        prefix.rows = None
    fA = torch.zeros((n + max_frames, prompt.shape[1]), dtype=torch.long, device="cuda")
    fA[:n] = prompt.cuda()
    fB = fA.clone()
    fB[:n] = strip_directions(prompt, model.tokenizer, prefix.length).cuda()
    pos = torch.arange(n + max_frames, device="cuda")
    hidden = read_prompt_cfg(model, prefix, fA, fB, pos, n)
    made, stop = 0, "length"
    graphs = model.cfg_graphs
    if graphs is not None:
        graphs.start(hidden, n - 1, fA, strength, temperature, top_k, top_p)
        for step in range(max_frames):
            sample, main = graphs.step(n - 1 + step)
            if main == model.config.eos_token_id:
                stop = "eos"
                break
            fA[n + step] = sample[0]
            made += 1
            if stream:
                stream(fA[n : n + made])
        return fA[n : n + made].cpu(), stop
    for step in range(max_frames):
        k = n - 1 + step
        logits = backbone.compute_logits(hidden).float()
        guided = logits[1:] + strength * (logits[:1] - logits[1:])
        sample = tower.decode_step(
            guided, hidden[:1], pos[k : k + 1], fA[k : k + 1], fA[: k + 1, 0],
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
        if sample[0, 0].item() == model.config.eos_token_id:
            stop = "eos"
            break
        fA[k + 1] = sample[0]
        made += 1
        if stream:
            stream(fA[n : n + made])
        g = tower.embed_backbone(sample, backbone.embed(sample[:, 0]))
        hidden = backbone(inputs_embeds=g[None].expand(2, -1, -1).contiguous(), input_pos=pos[k + 1 : k + 2])[:, 0]
    return fA[n : n + made].cpu(), stop


class StreamDecoder:
    """Turns generated frames into audio piece by piece. The codec is causal. Long-range context lives only in the
    quantizer's windowed attention (8 layers x 128 frames), the convolutional decoder needs only a few frames of overlap."""

    def __init__(self, codec, sink, first=6, step=8, context=1024, overlap=8):
        self.codec, self.sink = codec, sink
        self.first, self.step, self.context, self.overlap = first, step, context, overlap
        self.written = 0

    def __call__(self, frames, last=False):
        T = len(frames)
        need = self.first if self.written == 0 else self.step
        if T == self.written or (not last and T - self.written < need):
            return
        start = max(0, self.written - self.context)
        z = self.codec.quantizer.decode(frames[start:T, 1:].T[None].clone())
        tail = max(start, self.written - self.overlap)
        ratio = z.shape[-1] // (T - start)
        audio = self.codec.decoder(z[..., (tail - start) * ratio :])[0].squeeze()
        hop = audio.shape[-1] // (T - tail)
        self.sink(audio[(self.written - tail) * hop :].float().cpu().numpy())
        self.written = T


def synthesize_to_file(model, prefix, codec, text, ref_codes, ref_text, out, **opts):
    audio, info = synthesize(model, prefix, codec, text, ref_codes, ref_text, **opts)
    sf.write(out, audio, codec.sample_rate)
    return info


def synthesize(model, prefix, codec, text, ref_codes, ref_text, temperature=0.7, top_p=0.7, top_k=30, max_frames=1500, strength=0.0, stream=None):
    prompt = build_drama3_prompt(text, model.tokenizer, model.config, [ref_text], [ref_codes])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if strength or stream:
        frames, stop = generate_frames_cfg(model, prefix, prompt, strength, max_frames, temperature, top_k, top_p, stream)
        if stream:
            with torch.inference_mode():
                stream(frames.cuda(), last=True)
    else:
        frames, stop = generate_frames(model, prefix, prompt, max_frames, temperature, top_k, top_p)
    elapsed = time.perf_counter() - t0
    codes = frames[:, 1:].T.contiguous()
    with torch.inference_mode():
        audio = codec.from_indices(codes[None].cuda())[0].squeeze().float().cpu().numpy()
    return audio, {"frames": len(frames), "sec": round(len(audio) / codec.sample_rate, 2), "gen_sec": round(elapsed, 1), "stop": stop}


def main():
    from prepare_data import encode, load_codec

    p = argparse.ArgumentParser()
    p.add_argument("--sentences", default="ref/sentences.json")
    p.add_argument("--ref", default="ref/tikir_long.wav")
    p.add_argument("--ref_text", default="ref/tikir_long.txt")
    p.add_argument("--speaker")
    p.add_argument("--out", default="out")
    p.add_argument("--slots", type=int, default=1200)
    p.add_argument("--graph", action="store_true")
    p.add_argument("--disk", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--seed", type=int)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, store = build_model(a.slots, a.disk)
    if a.speaker:
        attach_speaker(model, a.speaker)
        if a.graph:
            os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "1"
    if a.cfg:
        OnDemandExperts.batch = 2
    print(f"model ready, GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    codec = load_codec()
    audio, sr = sf.read(a.ref, dtype="float32")
    ref_codes = encode(codec, audio if audio.ndim == 1 else audio.mean(1), sr).long()
    ref_text = open(a.ref_text, encoding="utf-8").read().strip()
    prefix = PrefixCache(model, ref_text, ref_codes)
    torch.cuda.reset_peak_memory_stats()
    results = []
    for n, text in enumerate(json.load(open(a.sentences, encoding="utf-8"))):
        if a.seed is not None:
            torch.manual_seed(a.seed)
        wav = f"{a.out}/{n:02d}.wav"
        info = synthesize_to_file(model, prefix, codec, text, ref_codes, ref_text, wav, temperature=a.temperature, top_p=a.top_p, strength=a.cfg)
        info["hit_rate"] = round(store.hits / max(1, store.hits + store.misses), 3)
        info["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        info["ram_gb"] = round(__import__("psutil").Process().memory_info().rss / 1e9, 1)
        results.append({"wav": wav, "text": text, **info})
        print(n, info, flush=True)
        json.dump(results, open(f"{a.out}/list.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
