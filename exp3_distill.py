import argparse
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from checkpoint import CKPT, Head, InputEmbed, Weights, config, layer_context, load_layer
from exp1_routing import AUDIO, build_rows

BF = torch.bfloat16
TAGS = [
    "öfkeyle", "kısık sesle", "neşeyle", "ağlamaklı", "yorgun bir sesle", "şaşkınlıkla", "alaycı bir tonla",
    "whispering", "laughs", "angry", "sad", "excited", "shouting", "nervously", "calmly", "sarcastic",
]


class SpeakerMoE(nn.Module):
    """MoE with k experts for audio rows. The shared expert comes from the teacher and stays frozen."""

    def __init__(self, block, selected):
        super().__init__()
        w13, w2 = block.experts.expert_weights(selected, torch.float32)
        self.w13 = nn.Parameter(w13.contiguous())
        self.w2 = nn.Parameter(w2.contiguous())
        self.gate = nn.Parameter(block.gate.weight.detach()[selected].float().clone())
        self.bias = nn.Parameter(torch.zeros(w13.shape[-1], device=w13.device))
        self.block = block
        self.top = block.top_k

    def forward(self, u):
        x = u.float()
        w, ids = torch.softmax(x @ self.gate.T, -1).topk(self.top, -1)
        w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for e in range(self.gate.shape[0]):
            row, col = (ids == e).nonzero(as_tuple=True)
            if len(row) == 0:
                continue
            g, up = (x[row] @ self.w13[e].T).chunk(2, -1)
            y = (F.silu(g) * up) @ self.w2[e].T
            out = out.index_add(0, row, y * w[row, col, None])
        with torch.no_grad():
            shared = torch.sigmoid(self.block.shared_expert_gate(u)) * self.block.shared_expert(u)
        return out + shared.float() + self.bias


def rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def split_layer(layer, h, ctx):
    mixer = layer.self_attn if layer.layer_type == "full_attention" else layer.linear_attn
    mid = h + mixer(layer.input_layernorm(h).to(BF), ctx)
    return mid, layer.post_attention_layernorm(mid).to(BF)


def train(student, U, Y, steps, batch):
    with torch.no_grad():
        residual = torch.cat([Y[b : b + 4096].float() - student(U[b : b + 4096]) for b in range(0, len(U), 4096)])
        student.bias.copy_(residual.mean(0))
        spread = Y.float().std(0)
        spread = spread + 0.05 * spread.median()
        start_loss = ((residual - residual.mean(0)) / spread).pow(2).mean().item()
    opt = torch.optim.Adam(
        [
            {"params": [student.gate, student.bias], "lr": 1e-3},
            {"params": [student.w13, student.w2], "lr": 3e-5},
        ]
    )
    g = torch.Generator(device="cuda").manual_seed(0)
    for _ in range(steps):
        s = torch.randint(0, len(U), (batch,), device="cuda", generator=g)
        loss = ((student(U[s]) - Y[s].float()) / spread).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return start_loss, loss.item()


def collect(layer, block, cfg, source, teacher, subset, mask, kind):
    U, Y = [], []
    with torch.inference_mode():
        for j in subset:
            hc = source[j].cuda()[None]
            mid, u = split_layer(layer, hc, layer_context(cfg, hc.shape[1]))
            m = mask[j]
            U.append(u[0][m])
            Y.append(block(u[0][m]) if kind == "function" else teacher[j].cuda()[m] - mid[0][m])
    return torch.cat(U).clone(), torch.cat(Y).clone()


def pack(student, selected):
    return {
        "selected": selected.cpu(),
        "gate": student.gate.detach().cpu(),
        "bias": student.bias.detach().cpu(),
        "w13": student.w13.detach().to(BF).cpu(),
        "w2": student.w2.detach().to(BF).cpu(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", default="32,64")
    p.add_argument("--train", type=int, default=300)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--layers", type=int, default=40)
    p.add_argument("--arms", default="select,function,trajectory")
    p.add_argument("--save")
    p.add_argument("--text_k", type=int, default=0)
    p.add_argument("--tag_rate", type=float, default=0.0)
    a = p.parse_args()

    cfg = config()
    tok = Tokenizer.from_file(str(CKPT / "tokenizer.json"))
    weights = Weights()
    data = torch.load(a.data)
    rng = random.Random(0)
    for record in data:
        if rng.random() < a.tag_rate:
            record["text"] = f"[{rng.choice(TAGS)}] {record['text']}"
    samples = [build_rows(r, tok, cfg) for r in data]
    train_set = range(a.train)
    test_set = range(a.train, len(samples))
    embed = InputEmbed(cfg, weights)
    start = [embed(s) for s, _, _ in samples]
    del embed
    audio = [(ph == AUDIO).cuda() for _, ph, _ in samples]
    text = [~m for m in audio]
    E = cfg.text.num_experts

    arms = [(kind, k) for k in map(int, a.k.split(",")) for kind in a.arms.split(",")]
    saved = []
    teacher = list(start)
    hidden = {arm: list(start) for arm in arms}
    del start
    log = []

    for i in range(a.layers):
        t0 = time.perf_counter()
        layer = load_layer(cfg, i, weights)
        block = layer.mlp
        count = torch.zeros(E, device="cuda")
        count_text = torch.zeros(E, device="cuda")
        with torch.inference_mode():
            for j, h in enumerate(teacher):
                hc = h.cuda()[None]
                mid, u = split_layer(layer, hc, layer_context(cfg, h.shape[0]))
                if j in train_set:
                    logits = block.gate(u[0][audio[j]])
                    count += torch.bincount(logits.topk(8, -1).indices.reshape(-1), minlength=E)
                    logits = block.gate(u[0][text[j]])
                    count_text += torch.bincount(logits.topk(8, -1).indices.reshape(-1), minlength=E)
                teacher[j] = (mid + block(u))[0].cpu()
        ranking = count.argsort(descending=True)
        ranking_text = count_text.argsort(descending=True)
        entry = {"layer": i}

        for arm in arms:
            kind, k = arm
            student = SpeakerMoE(block, ranking[:k])
            reader = SpeakerMoE(block, ranking_text[: a.text_k]) if a.text_k else None
            if kind != "select":
                U, Y = collect(layer, block, cfg, hidden[arm], teacher, train_set, audio, kind)
                entry[f"{kind}_{k}_loss"] = train(student, U, Y, a.steps, a.batch)
                del U, Y
                if reader is not None:
                    U, Y = collect(layer, block, cfg, hidden[arm], teacher, train_set, text, kind)
                    entry[f"{kind}_{k}_text_loss"] = train(reader, U, Y, a.steps, a.batch)
                    del U, Y
            drift = []
            with torch.no_grad():
                for j, h in enumerate(hidden[arm]):
                    hc = h.cuda()[None]
                    mid, u = split_layer(layer, hc, layer_context(cfg, h.shape[0]))
                    full = block(u)[0]
                    full[audio[j]] = student(u[0][audio[j]]).to(BF)
                    if reader is not None:
                        full[text[j]] = reader(u[0][text[j]]).to(BF)
                    new = mid[0] + full
                    hidden[arm][j] = new.cpu()
                    if j in test_set:
                        t = rms(teacher[j].cuda()[audio[j]].float())
                        drift.append(((rms(new[audio[j]].float()) - t).norm() / t.norm()).item())
            entry[f"{kind}_{k}_drift"] = sum(drift) / len(drift)
            if a.save == f"{kind}_{k}":
                saved.append(pack(student, ranking[:k]))
                if reader is not None:
                    saved[-1]["reader"] = pack(reader, ranking_text[: a.text_k])
                if i % 10 == 9 or i == a.layers - 1:
                    torch.save(saved, f"results/speaker_{a.save}{f'_t{a.text_k}' if a.text_k else ''}{'_tags' if a.tag_rate else ''}.pt")
            del student, reader
        del layer, block
        torch.cuda.empty_cache()
        log.append(entry)
        print(json.dumps(entry), f"{time.perf_counter() - t0:.0f}s", flush=True)

    head = Head(cfg, weights)
    result = {"log": log}
    with torch.no_grad():
        for arm in arms:
            for name, subset in (("test", test_set), ("train", train_set)):
                kl = nll = agree = total = 0.0
                for j in subset:
                    rows, _, n = samples[j]
                    t = F.log_softmax(head(teacher[j][n - 1 : -1].cuda()), -1)
                    s = F.log_softmax(head(hidden[arm][j][n - 1 : -1].cuda()), -1)
                    target = head.index[rows[n:, 0]].cuda()
                    kl += F.kl_div(s, t, log_target=True, reduction="sum").item()
                    nll += F.nll_loss(s, target, reduction="sum").item()
                    agree += (s.argmax(-1) == t.argmax(-1)).sum().item()
                    total += len(target)
                result[f"{arm[0]}_{arm[1]}_{name}"] = {"kl": kl / total, "nll": nll / total, "agree": agree / total}
                print(arm, name, json.dumps(result[f"{arm[0]}_{arm[1]}_{name}"]), flush=True)
        nll = total = 0.0
        for j in test_set:
            rows, _, n = samples[j]
            t = F.log_softmax(head(teacher[j][n - 1 : -1].cuda()), -1)
            nll += F.nll_loss(t, head.index[rows[n:, 0]].cuda(), reduction="sum").item()
            total += len(rows) - n
        result["teacher_test_nll"] = nll / total
        print("teacher test nll", nll / total)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
