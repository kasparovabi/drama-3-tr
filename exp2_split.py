import argparse
import json
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from checkpoint import CKPT, Head, InputEmbed, Weights, config, layer_context, load_layer
from exp1_routing import AUDIO, build_rows


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--routing", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", default="16,32,64,128")
    p.add_argument("--count", type=int, default=150)
    p.add_argument("--arms", default="split,pruned")
    a = p.parse_args()

    cfg = config()
    tok = Tokenizer.from_file(str(CKPT / "tokenizer.json"))
    weights = Weights()
    records = torch.load(a.data)[: a.count]
    counts = torch.load(a.routing)["counts"]
    E = cfg.text.num_experts
    samples = [build_rows(r, tok, cfg) for r in records]
    embed = InputEmbed(cfg, weights)
    start = [embed(s) for s, _, _ in samples]
    del embed

    arms = [("full", 0)]
    for k in map(int, a.k.split(",")):
        arms += [(arm, k) for arm in a.arms.split(",")]
    hidden = {arm: list(start) for arm in arms}
    audio_rows = [(ph == AUDIO).cuda() for _, ph, _ in samples]

    for i in range(len(counts)):
        t0 = time.perf_counter()
        layer = load_layer(cfg, i, weights)
        rankings = {
            "audio": counts[i, AUDIO].argsort(descending=True).cuda(),
            "total": (counts[i, AUDIO] / counts[i, AUDIO].sum() + counts[i, 0] / counts[i, 0].sum())
            .argsort(descending=True)
            .cuda(),
        }
        state = {}

        def mask(module, inputs, out):
            if state.get("blocked") is None:
                return None
            out = out.clone()
            out[state["rows"][:, None] & state["blocked"][None, :]] = float("-inf")
            return out

        hook = layer.mlp.gate.register_forward_hook(mask)
        for arm in arms:
            kind, k = arm
            blocked = None
            if kind != "full":
                blocked = torch.ones(E, dtype=torch.bool, device="cuda")
                blocked[rankings["total" if kind == "pruned_total" else "audio"][:k]] = False
            for j, h in enumerate(hidden[arm]):
                state["blocked"] = blocked
                state["rows"] = (
                    audio_rows[j]
                    if kind == "split"
                    else torch.ones_like(audio_rows[j])
                )
                hidden[arm][j] = layer(h.cuda()[None], layer_context(cfg, h.shape[0]))[0].cpu()
        hook.remove()
        del layer
        torch.cuda.empty_cache()
        print(f"layer {i} {time.perf_counter() - t0:.1f}s", flush=True)

    head = Head(cfg, weights)
    result = {}
    for arm in arms:
        kl = nll = agree = total = 0.0
        for j, (rows, _, prompt_len) in enumerate(samples):
            teacher = F.log_softmax(head(hidden[("full", 0)][j][prompt_len - 1 : -1].cuda()), -1)
            student = F.log_softmax(head(hidden[arm][j][prompt_len - 1 : -1].cuda()), -1)
            target = head.index[rows[prompt_len:, 0]].cuda()
            kl += F.kl_div(student, teacher, log_target=True, reduction="sum").item()
            nll += F.nll_loss(student, target, reduction="sum").item()
            agree += (student.argmax(-1) == teacher.argmax(-1)).sum().item()
            total += len(target)
        result[f"{arm[0]}_{arm[1]}"] = {
            "kl": kl / total,
            "nll": nll / total,
            "agree": agree / total,
        }
        print(arm, json.dumps(result[f"{arm[0]}_{arm[1]}"]), flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
