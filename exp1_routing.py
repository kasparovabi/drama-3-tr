import argparse
import json
import time

import torch
from tokenizers import Tokenizer

from checkpoint import CKPT, Head, InputEmbed, Weights, config, layer_context, load_layer
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt

TEXT, AUDIO = 0, 1


def build_rows(record, tok, cfg):
    prompt = build_drama3_prompt(record["text"], tok, cfg)
    codes = record["codes"].long()
    audio = torch.cat((codes[:1] + cfg.semantic_start_token_id, codes)).T
    end = torch.zeros((1, prompt.shape[1]), dtype=torch.long)
    end[0, 0] = cfg.eos_token_id
    rows = torch.cat((prompt, audio, end))
    phase = torch.full((len(rows),), TEXT, dtype=torch.int8)
    phase[len(prompt) : len(prompt) + audio.shape[0]] = AUDIO
    return rows, phase, len(prompt)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", type=int, default=40)
    a = p.parse_args()

    cfg = config()
    tok = Tokenizer.from_file(str(CKPT / "tokenizer.json"))
    weights = Weights()
    records = torch.load(a.data)
    embed = InputEmbed(cfg, weights)
    samples = [build_rows(r, tok, cfg) for r in records]
    hidden = [embed(s) for s, _, _ in samples]
    phases = torch.cat([ph for _, ph, _ in samples]).long()
    del embed

    E, K = cfg.text.num_experts, cfg.text.num_experts_per_tok
    counts = torch.zeros(a.layers, 2, E, dtype=torch.float64)
    mass = torch.zeros(a.layers, 2, E, dtype=torch.float64)
    choices = torch.zeros(a.layers, len(phases), K, dtype=torch.int16)

    for i in range(a.layers):
        t0 = time.perf_counter()
        layer = load_layer(cfg, i, weights)
        captured = []
        hook = layer.mlp.gate.register_forward_hook(
            lambda m, g, c: captured.append(c.reshape(-1, E).float())
        )
        for j, h in enumerate(hidden):
            hidden[j] = layer(h.cuda()[None], layer_context(cfg, h.shape[0]))[0].cpu()
        hook.remove()
        probs = torch.softmax(torch.cat(captured), dim=-1)
        top = probs.topk(K, dim=-1).indices.cpu()
        choices[i] = top.to(torch.int16)
        probs = probs.cpu().double()
        for phase in (TEXT, AUDIO):
            selected = phases == phase
            counts[i, phase] = torch.bincount(top[selected].reshape(-1), minlength=E).double()
            mass[i, phase] = probs[selected].sum(0)
        del layer
        torch.cuda.empty_cache()
        print(f"layer {i} {time.perf_counter() - t0:.1f}s", flush=True)

    head = Head(cfg, weights)
    loss, correct, total = 0.0, 0, 0
    for (rows, phase, prompt_len), h in zip(samples, hidden):
        logits = head(h[prompt_len - 1 : -1].cuda())
        target = head.index[rows[prompt_len:, 0]].cuda()
        loss += torch.nn.functional.cross_entropy(logits, target, reduction="sum").item()
        correct += (logits.argmax(-1) == target).sum().item()
        total += len(target)
    summary = {
        "sentences": len(samples),
        "rows": len(phases),
        "audio_rows": int((phases == AUDIO).sum()),
        "c0_nll": loss / total,
        "c0_accuracy": correct / total,
    }
    print(json.dumps(summary, ensure_ascii=False))
    torch.save(
        {"counts": counts, "mass": mass, "choices": choices, "phases": phases, "summary": summary},
        a.out,
    )


if __name__ == "__main__":
    main()
