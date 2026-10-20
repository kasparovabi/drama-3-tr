import soundfile as sf
import torch

import engine
from engine import PrefixCache, build_model, read_prompt
from fish_speech.models.text2semantic.fish_omni.generate import Drama3DecodeGraphs, build_drama3_prompt, ensure_drama3_caches
from prepare_data import encode, load_codec


@torch.inference_mode()
def main():
    model, _ = build_model(300)
    ensure_drama3_caches(model, Drama3DecodeGraphs.graph_capacity(model))
    codec = load_codec()
    audio, sr = sf.read("ref/tikir_ref.wav", dtype="float32")
    ref_codes = encode(codec, audio, sr).long()
    del codec
    ref_text = open("ref/tikir_ref.txt", encoding="utf-8").read().strip()

    def last_hidden(prefix, text):
        prompt = build_drama3_prompt(text, model.tokenizer, model.config, [ref_text], [ref_codes]).cuda()
        return read_prompt(model, prefix, prompt, torch.arange(len(prompt), device="cuda"), len(prompt)).float()

    plain = PrefixCache(model, ref_text, ref_codes)
    print("prefix rows", plain.length)
    plain.length = 0
    h0 = last_hidden(plain, "Ciddi misin? Bunu gerçekten benden mi bekliyordun?")
    prefix = PrefixCache(model, ref_text, ref_codes)
    h1 = last_hidden(prefix, "Ciddi misin? Bunu gerçekten benden mi bekliyordun?")
    last_hidden(prefix, "Dün akşam İstanbul'da hava on iki dereceydi.")
    h2 = last_hidden(prefix, "Ciddi misin? Bunu gerçekten benden mi bekliyordun?")
    for name, h in (("cold", h1), ("warm", h2)):
        print(name, "cosine", torch.cosine_similarity(h0, h).item(), "relative diff", ((h - h0).norm() / h0.norm()).item())
    engine.chunks.__defaults__ = (100,)
    plain.rows = None
    h3 = last_hidden(plain, "Ciddi misin? Bunu gerçekten benden mi bekliyordun?")
    print("chunk100", torch.cosine_similarity(h0, h3).item(), ((h3 - h0).norm() / h0.norm()).item())
    assert torch.equal(h1, h2)


if __name__ == "__main__":
    main()
