"""Prototipo: stato con immagine, readout dei logit delle lettere su Qwen3.5 multimodale.

    .venv/bin/python scripts/demo_immagini.py Qwen/Qwen3.5-2B-Base [examples/suite_immagini.jsonl]

Carica il modello completo (con la torre visiva), mette l'immagine nello stato con il
segnaposto <|vision_start|><|image_pad|><|vision_end|> (il processore lo espande nei
token visivi) e legge le lettere nell'ultima posizione, con 2 permutazioni.
La suite (una riga JSON per domanda: image, question, options, expected) è in
examples/suite_immagini.jsonl; le immagini e le loro licenze in examples/immagini/.
Richiede l'extra `vision`.
"""
import json, sys, time, string, numpy as np, torch, transformers
from PIL import Image
model_id = sys.argv[1]
proc = transformers.AutoProcessor.from_pretrained(model_id)
model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
print("memoria GPU", round(torch.cuda.memory_allocated() / 1e9, 2), "GB")
tok = proc.tokenizer
letters = [tok.encode(L, add_special_tokens=False)[0] for L in string.ascii_uppercase]
W = model.get_output_embeddings().weight[letters].detach().float()
SYSTEM = ("You are a decision engine. Read the state, apply the question to it, and choose exactly one of the listed "
          "options. Answer with only the uppercase letter of the chosen option, with no explanation or reasoning.")
def ask(image_path, question, options, expected):
    image = Image.open(image_path).convert("RGB")
    probs = np.zeros(len(options))
    for shift in range(2):  # 2 permutazioni (rotazione delle opzioni)
        order = [(i + shift) % len(options) for i in range(len(options))]
        opts = "\n".join(f"{string.ascii_uppercase[s]}. {options[i]}" for s, i in enumerate(order))
        IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"<state>\n{IMAGE}\n</state>\n\nQuestion: {question}\n\nOptions:\n{opts}"},
        ]
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        inputs = proc(text=[prompt], images=[image], return_tensors="pt").to("cuda")
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.model(**inputs, use_cache=False)
            h = out.last_hidden_state[0, -1].float()
        logits = (h @ W.T)[: len(options)].cpu().numpy()
        lp = logits - logits.max(); lp = lp - np.log(np.exp(lp).sum())
        back = np.empty(len(options)); back[order] = lp; probs += back / 2
        ms = (time.perf_counter() - t0) * 1000
    p = np.exp(probs); p /= p.sum()
    best = options[int(p.argmax())]
    ok = best == expected
    print(f"{'✓' if ok else '✗'} {image_path.split('/')[-1]:18s} {question[:50]:50s} -> {best[:30]:30s} p={p.max():.2f}"
          f"  p(atteso)={p[options.index(expected)]:.2f}  ({inputs['input_ids'].shape[1]} token, {ms:.0f} ms)")
    return ok
suite = sys.argv[2] if len(sys.argv) > 2 else "examples/suite_immagini.jsonl"
cases = [json.loads(line) for line in open(suite) if line.strip()]
results = [ask(c["image"], c["question"], c["options"], c["expected"]) for c in cases]
print(f"\n{model_id}: {sum(results)}/{len(results)} corrette")
