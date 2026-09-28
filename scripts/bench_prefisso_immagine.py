"""Benchmark: immagine ripetuta per ogni domanda (batch di prompt completi) contro immagine + prefisso
calcolati una volta, con la cache ibrida (DeltaNet + KV) duplicata per le domande e le posizioni
M-RoPE ricavate da rope_deltas.

    .venv/bin/python scripts/bench_prefisso_immagine.py Qwen/Qwen3.5-2B-Base 448 [immagine]
"""
import sys, time, statistics, numpy as np, torch, transformers
from PIL import Image
mid = sys.argv[1]; side = int(sys.argv[2]) if len(sys.argv) > 2 else 224
proc = transformers.AutoProcessor.from_pretrained(mid); tok = proc.tokenizer
model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(mid, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
body = model.model
W = model.get_output_embeddings().weight[[tok.encode(L, add_special_tokens=False)[0] for L in "ABCD"]].detach().float()
SYSTEM = "You are a decision engine. Read the state, apply the question to it, and choose exactly one of the listed options. Answer with only the uppercase letter of the chosen option, with no explanation or reasoning."
IMG = "<|vision_start|><|image_pad|><|vision_end|>"
qs = [("C'è un incendio in corso?", ["sì", "no"]),
      ("Quale azione è più appropriata?", ["nessuna azione", "monitorare la situazione", "chiamare subito i vigili del fuoco"]),
      ("Quanto è grave la situazione? (scala crescente)", ["nessun pericolo", "bassa", "media", "alta"])]
def full_text(q, o):
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content":
        f"<state>\n{IMG}\n</state>\n\nQuestion: {q}\n\nOptions:\n" + "\n".join(f"{'ABCD'[i]}. {t}" for i, t in enumerate(o))}],
        add_generation_prompt=True, enable_thinking=False, tokenize=False)
texts = [full_text(q, o) for q, o in qs]
cut = texts[0].index("Question:")
prefix_text = texts[0][:cut]
assert all(t[:cut] == prefix_text for t in texts)
suffix_ids = [tok.encode(t[cut:], add_special_tokens=False) for t in texts]
# il confine non deve cambiare la tokenizzazione (verificato sul testo senza immagine)
for t in texts:
    plain = t.replace(IMG, "X")
    c = plain.index("Question:")
    assert tok.encode(plain, add_special_tokens=False) == tok.encode(plain[:c], add_special_tokens=False) + tok.encode(plain[c:], add_special_tokens=False)
img = Image.open(sys.argv[3] if len(sys.argv) > 3 else "examples/immagini/casa_incendio.jpg").convert("RGB"); img.thumbnail((side, side))

def batched_full():
    tok.padding_side = "left"
    inputs = proc(text=texts, images=[img] * 3, return_tensors="pt", padding=True).to("cuda")
    h = body(**inputs, use_cache=False).last_hidden_state[:, -1].float()
    return (h @ W.T).cpu().numpy()

def single(i):
    inputs = proc(text=[texts[i]], images=[img], return_tensors="pt").to("cuda")
    h = body(**inputs, use_cache=False).last_hidden_state[:, -1].float()
    return (h @ W.T).cpu().numpy()

def shared():
    body.rope_deltas = None
    pre = proc(text=[prefix_text], images=[img], return_tensors="pt").to("cuda")
    out = body(**pre, use_cache=True)
    cache = out.past_key_values
    P = pre["input_ids"].shape[1]
    cache.reorder_cache(torch.zeros(3, dtype=torch.long, device="cuda"))  # 3 copie dello stesso prefisso
    S = max(map(len, suffix_ids))
    ids = torch.full((3, S), tok.pad_token_id, dtype=torch.long, device="cuda")
    mask = torch.zeros((3, P + S), dtype=torch.long, device="cuda"); mask[:, :P] = 1
    for r, s in enumerate(suffix_ids):
        ids[r, :len(s)] = torch.tensor(s, device="cuda"); mask[r, P:P + len(s)] = 1
    delta = body.rope_deltas.reshape(-1)[0]
    pos = (torch.arange(P, P + S, device="cuda") + delta).view(1, 1, -1).expand(3, 3, -1)
    h = body(input_ids=ids, attention_mask=mask, past_key_values=cache, position_ids=pos, use_cache=True).last_hidden_state
    last = torch.tensor([len(s) - 1 for s in suffix_ids], device="cuda")
    h = h[torch.arange(3, device="cuda"), last].float()
    return (h @ W.T).cpu().numpy()

def timed(fn, n=15):
    for _ in range(3): fn()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.perf_counter(); r = fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000, r

def answers(logits):
    return [o[int(np.argmax(logits[i, :len(o)]))] for i, (_, o) in enumerate(qs)]

with torch.inference_mode():
    t_full, l_full = timed(batched_full)
    t_one, _ = timed(lambda: single(0))
    t_shared, l_shared = timed(shared)
    vis_in = proc(text=[prefix_text], images=[img], return_tensors="pt").to("cuda")
    t_vis, _ = timed(lambda: body.visual(vis_in["pixel_values"], grid_thw=vis_in["image_grid_thw"]))
def probs(l):
    return [np.round(np.exp(l[i, :len(o)] - l[i, :len(o)].max()) / np.exp(l[i, :len(o)] - l[i, :len(o)].max()).sum(), 3).tolist() for i, (_, o) in enumerate(qs)]
print(f"{mid.split('/')[-1]}, lato {side}px, 3 domande (preparazione immagine inclusa):")
print(f"  batch di 3 prompt completi (immagine x3): {t_full:6.1f} ms  {answers(l_full)}")
print(f"  1 sola domanda:                           {t_one:6.1f} ms")
print(f"  immagine + prefisso UNA volta, 3 code:    {t_shared:6.1f} ms  {answers(l_shared)}")
print(f"  (torre visiva da sola, 1 immagine:        {t_vis:6.1f} ms)")
print(f"  probabilità completo: {probs(l_full)}")
print(f"  probabilità condiviso: {probs(l_shared)}")
