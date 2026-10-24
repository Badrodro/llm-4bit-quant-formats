"""Does a trellis code keep an advantage over 4-bit scalar formats at equal quality on real LLMs?

For one model: every linear layer of the transformer blocks (q, k, v, o, gate, up, down) is quantized
then dequantized (embeddings and LM head stay in BF16), and perplexity is measured on WikiText-2
(full test set) and C4 (128 validation windows), with 2048-token windows.

Formats:
  - scalar: NVFP4 (with and without scale search), MXFP4, INT4 g128, Lloyd-Max 4 and 3 bits g32 (E4M3),
    each WITHOUT and WITH a random Hadamard rotation (the best variant is kept, to be fair);
  - Gaussian-code trellis (QTIP-style bitshift trellis), always with rotation: (k=7, V=2) and (k=4, V=1),
    L=14, sequences of 1024 weights, FP16 scale per row.
Rounding: nearest (scalar) or Viterbi (trellis). No calibration data for any format.
Bits per weight count every scale and the initial trellis state.

Usage: python eval_perplexity.py <HF model> [--quick] [--extra]
  --quick  evaluates only 16 windows per corpus
  --extra  adds the trellis point k=3 (about 3 bits per weight)
"""
import gzip, json, math, os, sys, time
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from formats import q_nvfp4, q_mxfp4, q_int4_g128, q_lloyd_block, LM3, LM4

DEV = "cuda"
MODEL = sys.argv[1]
QUICK = "--quick" in sys.argv
EXTRA = "--extra" in sys.argv
SEQ = 2048
L_TCQ = 14
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", MODEL.split("/")[-1] + (".quick" if QUICK else "") + ".json")
torch.manual_seed(0)


# ------------------------------------------------------------------ random Hadamard transform
def hadamard_block(n):
    b = n & -n  # largest power of two dividing n
    return min(b, 1024)


_SIGNS = {}


def signs(n):
    if n not in _SIGNS:
        g = torch.Generator().manual_seed(1000 + n)
        _SIGNS[n] = (torch.randint(0, 2, (n,), generator=g).float() * 2 - 1).to(DEV)
    return _SIGNS[n]


def fwht(x, blk):
    shp = x.shape
    n = shp[-1]
    y = x.reshape(-1, n // blk, blk)
    h = 1
    while h < blk:
        y = y.view(-1, n // blk, blk // (2 * h), 2, h)
        y = torch.stack((y[:, :, :, 0, :] + y[:, :, :, 1, :], y[:, :, :, 0, :] - y[:, :, :, 1, :]), dim=3)
        h *= 2
    return y.reshape(shp) / math.sqrt(blk)


def rht(W):
    return fwht(W * signs(W.shape[1]), hadamard_block(W.shape[1]))


def irht(Y):
    return fwht(Y, hadamard_block(Y.shape[1])) * signs(Y.shape[1])


# ------------------------------------------------------------------ Gaussian-code trellis (fast Viterbi)
_CODES = {}


def gauss_code(L, V):
    if (L, V) not in _CODES:
        g = torch.Generator().manual_seed(1)
        _CODES[(L, V)] = torch.randn((1 << L, V), generator=g).to(DEV)
    return _CODES[(L, V)]


@torch.no_grad()
def viterbi_gauss(y, code, k, V, B=1024):
    """y: (nseq, T*V) normalized. Branch cost = |c|^2 - 2 <y, c> (|y|^2 does not change the argmin)."""
    S, P = code.shape[0], 1 << k
    R = S >> k
    c2 = (code ** 2).sum(1)
    bp_dtype = torch.uint8 if P <= 256 else torch.int16
    outputs = []
    for i0 in range(0, y.shape[0], B):
        yb = y[i0:i0 + B]
        nb = yb.shape[0]
        T = yb.shape[1] // V
        yb = yb.view(nb, T, V)

        def cost(t):
            c = torch.addcmul(c2, yb[:, t, 0:1], code[:, 0][None], value=-2.0)
            for v in range(1, V):
                c.addcmul_(yb[:, t, v:v + 1], code[:, v][None], value=-2.0)
            return c

        D = cost(0)
        bp = torch.empty((T, nb, R), dtype=bp_dtype, device=DEV)
        for t in range(1, T):
            m, a = D.view(nb, P, R).min(dim=1)
            bp[t] = a.to(bp_dtype)
            D = cost(t).view(nb, R, P).add_(m.unsqueeze(-1)).view(nb, S)
        s = D.argmin(dim=1)
        states = torch.empty((T, nb), dtype=torch.long, device=DEV)
        states[T - 1] = s
        for t in range(T - 1, 0, -1):
            r = s >> k
            j = bp[t].gather(1, r[:, None]).squeeze(1).long()
            s = j * R + r
            states[t - 1] = s
        outputs.append(code[states.T].reshape(nb, T * V))
    return torch.cat(outputs)


GAINS = {}


def tune_gain(k, V):
    """Code gain tuned on an independent Gaussian source (not on the evaluated model)."""
    if (k, V) not in GAINS:
        z = torch.randn(64, 256, generator=torch.Generator().manual_seed(99)).to(DEV)
        best = None
        for g in [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]:
            zq = viterbi_gauss(z, gauss_code(L_TCQ, V) * g, k, V)
            e = float((z - zq).pow(2).mean())
            if best is None or e < best[0]:
                best = (e, g)
        GAINS[(k, V)] = best[1]
    return GAINS[(k, V)]


T_SEQ = 1024  # weights per trellis sequence: initial-state overhead is (L-k)/1024


def q_tcq(W, k, V):
    """Rotation, FP16 scale per row, then trellis over sequences of 1024 consecutive weights."""
    Y = rht(W)
    rms = Y.pow(2).mean(1, keepdim=True).sqrt().half().float().clamp(min=1e-12)
    flat = (Y / rms).reshape(-1)
    n = flat.numel()
    pad = (-n) % T_SEQ
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    seq = flat.view(-1, T_SEQ)
    T = T_SEQ // V
    R = (1 << L_TCQ) >> k
    B = max(64, min(1024, int(6e8 // (T * R))))  # back-pointer memory <= ~600 MB
    yq = viterbi_gauss(seq, gauss_code(L_TCQ, V) * tune_gain(k, V), k, V, B=B).reshape(-1)[:n]
    Wq = irht(yq.view(W.shape) * rms)
    bits = (seq.shape[0] * (L_TCQ + k * (T - 1)) + 16 * W.shape[0])
    return Wq, bits / n


def with_rht(f):
    def g(W):
        Yq, bpw = f(rht(W).reshape(-1))
        return irht(Yq.reshape(W.shape)), bpw
    return g


def without_rht(f):
    def g(W):
        Wq, bpw = f(W.reshape(-1))
        return Wq.reshape(W.shape), bpw
    return g


SCALAR = {
    "NVFP4": lambda x: q_nvfp4(x),
    "NVFP4 + scale search": lambda x: q_nvfp4(x, True),
    "MXFP4": q_mxfp4,
    "INT4 g128": q_int4_g128,
    "Lloyd-Max 4b g32": lambda x: q_lloyd_block(x, LM4, 4),
    "Lloyd-Max 3b g32": lambda x: q_lloyd_block(x, LM3, 3),
}
CONFIGS = {}
for name, f in SCALAR.items():
    CONFIGS[name] = without_rht(f)
    CONFIGS[name + " +RHT"] = with_rht(f)
TCQ = [(7, 2), (4, 1)] + ([(3, 1)] if EXTRA else [])
for k, V in TCQ:
    CONFIGS[f"Trellis k={k} V={V}"] = (lambda k, V: (lambda W: q_tcq(W, k, V)))(k, V)


# ------------------------------------------------------------------ data and perplexity
def tokens(tok):
    p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    wiki = tok("\n\n".join(pd.read_parquet(p)["text"].tolist()), return_tensors="pt").input_ids[0]
    p = hf_hub_download("allenai/c4", "en/c4-validation.00000-of-00008.json.gz", repo_type="dataset")
    docs, n = [], 0
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line)["text"])
            n += len(docs[-1])
            if n > 128 * SEQ * 6:  # ~6 characters per token: enough margin
                break
    c4 = tok("\n\n".join(docs), return_tensors="pt").input_ids[0][: 128 * SEQ]
    return {"wikitext2": wiki, "c4": c4}


@torch.no_grad()
def perplexity(model, ids):
    n = ids.numel() // SEQ
    if QUICK:
        n = min(n, 16)
    nll = 0.0
    for i in range(n):
        x = ids[i * SEQ:(i + 1) * SEQ].unsqueeze(0).to(DEV)
        nll += float(model(x, labels=x).loss) * (SEQ - 1)
    return math.exp(nll / (n * (SEQ - 1)))


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    data = tokens(tok)
    linears = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and ".layers." in n}
    original = {n: m.weight.data.clone().cpu() for n, m in linears.items()}
    n_weights = sum(w.numel() for w in original.values())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    res.setdefault("_meta", {"model": MODEL, "quantized_weights": n_weights, "window": SEQ, "trellis_L": L_TCQ,
                             "tokens": {k: int(v.numel()) for k, v in data.items()}, "quick": QUICK})
    print(f"{MODEL}: {n_weights/1e6:.0f} M quantized weights in {len(linears)} layers", flush=True)
    if "BF16" not in res:
        res["BF16"] = {"bpw": 16.0, **{d: perplexity(model, ids) for d, ids in data.items()}}
        print("  BF16", res["BF16"], flush=True)
    for name, f in CONFIGS.items():
        if name in res:
            continue
        t1 = time.time()
        total_bits = 0.0
        for n, m in linears.items():
            W = original[n].to(DEV, torch.float32)
            Wq, bpw = f(W)
            total_bits += bpw * W.numel()
            m.weight.data.copy_(Wq.to(torch.bfloat16))
            del W, Wq
        tq = time.time() - t1
        res[name] = {"bpw": round(total_bits / n_weights, 4), **{d: perplexity(model, ids) for d, ids in data.items()},
                     "quantization_seconds": round(tq, 1)}
        if name.startswith("Trellis"):
            res[name]["code_gain"] = {f"k={k},V={V}": g for (k, V), g in GAINS.items()}
        print(f"  {name:26s} bpw={res[name]['bpw']:.3f}  wiki={res[name]['wikitext2']:.3f}  c4={res[name]['c4']:.3f}  "
              f"(quantization {tq:.0f} s)", flush=True)
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)  # saved after each format: the run can resume
    for n, m in linears.items():
        m.weight.data.copy_(original[n].to(DEV))
    print(f"Done in {time.time()-t0:.0f} s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
