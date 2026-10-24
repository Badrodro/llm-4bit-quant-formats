"""Scalar 4-bit (and 3-bit) weight formats, simulated in PyTorch (quantize then dequantize).

Every function takes a flat float32 tensor whose length is a multiple of the block size and returns
(dequantized tensor, bits per weight). Bits per weight include the per-block scales.

  - INT4 g128   : symmetric INT4, FP16 scale per block of 128            -> 4.125 bpw
  - MXFP4       : OCP MX, E2M1 values, power-of-two (E8M0) scale per 32 -> 4.25 bpw
  - NVFP4       : E2M1 values, FP8 E4M3 scale per 16 + FP32 per tensor  -> 4.5 bpw
                  (optional search over 8 scale multipliers per block)
  - Lloyd-Max   : MSE-optimal levels for a Gaussian source, E4M3 scale per 32 searched per block
"""
import numpy as np
import torch
from scipy.stats import norm

# E2M1 (FP4) values; -0 and +0 collapse into one level.
FP4_LEVELS = torch.unique(torch.tensor([-6, -4, -3, -2, -1.5, -1, -0.5, -0.0, 0.0, 0.5, 1, 1.5, 2, 3, 4, 6],
                                       dtype=torch.float32))


def round_to_levels(y, levels):
    """Round to the nearest value of a sorted set of levels (saturating at both ends)."""
    levels = levels.to(y.device)
    midpoints = (levels[1:] + levels[:-1]) / 2
    return levels[torch.bucketize(y, midpoints)]


def e4m3(s):
    """Round positive scales to FP8 E4M3 (max 448)."""
    return s.clamp(max=448.0).to(torch.float8_e4m3fn).to(torch.float32)


def lloyd_max(bits, iters=300):
    """Lloyd-Max quantizer levels for a unit Gaussian."""
    n = 1 << bits
    lv = np.linspace(-2.5, 2.5, n)
    for _ in range(iters):
        t = np.concatenate([[-np.inf], (lv[1:] + lv[:-1]) / 2, [np.inf]])
        a, b = t[:-1], t[1:]
        lv = (norm.pdf(a) - norm.pdf(b)) / (norm.cdf(b) - norm.cdf(a))
    return torch.tensor(lv, dtype=torch.float32)


LM3, LM4 = lloyd_max(3), lloyd_max(4)


def blocks(x, g):
    return x.reshape(-1, g)


def q_int4_g128(x):
    xb = blocks(x, 128)
    s = (xb.abs().amax(1, keepdim=True) / 7).half().float().clamp(min=1e-12)
    return (torch.clamp(torch.round(xb / s), -8, 7) * s).reshape(x.shape), 4 + 16 / 128


def q_mxfp4(x):
    xb = blocks(x, 32)
    amax = xb.abs().amax(1, keepdim=True).clamp(min=1e-30)
    s = torch.exp2(torch.floor(torch.log2(amax)) - 2)
    return (round_to_levels(xb / s, FP4_LEVELS) * s).reshape(x.shape), 4 + 8 / 32


def q_nvfp4(x, scale_search=False):
    xb = blocks(x, 16)
    g = x.abs().max().clamp(min=1e-30) / (6 * 448)  # FP32 per-tensor scale
    amax = xb.abs().amax(1, keepdim=True)
    mults = [1.0] if not scale_search else [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]
    best, best_err = None, None
    for m in mults:
        s = (e4m3(amax / 6 * m / g) * g).clamp(min=1e-30)
        xq = round_to_levels(xb / s, FP4_LEVELS) * s
        err = ((xb - xq) ** 2).sum(1, keepdim=True)
        if best is None:
            best, best_err = xq, err
        else:
            better = err < best_err
            best = torch.where(better, xq, best)
            best_err = torch.where(better, err, best_err)
    return best.reshape(x.shape), 4 + 8 / 16


def q_lloyd_block(x, levels, bits):
    """Gaussian Lloyd-Max quantizer, E4M3 scale per block of 32 chosen to minimize the block error."""
    xb = blocks(x, 32)
    g = x.abs().max().clamp(min=1e-30) / 448
    rms = xb.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-30)
    best, best_err = None, None
    for m in np.arange(0.7, 1.35, 0.05):
        s = (e4m3(rms * float(m) / g) * g).clamp(min=1e-30)
        xq = round_to_levels(xb / s, levels) * s
        err = ((xb - xq) ** 2).sum(1, keepdim=True)
        if best is None:
            best, best_err = xq, err
        else:
            better = err < best_err
            best = torch.where(better, xq, best)
            best_err = torch.where(better, err, best_err)
    return best.reshape(x.shape), bits + 8 / 32
