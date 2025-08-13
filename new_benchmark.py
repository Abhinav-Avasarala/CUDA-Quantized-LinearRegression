
import time
import torch
import qc_ext  # compiled extension

device = "cuda" if torch.cuda.is_available() else "cpu"
N = 10_000_000
x = torch.randn(N, device=device, dtype=torch.float32)
bit_widths = [2, 4, 8, 16, 32]
eps = 1e-8

# --- Python (pure PyTorch) quantizers ---
def quantize_py(v, scale):
    # simple round (no clamp)
    return torch.round(v / scale) * scale

def quantize_py_clamped(v, scale, bits):
    # matches integer clamp in extension
    max_q = (1 << (bits - 1)) - 1
    min_q = -max_q - 1
    q = torch.round(v / scale).clamp(min_q, max_q)
    return q * scale

def _sync():
    if device == "cuda":
        torch.cuda.synchronize()

print(f"device={device}, N={N}")

print("\n=== Python quantization (baseline) ===")
for bits in bit_widths:
    # compute scale outside timing for fairness
    scale = (x.max() - x.min()) / (2 ** bits - 1 + eps)

    _sync(); t0 = time.time()
    out_nc = quantize_py(x, scale)            # no clamp
    _sync(); dt_nc = time.time() - t0
    print(f"{bits}-bit python (no clamp): {dt_nc:.6f}s")

    _sync(); t0 = time.time()
    out_c = quantize_py_clamped(x, scale, bits)  # clamped
    _sync(); dt_c = time.time() - t0
    print(f"{bits}-bit python (clamped):  {dt_c:.6f}s")

print("\n=== CUDA extension: float-based quantization ===")
for bits in bit_widths:
    scale = (x.max() - x.min()) / (2 ** bits - 1 + eps)
    _sync(); t0 = time.time()
    out = qc_ext.quantize(x, float(scale), int(bits))
    _sync(); dt = time.time() - t0
    print(f"{bits}-bit ext float-sim: {dt:.6f}s")

print("\n=== CUDA extension: packed quantization ===")
for bits in bit_widths:
    scale = (x.max() - x.min()) / (2 ** bits - 1 + eps)
    _sync(); t0 = time.time()
    packed = qc_ext.quantize_packed(x, float(scale), int(bits))
    _sync(); dt = time.time() - t0
    print(f"{bits}-bit packed:         {dt:.6f}s")

    # correctness check against the *clamped* python baseline
    unpacked = qc_ext.unpack_packed(packed, float(scale), int(bits), N)
    max_err_nc = (unpacked - quantize_py(x, scale)).abs().max().item()
    max_err_c  = (unpacked - quantize_py_clamped(x, scale, bits)).abs().max().item()
    print(f"   max abs err vs python (no clamp): {max_err_nc:.6e}")
    print(f"   max abs err vs python (clamped):  {max_err_c:.6e}")
