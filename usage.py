import torch
from qc_ext import quantize

# ─── Settings & reproducibility ──────────────────────────────────────────────────
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
max_iter  = 20
lr         = 1e-5
num_bits   = 16
eps        = 1e-8

# ─── Problem setup ───────────────────────────────────────────────────────────────
W  = torch.randn(5000, 10000, device=device)
b  = torch.randn(5000,    device=device)
x0 = torch.randn(10000,   device=device)

# ─── Custom fake-grad with our CUDA quantize ↓─────────────────────────────────────
def fake_grad_resid_reuse(W, b, Wxx, bits):
    resid = Wxx - b
    scale = (resid.max() - resid.min()) / (2**bits - 1 + eps)
    qres  = quantize(resid, float(scale))     # <-- calls our CUDA kernel
    return 2 * W.t().mv(qres)

# ─── Quick smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    x = torch.randn(100, device=device)
    qx = quantize(x, 0.1)
    print("Input slice:", x[:5])
    print("Quantized:",   qx[:5])