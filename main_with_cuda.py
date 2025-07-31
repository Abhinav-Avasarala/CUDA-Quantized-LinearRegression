import torch
from qc_ext import quantize

# ─── Settings & reproducibility ──────────────────────────────────────────────────
torch.manual_seed(0)
device        = "cuda" if torch.cuda.is_available() else "cpu"
max_iter      = 20
lr            = 1e-5
num_bits      = 8
eps           = 1e-8

# ─── Problem setup ───────────────────────────────────────────────────────────────
W  = torch.randn(5000, 10000, device=device)
b  = torch.randn(5000,    device=device)
x0 = torch.randn(10000,   device=device)

# ─── Fake-grad variants using CUDA quantize ───────────────────────────────────────
def fake_grad_resid_reuse(W, b, Wxx, bits):
    resid = Wxx - b
    scale = (resid.max() - resid.min()) / (2**bits - 1 + eps)
    qres = quantize(resid, float(scale), num_bits)
    return 2 * W.t().mv(qres)


def fake_grad_resid_naive(W, b, xx, bits):
    Ax    = W @ xx
    resid = Ax - b
    scale = (resid.max() - resid.min()) / (2**bits - 1 + eps)
    qres = quantize(resid, float(scale), num_bits)
    return 2 * W.t().mv(qres)


def fake_grad_input(W, b, xx, bits):
    scale = (xx.max() - xx.min()) / (2**bits - 1 + eps)
    qx = quantize(xx, float(scale), num_bits)
    return 2 * W.t().mv(W @ qx - b)

# ─── Initialize all trajectories ────────────────────────────────────────────────
x_oracle        = x0.clone()
x_accum         = x0.clone()
x_resid_rr      = x0.clone()
x_resid_rn      = x0.clone()
x_input         = x0.clone()

W_x_accum       = W @ x_accum
W_x_resid_rr    = W @ x_resid_rr

# ─── Run & diagnostics ────────────────────────────────────────────────────────────
hdr = (
    f"{'t':>2s}  {'err_acc':>8s}  {'err_rr':>8s}  {'err_rn':>8s}  {'err_in':>8s}   "
    f"{'res_t':>6s}  {'res_a':>6s}  {'res_rr':>6s}  {'res_rn':>6s}  {'res_in':>6s}"
)
print(hdr)
print("-"*len(hdr))

for t in range(1, max_iter+1):
    # — Oracle —
    grad_true       = 2 * W.t().mv(W @ x_oracle - b)
    x_oracle_next   = x_oracle - lr * grad_true
    Ax_true         = W @ x_oracle_next

    # — 1) Quant-accumulation —
    delta           = x_oracle_next - x_accum
    scale_d         = (delta.max() - delta.min()) / (2**num_bits - 1 + eps)
    qdelta          = quantize(delta, float(scale_d), num_bits)
    x_accum_next    = x_accum + qdelta
    W_x_accum_next  = W_x_accum + W @ qdelta

    # — 2) Quant-residual-reuse —
    grad_rr         = fake_grad_resid_reuse(W, b, W_x_resid_rr, num_bits)
    x_resid_rr_next = x_resid_rr - lr * grad_rr
    W_x_rr_next     = W_x_resid_rr - lr * (W @ grad_rr)

    # — 3) Quant-residual-naive —
    grad_rn         = fake_grad_resid_naive(W, b, x_resid_rn, num_bits)
    x_resid_rn_next = x_resid_rn - lr * grad_rn
    W_x_rn_next     = W @ x_resid_rn_next

    # — 4) Quant-input —
    grad_in         = fake_grad_input(W, b, x_input, num_bits)
    x_input_next    = x_input - lr * grad_in
    W_x_in_next     = W @ x_input_next

    # — Diagnostics —
    err_acc   = torch.norm(Ax_true    - W_x_accum_next).item()
    err_rr    = torch.norm(Ax_true    - W_x_rr_next).item()
    err_rn    = torch.norm(Ax_true    - W_x_rn_next).item()
    err_in    = torch.norm(Ax_true    - W_x_in_next).item()

    res_t     = torch.norm(Ax_true          - b).item()
    res_a     = torch.norm(W_x_accum_next   - b).item()
    res_rr    = torch.norm(W_x_rr_next      - b).item()
    res_rn    = torch.norm(W_x_rn_next      - b).item()
    res_in    = torch.norm(W_x_in_next      - b).item()

    print(f"{t:2d}  "
          f"{err_acc:12.4e}{err_rr:12.4e}{err_rn:12.4e}{err_in:12.4e}   "
          f"{res_t:9.2e}{res_a:9.2e}{res_rr:9.2e}{res_rn:9.2e}{res_in:9.2e}")

    # Update for next iteration
    x_oracle        = x_oracle_next
    x_accum         = x_accum_next
    x_resid_rr      = x_resid_rr_next
    x_resid_rn      = x_resid_rn_next
    x_input         = x_input_next
    W_x_accum       = W_x_accum_next
    W_x_resid_rr    = W_x_rr_next