import torch

# ─── Settings & reproducibility ──────────────────────────────────────────────────
torch.manual_seed(0)
device        = "cuda" if torch.cuda.is_available() else "cpu"
max_iter      = 40
# lr            = 1e-7
num_bits      = 3
eps           = 1e-8


#====> random times scaling factor to make it larger 

# ─── Problem setup ───────────────────────────────────────────────────────────────
W  = torch.randn(5000, 10000, device=device) * 10
b  = torch.randn(5000,    device=device) * 10
x0 = torch.randn(10000,   device=device)

def power_iteration(W, num_iters=10):
    """Estimate σ_max(W) via a few power‐iteration steps on W^T W."""
    v = torch.randn(W.shape[1], device=W.device)
    for _ in range(num_iters):
        v = W.t().mv(W.mv(v))
        v = v / (v.norm() + 1e-12)
    # after convergence, ‖W v‖ ≈ σ_max(W)
    return W.mv(v).norm()

# ── right after you set up W, b ──
sigma_max = power_iteration(W, num_iters=20)
L_lip     = 2 * sigma_max**2
lr        = 0.9 / L_lip   # 90% of the maximum stable step‐size
print(f"Estimated σ_max(W)={sigma_max:.3e}, setting lr={lr:.3e}")

# ─── Quantizer ───────────────────────────────────────────────────────────────────
def quantize(v, scale):
    return torch.round(v/scale) * scale

# ─── Fake-grad variants ───────────────────────────────────────────────────────────
def fake_grad_resid_reuse(W, b, Wxx, bits):
    # quantize (W x̂ – b) then ∇f = 2 Wᵀ q(resid)
    resid = Wxx - b
    scale = (resid.max() - resid.min())/(2**bits - 1 + eps)
    return 2*W.t().mv(quantize(resid, scale))

def fake_grad_resid_naive(W, b, xx, bits):
    # recompute W@x̂, quantize that residual every step
    Ax    = W @ xx
    resid = Ax - b
    scale = (resid.max() - resid.min())/(2**bits - 1 + eps)
    return 2*W.t().mv(quantize(resid, scale))

def fake_grad_input(W, b, xx, bits):
    # quantize x then ∇f = 2 Wᵀ (W q(x) – b)
    scale = (xx.max() - xx.min())/(2**bits - 1 + eps)
    qx    = quantize(xx, scale)
    return 2*W.t().mv(W @ qx - b)

# ─── Initialize all trajectories ────────────────────────────────────────────────
x_oracle        = x0.clone()
x_accum         = x0.clone()
x_resid_rr      = x0.clone()
x_resid_rn      = x0.clone()
x_input         = x0.clone()


W_x_accum       = W @ x_accum
W_x_resid_rr    = W @ x_resid_rr
# W_x_resid_rn is recomputed each step
# W_x_input is recomputed each step

# ─── Run & diagnostics ────────────────────────────────────────────────────────────
hdr = (
    f"{'t':>2s}  {'err_true':>9s}  {'err_acc':>9s}  {'err_rr':>9s}  {'err_rn':>9s}  {'err_in':>9s}   "
    f"{'res_t':>7s}  {'res_a':>7s}  {'res_rr':>7s}  {'res_rn':>7s}  {'res_in':>7s}"
)
print(hdr)
print("-"*len(hdr))


with torch.no_grad():
    x_star   = torch.linalg.lstsq(W, b).solution  # more stable than explicit inverse
    Ax_final  = W @ x_star

for t in range(1, max_iter+1):
    # — Oracle —
    grad_true       = 2*W.t().mv(W @ x_oracle - b)
    x_oracle_next   = x_oracle - lr*grad_true
    Ax_true         = W @ x_oracle_next

    # — 1) Quant-accumulation —
    delta           = x_oracle_next - x_accum
    scale_d         = (delta.max()-delta.min())/(2**num_bits-1+eps)
    qdelta          = quantize(delta, scale_d)
    x_accum_next    = x_accum + qdelta
    W_x_accum_next  = W_x_accum + W @ qdelta

    # — 2) Quant-residual-reuse —
    grad_rr         = fake_grad_resid_reuse(W, b, W_x_resid_rr, num_bits)
    x_resid_rr_next = x_resid_rr - lr*grad_rr
    W_x_rr_next     = W_x_resid_rr - lr*(W @ grad_rr)

    # — 3) Quant-residual-naive —
    grad_rn         = fake_grad_resid_naive(W, b, x_resid_rn, num_bits)
    x_resid_rn_next = x_resid_rn - lr*grad_rn
    W_x_rn_next     = W @ x_resid_rn_next

    # — 4) Quant-input —
    grad_in         = fake_grad_input(W, b, x_input, num_bits)
    x_input_next    = x_input - lr*grad_in
    W_x_in_next     = W @ x_input_next

    # — Diagnostics —
    err_true = torch.norm(Ax_final - Ax_true).item()
    err_acc   = torch.norm(Ax_final - W_x_accum_next).item()
    err_rr    = torch.norm(Ax_final - W_x_rr_next).item()
    err_rn    = torch.norm(Ax_final - W_x_rn_next).item()
    err_in    = torch.norm(Ax_final - W_x_in_next).item()

    res_t     = torch.norm(Ax_true - b).item()
    res_a     = torch.norm(W_x_accum_next - b).item()
    res_rr    = torch.norm(W_x_rr_next - b).item()
    res_rn    = torch.norm(W_x_rn_next - b).item()
    res_in    = torch.norm(W_x_in_next - b).item()

    print(f"{t:2d}  "
      f"{err_true:11.4e}{err_acc:11.4e}{err_rr:11.4e}{err_rn:11.4e}{err_in:11.4e}   "
      f"{res_t:9.2e}{res_a:9.2e}{res_rr:9.2e}{res_rn:9.2e}{res_in:9.2e}")

    # Update for next iteration
    x_oracle     = x_oracle_next
    x_accum      = x_accum_next
    x_resid_rr   = x_resid_rr_next
    x_resid_rn   = x_resid_rn_next
    x_input      = x_input_next
    W_x_accum    = W_x_accum_next
    W_x_resid_rr = W_x_rr_next
