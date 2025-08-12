import torch
import time
import qc_ext  # your compiled extension name

device = "cuda"

N = 10_000_000
x = torch.randn(N, device=device)
scale = 0.1

bit_widths = [2, 4, 8, 16, 32]

print("=== Float-based quantization ===")
for bits in bit_widths:
    torch.cuda.synchronize()
    t0 = time.time()
    out = qc_ext.quantize(x, scale, bits)
    torch.cuda.synchronize()
    print(f"{bits}-bit float-sim: {time.time() - t0:.6f}s")

print("\n=== Packed quantization ===")
for bits in bit_widths:
    torch.cuda.synchronize()
    t0 = time.time()
    packed = qc_ext.quantize_packed(x, scale, bits)
    torch.cuda.synchronize()
    print(f"{bits}-bit packed: {time.time() - t0:.6f}s")

    # Optional: verify correctness
    unpacked = qc_ext.unpack_packed(packed, scale, bits, N)
    max_err = (unpacked - x).abs().max().item()
    print(f"   max abs error: {max_err}")
