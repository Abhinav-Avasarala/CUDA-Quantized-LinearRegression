import time
import torch
from qc_ext import quantize as quant_cuda

# Original Python quantize
def old_quant(v, scale):
    return torch.round(v/scale) * scale

# Benchmark helper
def benchmark(fn, *args, runs=100):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        _ = fn(*args)
    torch.cuda.synchronize()
    print(f"{fn.__name__}: {time.time() - start:.4f}s over {runs} runs")

if __name__ == "__main__":
    device = "cuda"
    x = torch.randn(5000, 10000, device=device)
    scale = (x.max() - x.min()) / (2**16 - 1 + 1e-8)
    benchmark(old_quant, x, scale)
    benchmark(quant_cuda, x, float(scale))