#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>  // for printf
#include <cmath>


// CUDA kernel: quantize each element in parallel
__global__ void quantize_kernel(
    const float* __restrict__ in,
    float*       __restrict__ out,
    float                 scale,
    int                   bits,
    size_t                N
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < N) {
        // Normalize
        float x = in[idx] / scale;
        // Compute symmetric integer range (e.g. for 4 bits: range -8…+7).
        int max_q = (1 << (bits - 1)) - 1; 
        int min_q = -max_q - 1;
        // Round to nearest integer
        int q = lroundf(x);
        // Clamp
        if      (q > max_q) q = max_q;
        else if (q < min_q) q = min_q;
        // Dequantize
        out[idx] = q * scale;
    }
}

// Host wrapper: configure grid/block and launch
void launch_quantize_kernel(
    const float* in,
    float*       out,
    float        scale,
    int          bits,
    size_t       N
) {
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    quantize_kernel<<<blocks, threads>>>(in, out, scale, bits, N);
}