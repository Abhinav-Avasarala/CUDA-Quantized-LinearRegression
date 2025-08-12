#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>  // for printf
#include <cmath>
#include <stdint.h>
#include <math.h>


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

// ----------- NEW CODE FOR PACKED QUANTIZATION -----------

// ==================== Packed Quantization Kernel ====================
__global__ void quantize_packed_kernel(
    const float* __restrict__ in,
    uint8_t* __restrict__ out,
    float scale,
    int bits,
    size_t N
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int max_q = (1 << (bits - 1)) - 1;
    int min_q = -max_q - 1;

    // Quantize value
    float x = in[idx] / scale;
    int q = lroundf(x);
    if (q > max_q) q = max_q;
    if (q < min_q) q = min_q;

    // ---------------- PACKING ----------------
    if (bits == 32) {
        // Store full float
        reinterpret_cast<float*>(out)[idx] = q * scale;
    }
    else if (bits == 16) {
        // Store as uint16_t
        reinterpret_cast<uint16_t*>(out)[idx] = static_cast<uint16_t>(q - min_q);
    }
    else if (bits == 8) {
        out[idx] = static_cast<uint8_t>(q - min_q);
    }
    else if (bits == 4) {
        size_t byte_idx = idx / 2;
        bool high_nibble = (idx % 2) != 0;
        uint8_t val4 = static_cast<uint8_t>(q - min_q) & 0xF;
        uint8_t old = out[byte_idx];
        if (high_nibble) {
            out[byte_idx] = (old & 0x0F) | (val4 << 4);
        } else {
            out[byte_idx] = (old & 0xF0) | val4;
        }
    }
    else if (bits == 2) {
        size_t byte_idx = idx / 4;
        int shift = (idx % 4) * 2;
        uint8_t val2 = static_cast<uint8_t>(q - min_q) & 0x3;
        uint8_t old = out[byte_idx];
        old &= ~(0x3 << shift);
        old |= (val2 << shift);
        out[byte_idx] = old;
    }
}

// ==================== Packed Dequantization Kernel ====================
__global__ void unpack_packed_kernel(
    const uint8_t* __restrict__ in,
    float* __restrict__ out,
    float scale,
    int bits,
    size_t N
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int max_q = (1 << (bits - 1)) - 1;
    int min_q = -max_q - 1;

    if (bits == 32) {
        out[idx] = reinterpret_cast<const float*>(in)[idx];
    }
    else if (bits == 16) {
        uint16_t val16 = reinterpret_cast<const uint16_t*>(in)[idx];
        int q = static_cast<int>(val16) + min_q;
        out[idx] = q * scale;
    }
    else if (bits == 8) {
        uint8_t val8 = in[idx];
        int q = static_cast<int>(val8) + min_q;
        out[idx] = q * scale;
    }
    else if (bits == 4) {
        size_t byte_idx = idx / 2;
        bool high_nibble = (idx % 2) != 0;
        uint8_t packed = in[byte_idx];
        uint8_t val4 = high_nibble ? ((packed >> 4) & 0xF) : (packed & 0xF);
        int q = static_cast<int>(val4) + min_q;
        out[idx] = q * scale;
    }
    else if (bits == 2) {
        size_t byte_idx = idx / 4;
        int shift = (idx % 4) * 2;
        uint8_t packed = in[byte_idx];
        uint8_t val2 = (packed >> shift) & 0x3;
        int q = static_cast<int>(val2) + min_q;
        out[idx] = q * scale;
    }
}

// ==================== Kernel Launchers ====================
void launch_quantize_packed_kernel(
    const float* in,
    uint8_t* out,
    float scale,
    int bits,
    size_t N
) {
    size_t packed_size = N;
    if (bits == 4) packed_size = (N + 1) / 2;
    else if (bits == 2) packed_size = (N + 3) / 4;
    else if (bits == 16) packed_size = N * 2;
    else if (bits == 32) packed_size = N * 4;

    cudaMemset(out, 0, packed_size);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    quantize_packed_kernel<<<blocks, threads>>>(in, out, scale, bits, N);
}

void launch_unpack_packed_kernel(
    const uint8_t* in,
    float* out,
    float scale,
    int bits,
    size_t N
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    unpack_packed_kernel<<<blocks, threads>>>(in, out, scale, bits, N);
}