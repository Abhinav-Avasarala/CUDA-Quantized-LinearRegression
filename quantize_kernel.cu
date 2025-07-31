#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>  // for printf

// CUDA kernel: quantize each element in parallel
__global__ void quantize_kernel(
    const float* __restrict__ in,
    float*       __restrict__ out,
    float        scale,
    size_t       N
) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float v = in[idx];
    out[idx] = roundf(v/scale) * scale;
  }
}

// Host wrapper: configure grid/block and launch
void launch_quantize_kernel(
    const float* in,
    float*       out,
    float        scale,
    size_t       N
) {
  const int threads = 256;
  const int blocks  = (N + threads - 1) / threads;
  quantize_kernel<<<blocks, threads>>>(in, out, scale, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("quantize_kernel launch failed: %s\n", cudaGetErrorString(err));
  }
}