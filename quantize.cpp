#include <torch/extension.h>
#include <vector>


// Forward-declare the CUDA launcher
void launch_quantize_kernel(
    const float* in,
    float*       out,
    float        scale,
    int          bits,
    size_t       N
);

void launch_quantize_packed_kernel(
    const float* in,
    uint8_t* out,
    float scale,
    int bits,
    size_t N
);

void launch_unpack_packed_kernel(
    const uint8_t* in,
    float* out,
    float scale,
    int bits,
    size_t N
);


// C++ binding: takes a PyTorch tensor, calls our CUDA code
at::Tensor quantize_cuda(const at::Tensor& in, double scale, int bits) {
  auto in_contig = in.contiguous();
  auto out       = torch::empty_like(in_contig);
  const size_t N = in_contig.numel();
  const float scale_f = static_cast<float>(scale);

  launch_quantize_kernel(
    in_contig.data_ptr<float>(),
    out.data_ptr<float>(),
    scale_f,
    bits,            // pass bits here
    N
  );

  return out;
}

at::Tensor quantize_packed_cuda(const at::Tensor& in, double scale, int bits) {
    auto in_contig = in.contiguous();
    const size_t N = in_contig.numel();
    const float scale_f = static_cast<float>(scale);

    size_t packed_size = N;
    if (bits == 4) packed_size = (N + 1) / 2;
    else if (bits == 2) packed_size = (N + 3) / 4;
    else if (bits == 16) packed_size = N * 2;
    else if (bits == 32) packed_size = N * 4;

    auto out = torch::empty({(long)packed_size}, torch::dtype(torch::kUInt8).device(in.device()));

    launch_quantize_packed_kernel(
        in_contig.data_ptr<float>(),
        out.data_ptr<uint8_t>(),
        scale_f,
        bits,
        N
    );

    return out;
}

at::Tensor unpack_packed_cuda(const at::Tensor& in, double scale, int bits, size_t N) {
    auto out = torch::empty({(long)N}, torch::dtype(torch::kFloat32).device(in.device()));

    launch_unpack_packed_kernel(
        in.data_ptr<uint8_t>(),
        out.data_ptr<float>(),
        static_cast<float>(scale),
        bits,
        N
    );

    return out;
}

// Bind to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize", &quantize_cuda, "Quantize tensor (CUDA)");
    m.def("quantize_packed", &quantize_packed_cuda, "Quantize tensor into packed format (CUDA)");
    m.def("unpack_packed", &unpack_packed_cuda, "Unpack packed tensor back to float (CUDA)",
          py::arg("input"), py::arg("scale"), py::arg("bits"), py::arg("N"));
}