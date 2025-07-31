#include <torch/extension.h>

// Forward-declare the CUDA launcher
void launch_quantize_kernel(
    const float* in,
    float*       out,
    float        scale,
    int          bits,
    size_t       N
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
// Bind to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize",
        &quantize_cuda,
        "Quantize tensor (CUDA)",
        py::arg("input"),
        py::arg("scale"),
        py::arg("bits"));
}