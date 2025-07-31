from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
  name="qc_ext",
  ext_modules=[
    CUDAExtension(
      name="qc_ext",
      sources=["quantize.cpp", "quantize_kernel.cu"],
    )
  ],
  cmdclass={"build_ext": BuildExtension}
)