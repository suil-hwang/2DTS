#include <torch/extension.h>
#include "src/extension_interface.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("rasterize_hybrid", &rasterizeHybridForward);
}
