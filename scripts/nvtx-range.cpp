// Header-only NVTX v3 dispatch lets Nsight timestamp the Python call boundary.
#include <nvtx3/nvToolsExt.h>
extern "C" int fnn_range_push(const char *name) { return nvtxRangePushA(name); }
extern "C" int fnn_range_pop() { return nvtxRangePop(); }
