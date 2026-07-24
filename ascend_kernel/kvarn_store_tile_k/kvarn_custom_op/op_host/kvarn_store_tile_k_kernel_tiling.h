#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(KvarnStoreTileKKernelTilingData)
  TILING_DATA_FIELD_DEF(int32_t, bits);
  TILING_DATA_FIELD_DEF(int32_t, sinkhorn_iters);
  TILING_DATA_FIELD_DEF(int32_t, tile_count);
  TILING_DATA_FIELD_DEF(int32_t, packed_group);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvarnStoreTileKKernel, KvarnStoreTileKKernelTilingData)
}
