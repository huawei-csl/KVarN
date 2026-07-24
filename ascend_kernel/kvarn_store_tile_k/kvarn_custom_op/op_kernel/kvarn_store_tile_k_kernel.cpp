#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "tensorutils.h"
#include "kvarn_store_tile_k_kernel_vec.h"


extern "C" __global__ __aicore__ void kvarn_store_tile_k_kernel(GM_ADDR k_tile_rotated, GM_ADDR q_packed, GM_ADDR s_col_k, GM_ADDR zp_k, GM_ADDR s_row_k, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    PipeBarrier<PIPE_ALL>();
    int bits = tiling_data.bits;
    int sinkhorn_iters = tiling_data.sinkhorn_iters;
    int tile_count = tiling_data.tile_count;
    int packed_group = tiling_data.packed_group;
    if ASCEND_IS_AIV{
        kvarn_store_tile_k_kernel_vec(k_tile_rotated, q_packed, s_col_k, zp_k, s_row_k, workspace, bits, sinkhorn_iters, tile_count, packed_group);
    }

}
