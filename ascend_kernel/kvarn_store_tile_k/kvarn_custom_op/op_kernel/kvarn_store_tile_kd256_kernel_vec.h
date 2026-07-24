#pragma once
#include "tensorutils.h"
#include "compute_col_std_d256_vf.h"
#include "init_log_row_d256_vf.h"
#include "update_log_col_d256_vf.h"
#include "update_log_row_d256_vf.h"
#include "quantize_pack_4bit_d256_vf.h"
#include "init_log_col_d256_vf.h"
#include "quantize_pack_2bit_d256_vf.h"
#include "compute_row_std_d256_vf.h"
#include "update_best_d256_vf.h"
#include "cast_log_to_half_d256_vf.h"
#include "init_best_imbalance_d256_vf.h"

__aicore__ inline void kvarn_store_tile_kd256_kernel_vec(GM_ADDR k_tile_rotated_, GM_ADDR q_packed_, GM_ADDR s_col_k_, GM_ADDR zp_k_, GM_ADDR s_row_k_, GM_ADDR workspace, int bits, int sinkhorn_iters, int tile_count, int packed_group) {
    TPipe pipe;
    TPipe* pipe_ptr = GetTPipePtr();
    int _offset = 0;
    SEvent<PIPE_V, PIPE_MTE2, true> _tmp_sevent_valid_ubin_0;
    SEvent<PIPE_MTE2, PIPE_V, false> _tmp_sevent_ready_ubin_0;
    DEvent<PIPE_MTE3, PIPE_V, true> _tmp_devent_valid_ubout_1;
    DEvent<PIPE_V, PIPE_MTE3, false> _tmp_devent_ready_ubout_1;
    SEvent<PIPE_V, PIPE_MTE3, false> _tmp_sevent_ready_ubout_0;
    SEvent<PIPE_MTE3, PIPE_V, true> _tmp_sevent_valid_ubout_0;
    GlobalTensor<half> k_tile_rotated; k_tile_rotated.SetGlobalBuffer((__gm__ half*) k_tile_rotated_);
    GlobalTensor<uint8_t> q_packed; q_packed.SetGlobalBuffer((__gm__ uint8_t*) q_packed_);
    GlobalTensor<half> s_col_k; s_col_k.SetGlobalBuffer((__gm__ half*) s_col_k_);
    GlobalTensor<half> zp_k; zp_k.SetGlobalBuffer((__gm__ half*) zp_k_);
    GlobalTensor<half> s_row_k; s_row_k.SetGlobalBuffer((__gm__ half*) s_row_k_);
    // pipe_ptr->Reset();
    // OccupyMMTE1Events();
    // END: Auto inserted buffers/events/GMTensors
    // ----- User's codes start from here -----
    LocalTensor<half> ub_tile = AllocateLocalTensor<TPosition::VECCALC, half>(256 * 128);
    LocalTensor<half> ub_tile_t = AllocateLocalTensor<TPosition::VECCALC, half>(128 * 256);
    LocalTensor<float> ub_col_log_scale = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 128);
    LocalTensor<float> ub_row_log_scale = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 256);
    LocalTensor<float> ub_best_log_col = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 128);
    LocalTensor<float> ub_best_log_row = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 256);
    LocalTensor<float> ub_col_std = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 128);
    LocalTensor<float> ub_row_std = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 256);
    DBuff<uint8_t, TPosition::VECCALC> ub_q_scratch; ub_q_scratch.Init(1 * 512);
    LocalTensor<half> ub_s_col_half = AllocateLocalTensor<TPosition::VECCALC, half>(1 * 256);
    LocalTensor<half> ub_zp_half = AllocateLocalTensor<TPosition::VECCALC, half>(1 * 256);
    LocalTensor<half> ub_s_row_half = AllocateLocalTensor<TPosition::VECCALC, half>(1 * 128);
    LocalTensor<float> ub_best_imbalance = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 8);
    int tiles_per_vec = CeilDiv(tile_count, GetBlockNum());
    int tile_begin = tiles_per_vec*GetBlockIdx();
    int tile_end = Min(tile_begin + tiles_per_vec, tile_count);
    for (int tile_idx = tile_begin; tile_idx < tile_end; tile_idx += 1) {
        _tmp_sevent_valid_ubin_0.wait();
        GM2UBPAD(ub_tile, k_tile_rotated[tile_idx * 256 * 128], 256, 256, 0, 0);
        AscendC::NdDmaLoopInfo<2> _nd_dma_loop_info_ub_tile_t_281465764531920{{128, 1}, {1, 256}, {256, 128}, {0, 0}, {0, 0}};
        static constexpr AscendC::NdDmaConfig _nd_dma_config_ub_tile_t_281465764531920{false, AscendC::NdDmaConfig::unsetPad, AscendC::NdDmaConfig::unsetPad, false};
        GM2UB_ND_DMA<half, 2, _nd_dma_config_ub_tile_t_281465764531920>(ub_tile_t, k_tile_rotated[tile_idx * 256 * 128], _nd_dma_loop_info_ub_tile_t_281465764531920, (half)0);
        _tmp_sevent_ready_ubin_0.set();
        _tmp_sevent_ready_ubin_0.wait();
        _tmp_sevent_valid_ubout_0.wait();
        init_log_col_d256_vf((__ubuf__ float*) (ub_col_log_scale).GetPhyAddr());
        init_log_row_d256_vf((__ubuf__ float*) (ub_row_log_scale).GetPhyAddr());
        compute_col_std_d256_vf((__ubuf__ half*) (ub_tile_t).GetPhyAddr(), (__ubuf__ float*) (ub_col_log_scale).GetPhyAddr(), (__ubuf__ float*) (ub_row_log_scale).GetPhyAddr(), (__ubuf__ float*) (ub_col_std).GetPhyAddr());
        compute_row_std_d256_vf((__ubuf__ half*) (ub_tile).GetPhyAddr(), (__ubuf__ float*) (ub_col_log_scale).GetPhyAddr(), (__ubuf__ float*) (ub_row_log_scale).GetPhyAddr(), (__ubuf__ float*) (ub_row_std).GetPhyAddr());
        UB2UB(ub_best_log_col, ub_col_log_scale, 1, CeilDiv(128, 8), CeilDiv(0, 8), CeilDiv(0, 8));
        UB2UB(ub_best_log_row, ub_row_log_scale, 1, CeilDiv(256, 8), CeilDiv(0, 8), CeilDiv(0, 8));
        init_best_imbalance_d256_vf((__ubuf__ float*) (ub_best_imbalance).GetPhyAddr());
        update_best_d256_vf((__ubuf__ float*) (ub_col_std).GetPhyAddr(), (__ubuf__ float*) (ub_row_std).GetPhyAddr(), (__ubuf__ float*) (ub_col_log_scale).GetPhyAddr(), (__ubuf__ float*) (ub_row_log_scale).GetPhyAddr(), (__ubuf__ float*) (ub_best_log_col).GetPhyAddr(), (__ubuf__ float*) (ub_best_log_row).GetPhyAddr(), (__ubuf__ float*) (ub_best_imbalance).GetPhyAddr());
        for (int sinkhorn_step = 0; sinkhorn_step < sinkhorn_iters; sinkhorn_step += 1) {
            update_log_col_d256_vf((__ubuf__ float*) (ub_col_log_scale).GetPhyAddr(), (__ubuf__ float*) (ub_col_std).GetPhyAddr());
            compute_row_std_d256_vf((__ubuf__ half*) (ub_tile).GetPhyAddr(), (__ubuf__ float*) (ub_col_log_scale).GetPhyAddr(), (__ubuf__ float*) (ub_row_log_scale).GetPhyAddr(), (__ubuf__ float*) (ub_row_std).GetPhyAddr());
            update_log_row_d256_vf((__ubuf__ float*) (ub_row_log_scale).GetPhyAddr(), (__ubuf__ float*) (ub_row_std).GetPhyAddr());
            compute_col_std_d256_vf((__ubuf__ half*) (ub_tile_t).GetPhyAddr(), (__ubuf__ float*) (ub_col_log_scale).GetPhyAddr(), (__ubuf__ float*) (ub_row_log_scale).GetPhyAddr(), (__ubuf__ float*) (ub_col_std).GetPhyAddr());
            compute_row_std_d256_vf((__ubuf__ half*) (ub_tile).GetPhyAddr(), (__ubuf__ float*) (ub_col_log_scale).GetPhyAddr(), (__ubuf__ float*) (ub_row_log_scale).GetPhyAddr(), (__ubuf__ float*) (ub_row_std).GetPhyAddr());
            update_best_d256_vf((__ubuf__ float*) (ub_col_std).GetPhyAddr(), (__ubuf__ float*) (ub_row_std).GetPhyAddr(), (__ubuf__ float*) (ub_col_log_scale).GetPhyAddr(), (__ubuf__ float*) (ub_row_log_scale).GetPhyAddr(), (__ubuf__ float*) (ub_best_log_col).GetPhyAddr(), (__ubuf__ float*) (ub_best_log_row).GetPhyAddr(), (__ubuf__ float*) (ub_best_imbalance).GetPhyAddr());
        }
        _tmp_sevent_ready_ubout_0.set();
        _tmp_sevent_ready_ubout_0.wait();
        if (bits == 2) {
            for (int quant_row_2 = 0; quant_row_2 < 256; quant_row_2 += 1) {
                _tmp_devent_valid_ubout_1.wait();
                quantize_pack_2bit_d256_vf((__ubuf__ half*) (ub_tile[quant_row_2 * 128]).GetPhyAddr(), (__ubuf__ float*) (ub_best_log_col).GetPhyAddr(), (__ubuf__ float*) (ub_best_log_row).GetPhyAddr(), (__ubuf__ uint8_t*) (ub_q_scratch.get(quant_row_2)).GetPhyAddr(), (__ubuf__ half*) (ub_s_col_half[quant_row_2]).GetPhyAddr(), (__ubuf__ half*) (ub_zp_half[quant_row_2]).GetPhyAddr());
                _tmp_devent_ready_ubout_1.set();
                _tmp_devent_ready_ubout_1.wait();
                UB2GMPAD(q_packed[tile_idx * 256 * packed_group + quant_row_2 * packed_group], ub_q_scratch.get(quant_row_2), 2, 16, 0, 0);
                _tmp_devent_valid_ubout_1.set();
            }
        } else {
            for (int quant_row_4 = 0; quant_row_4 < 256; quant_row_4 += 1) {
                _tmp_devent_valid_ubout_1.wait();
                quantize_pack_4bit_d256_vf((__ubuf__ half*) (ub_tile[quant_row_4 * 128]).GetPhyAddr(), (__ubuf__ float*) (ub_best_log_col).GetPhyAddr(), (__ubuf__ float*) (ub_best_log_row).GetPhyAddr(), (__ubuf__ uint8_t*) (ub_q_scratch.get(quant_row_4)).GetPhyAddr(), (__ubuf__ half*) (ub_s_col_half[quant_row_4]).GetPhyAddr(), (__ubuf__ half*) (ub_zp_half[quant_row_4]).GetPhyAddr());
                _tmp_devent_ready_ubout_1.set();
                _tmp_devent_ready_ubout_1.wait();
                UB2GMPAD(q_packed[tile_idx * 256 * packed_group + quant_row_4 * packed_group], ub_q_scratch.get(quant_row_4), 1, 64, 14, packed_group - 64);
                _tmp_devent_valid_ubout_1.set();
            }
        }
        _tmp_sevent_valid_ubout_0.set();
        _tmp_sevent_valid_ubout_0.wait();
        cast_log_to_half_d256_vf((__ubuf__ float*) (ub_best_log_col).GetPhyAddr(), (__ubuf__ half*) (ub_s_row_half).GetPhyAddr());
        _tmp_sevent_ready_ubout_0.set();
        _tmp_sevent_ready_ubout_0.wait();
        UB2GMPAD(s_col_k[tile_idx * 256], ub_s_col_half, 1, 512, 0, 0);
        UB2GMPAD(zp_k[tile_idx * 256], ub_zp_half, 1, 512, 0, 0);
        UB2GMPAD(s_row_k[tile_idx * 128], ub_s_row_half, 1, 256, 0, 0);
        _tmp_sevent_valid_ubin_0.set();
        _tmp_sevent_valid_ubout_0.set();
    }

    // Auto generated code. Readability is not guaranteed.
}
