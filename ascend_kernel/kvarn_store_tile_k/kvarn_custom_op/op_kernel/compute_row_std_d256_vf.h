#pragma once
#include "tensorutils.h"

static constexpr MicroAPI::CastTrait compute_row_std_d256_vf_default_castcfg = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };

__aicore__ inline void compute_row_std_d256_vf(__ubuf__ half* src, __ubuf__ float* col_log_scale, __ubuf__ float* row_log_scale, __ubuf__ float* dst){
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg _tmp_maskreg_1 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<float> _tmp_reg_2;
        MicroAPI::RegTensor<half> _tmp_reg_0;
        MicroAPI::RegTensor<float> values[2];
        MicroAPI::RegTensor<float> col_log_regs[2];
        MicroAPI::RegTensor<float> cols[2];
        MicroAPI::RegTensor<float> total;
        MicroAPI::RegTensor<float> mean;
        MicroAPI::RegTensor<float> centered[2];
        MicroAPI::RegTensor<float> sq_total;
        MicroAPI::RegTensor<float> variance;
        MicroAPI::RegTensor<float> std;
        MicroAPI::RegTensor<float> row_factor;
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(col_log_regs[0], col_log_scale);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(col_log_regs[1], col_log_scale + 64);
        for (uint16_t row_std_idx = (uint16_t)0; row_std_idx < (uint16_t)256; row_std_idx += (uint16_t)1) {
            MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(_tmp_reg_0, src + row_std_idx * 128);
            MicroAPI::Cast<float, half, compute_row_std_d256_vf_default_castcfg>(values[0], _tmp_reg_0, _tmp_maskreg_1);
            MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(_tmp_reg_0, src + row_std_idx * 128 + 64);
            MicroAPI::Cast<float, half, compute_row_std_d256_vf_default_castcfg>(values[1], _tmp_reg_0, _tmp_maskreg_1);
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(row_factor, row_log_scale + row_std_idx);
            MicroAPI::Duplicate(row_factor, row_factor, _tmp_maskreg_1);
            MicroAPI::Add(cols[0], col_log_regs[0], row_factor, _tmp_maskreg_1);
            MicroAPI::Add(cols[1], col_log_regs[1], row_factor, _tmp_maskreg_1);
            MicroAPI::Neg(cols[0], cols[0], _tmp_maskreg_1);
            MicroAPI::Neg(cols[1], cols[1], _tmp_maskreg_1);
            MicroAPI::Exp(cols[0], cols[0], _tmp_maskreg_1);
            MicroAPI::Exp(cols[1], cols[1], _tmp_maskreg_1);
            MicroAPI::Mul(values[0], values[0], cols[0], _tmp_maskreg_1);
            MicroAPI::Mul(values[1], values[1], cols[1], _tmp_maskreg_1);
            MicroAPI::Add(_tmp_reg_2, values[0], values[1], _tmp_maskreg_1);
            MicroAPI::ReduceSum(total, _tmp_reg_2, _tmp_maskreg_1);
            MicroAPI::Duplicate(mean, total, _tmp_maskreg_1);
            MicroAPI::Muls(mean, mean, 0.0078125f, _tmp_maskreg_1);
            MicroAPI::Sub(centered[0], values[0], mean, _tmp_maskreg_1);
            MicroAPI::Sub(centered[1], values[1], mean, _tmp_maskreg_1);
            MicroAPI::Mul(centered[0], centered[0], centered[0], _tmp_maskreg_1);
            MicroAPI::Mul(centered[1], centered[1], centered[1], _tmp_maskreg_1);
            MicroAPI::Add(_tmp_reg_2, centered[0], centered[1], _tmp_maskreg_1);
            MicroAPI::ReduceSum(sq_total, _tmp_reg_2, _tmp_maskreg_1);
            MicroAPI::Muls(variance, sq_total, 0.007874015748031496f, _tmp_maskreg_1);
            MicroAPI::Sqrt(std, variance, _tmp_maskreg_1);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dst + row_std_idx, std, _tmp_maskreg_1);
        }
    }
}

