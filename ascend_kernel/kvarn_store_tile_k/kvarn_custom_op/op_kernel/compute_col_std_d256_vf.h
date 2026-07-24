#pragma once
#include "tensorutils.h"

static constexpr MicroAPI::CastTrait compute_col_std_d256_vf_default_castcfg = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };

__aicore__ inline void compute_col_std_d256_vf(__ubuf__ half* src_t, __ubuf__ float* col_log_scale, __ubuf__ float* row_log_scale, __ubuf__ float* dst){
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg _tmp_maskreg_1 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<float> _tmp_reg_3;
        MicroAPI::RegTensor<float> _tmp_reg_2;
        MicroAPI::RegTensor<half> _tmp_reg_0;
        MicroAPI::RegTensor<float> values[4];
        MicroAPI::RegTensor<float> row_log_regs[4];
        MicroAPI::RegTensor<float> rows[4];
        MicroAPI::RegTensor<float> total;
        MicroAPI::RegTensor<float> mean;
        MicroAPI::RegTensor<float> centered[4];
        MicroAPI::RegTensor<float> sq_total;
        MicroAPI::RegTensor<float> variance;
        MicroAPI::RegTensor<float> std;
        MicroAPI::RegTensor<float> col_factor;
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(row_log_regs[0], row_log_scale);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(row_log_regs[1], row_log_scale + 64);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(row_log_regs[2], row_log_scale + 128);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(row_log_regs[3], row_log_scale + 192);
        for (uint16_t col_std_idx = (uint16_t)0; col_std_idx < (uint16_t)128; col_std_idx += (uint16_t)1) {
            MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(_tmp_reg_0, src_t + col_std_idx * 256);
            MicroAPI::Cast<float, half, compute_col_std_d256_vf_default_castcfg>(values[0], _tmp_reg_0, _tmp_maskreg_1);
            MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(_tmp_reg_0, src_t + col_std_idx * 256 + 64);
            MicroAPI::Cast<float, half, compute_col_std_d256_vf_default_castcfg>(values[1], _tmp_reg_0, _tmp_maskreg_1);
            MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(_tmp_reg_0, src_t + col_std_idx * 256 + 128);
            MicroAPI::Cast<float, half, compute_col_std_d256_vf_default_castcfg>(values[2], _tmp_reg_0, _tmp_maskreg_1);
            MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(_tmp_reg_0, src_t + col_std_idx * 256 + 192);
            MicroAPI::Cast<float, half, compute_col_std_d256_vf_default_castcfg>(values[3], _tmp_reg_0, _tmp_maskreg_1);
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(col_factor, col_log_scale + col_std_idx);
            MicroAPI::Duplicate(col_factor, col_factor, _tmp_maskreg_1);
            MicroAPI::Add(rows[0], row_log_regs[0], col_factor, _tmp_maskreg_1);
            MicroAPI::Add(rows[1], row_log_regs[1], col_factor, _tmp_maskreg_1);
            MicroAPI::Add(rows[2], row_log_regs[2], col_factor, _tmp_maskreg_1);
            MicroAPI::Add(rows[3], row_log_regs[3], col_factor, _tmp_maskreg_1);
            MicroAPI::Neg(rows[0], rows[0], _tmp_maskreg_1);
            MicroAPI::Neg(rows[1], rows[1], _tmp_maskreg_1);
            MicroAPI::Neg(rows[2], rows[2], _tmp_maskreg_1);
            MicroAPI::Neg(rows[3], rows[3], _tmp_maskreg_1);
            MicroAPI::Exp(rows[0], rows[0], _tmp_maskreg_1);
            MicroAPI::Exp(rows[1], rows[1], _tmp_maskreg_1);
            MicroAPI::Exp(rows[2], rows[2], _tmp_maskreg_1);
            MicroAPI::Exp(rows[3], rows[3], _tmp_maskreg_1);
            MicroAPI::Mul(values[0], values[0], rows[0], _tmp_maskreg_1);
            MicroAPI::Mul(values[1], values[1], rows[1], _tmp_maskreg_1);
            MicroAPI::Mul(values[2], values[2], rows[2], _tmp_maskreg_1);
            MicroAPI::Mul(values[3], values[3], rows[3], _tmp_maskreg_1);
            MicroAPI::Add(_tmp_reg_2, values[0], values[1], _tmp_maskreg_1);
            MicroAPI::Add(_tmp_reg_3, values[2], values[3], _tmp_maskreg_1);
            MicroAPI::Add(_tmp_reg_2, _tmp_reg_2, _tmp_reg_3, _tmp_maskreg_1);
            MicroAPI::ReduceSum(total, _tmp_reg_2, _tmp_maskreg_1);
            MicroAPI::Duplicate(mean, total, _tmp_maskreg_1);
            MicroAPI::Muls(mean, mean, 0.00390625f, _tmp_maskreg_1);
            MicroAPI::Sub(centered[0], values[0], mean, _tmp_maskreg_1);
            MicroAPI::Sub(centered[1], values[1], mean, _tmp_maskreg_1);
            MicroAPI::Sub(centered[2], values[2], mean, _tmp_maskreg_1);
            MicroAPI::Sub(centered[3], values[3], mean, _tmp_maskreg_1);
            MicroAPI::Mul(centered[0], centered[0], centered[0], _tmp_maskreg_1);
            MicroAPI::Mul(centered[1], centered[1], centered[1], _tmp_maskreg_1);
            MicroAPI::Mul(centered[2], centered[2], centered[2], _tmp_maskreg_1);
            MicroAPI::Mul(centered[3], centered[3], centered[3], _tmp_maskreg_1);
            MicroAPI::Add(_tmp_reg_2, centered[0], centered[1], _tmp_maskreg_1);
            MicroAPI::Add(_tmp_reg_3, centered[2], centered[3], _tmp_maskreg_1);
            MicroAPI::Add(_tmp_reg_2, _tmp_reg_2, _tmp_reg_3, _tmp_maskreg_1);
            MicroAPI::ReduceSum(sq_total, _tmp_reg_2, _tmp_maskreg_1);
            MicroAPI::Muls(variance, sq_total, 0.00392156862745098f, _tmp_maskreg_1);
            MicroAPI::Sqrt(std, variance, _tmp_maskreg_1);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dst + col_std_idx, std, _tmp_maskreg_1);
        }
    }
}

