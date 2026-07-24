#pragma once
#include "tensorutils.h"

static constexpr MicroAPI::CastTrait quantize_pack_4bit_d256_vf_cast_cfg = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };
static constexpr MicroAPI::CastTrait quantize_pack_4bit_d256_vf_half_cfg = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };
static constexpr MicroAPI::CastTrait quantize_pack_4bit_d256_vf_default_castcfg = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };

__aicore__ inline void quantize_pack_4bit_d256_vf(__ubuf__ half* src, __ubuf__ float* best_log_col, __ubuf__ float* best_log_row, __ubuf__ uint8_t* q_scratch, __ubuf__ half* scale_half_out, __ubuf__ half* zp_half_out){
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg _tmp_maskreg_5 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg _tmp_maskreg_0 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<float> _tmp_reg_2;
        MicroAPI::RegTensor<half> _tmp_reg_1;
        MicroAPI::RegTensor<float> values[2];
        MicroAPI::RegTensor<float> col_regs[2];
        MicroAPI::RegTensor<float> row_factor;
        MicroAPI::RegTensor<float> lo;
        MicroAPI::RegTensor<float> hi;
        MicroAPI::RegTensor<float> scale;
        MicroAPI::RegTensor<float> zp;
        MicroAPI::RegTensor<float> value_range;
        MicroAPI::RegTensor<float> range_vec;
        MicroAPI::RegTensor<float> zp_vec;
        MicroAPI::RegTensor<float> abs_scale;
        MicroAPI::RegTensor<float> abs_zp;
        MicroAPI::RegTensor<half> half_scale;
        MicroAPI::RegTensor<half> half_zp;
        MicroAPI::RegTensor<float> q_values[2];
        MicroAPI::RegTensor<int> q_int[2];
        MicroAPI::RegTensor<uint16_t> q_u16[2];
        MicroAPI::RegTensor<uint8_t> dense_u8[2];
        MicroAPI::RegTensor<uint8_t> even_u8;
        MicroAPI::RegTensor<uint8_t> odd_u8;
        MicroAPI::RegTensor<uint8_t> shift_u8;
        MicroAPI::RegTensor<uint8_t> packed_u8;
        MicroAPI::MaskReg float_mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg u8_mask = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL128>();
        MicroAPI::MaskReg q4_store_mask = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL32>();
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(col_regs[0], best_log_col);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(col_regs[1], best_log_col + 64);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(row_factor, best_log_row);
        MicroAPI::Duplicate(row_factor, row_factor, _tmp_maskreg_0);
        MicroAPI::Add(col_regs[0], col_regs[0], row_factor, _tmp_maskreg_0);
        MicroAPI::Add(col_regs[1], col_regs[1], row_factor, _tmp_maskreg_0);
        MicroAPI::Neg(col_regs[0], col_regs[0], _tmp_maskreg_0);
        MicroAPI::Neg(col_regs[1], col_regs[1], _tmp_maskreg_0);
        MicroAPI::Exp(col_regs[0], col_regs[0], _tmp_maskreg_0);
        MicroAPI::Exp(col_regs[1], col_regs[1], _tmp_maskreg_0);
        MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(_tmp_reg_1, src);
        MicroAPI::Cast<float, half, quantize_pack_4bit_d256_vf_default_castcfg>(values[0], _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::LoadAlign<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(_tmp_reg_1, src + 64);
        MicroAPI::Cast<float, half, quantize_pack_4bit_d256_vf_default_castcfg>(values[1], _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::Mul(values[0], values[0], col_regs[0], _tmp_maskreg_0);
        MicroAPI::Mul(values[1], values[1], col_regs[1], _tmp_maskreg_0);
        MicroAPI::Min(_tmp_reg_2, values[0], values[1], _tmp_maskreg_0);
        MicroAPI::ReduceMin(lo, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Max(_tmp_reg_2, values[0], values[1], _tmp_maskreg_0);
        MicroAPI::ReduceMax(hi, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Copy(zp, lo, _tmp_maskreg_0);
        MicroAPI::Sub(_tmp_reg_2, hi, lo, _tmp_maskreg_0);
        MicroAPI::Muls(scale, _tmp_reg_2, 0.06666666666666667f, _tmp_maskreg_0);
        MicroAPI::Maxs(scale, scale, 1e-10f, _tmp_maskreg_0);
        MicroAPI::Sub(value_range, hi, lo, _tmp_maskreg_0);
        MicroAPI::Maxs(value_range, value_range, 1.5e-09f, _tmp_maskreg_0);
        MicroAPI::Duplicate(range_vec, value_range, _tmp_maskreg_0);
        MicroAPI::Duplicate(zp_vec, zp, _tmp_maskreg_0);
        MicroAPI::Sub(q_values[0], values[0], zp_vec, _tmp_maskreg_0);
        MicroAPI::Sub(q_values[1], values[1], zp_vec, _tmp_maskreg_0);
        MicroAPI::Muls(q_values[0], q_values[0], 15.0f, _tmp_maskreg_0);
        MicroAPI::Muls(q_values[1], q_values[1], 15.0f, _tmp_maskreg_0);
        MicroAPI::Div(q_values[0], q_values[0], range_vec, _tmp_maskreg_0);
        MicroAPI::Div(q_values[1], q_values[1], range_vec, _tmp_maskreg_0);
        MicroAPI::Maxs(q_values[0], q_values[0], 0.0f, _tmp_maskreg_0);
        MicroAPI::Maxs(q_values[1], q_values[1], 0.0f, _tmp_maskreg_0);
        MicroAPI::Mins(q_values[0], q_values[0], 15.0f, _tmp_maskreg_0);
        MicroAPI::Mins(q_values[1], q_values[1], 15.0f, _tmp_maskreg_0);
        MicroAPI::Cast<int, float, quantize_pack_4bit_d256_vf_cast_cfg>(q_int[0], q_values[0], float_mask);
        MicroAPI::RegTensor<uint32_t>& _reg_3 = *(MicroAPI::RegTensor<uint32_t>*) &q_int[0];
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(q_u16[0], _reg_3);
        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(dense_u8[0], q_u16[0]);
        MicroAPI::DeInterleave(even_u8, odd_u8, dense_u8[0], dense_u8[0]);
        MicroAPI::ShiftLefts(shift_u8, odd_u8, (int16_t)4, u8_mask);
        MicroAPI::Or(packed_u8, even_u8, shift_u8, u8_mask);
        MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_NORM_B8>(q_scratch, packed_u8, q4_store_mask);
        MicroAPI::Cast<int, float, quantize_pack_4bit_d256_vf_cast_cfg>(q_int[1], q_values[1], float_mask);
        MicroAPI::RegTensor<uint32_t>& _reg_4 = *(MicroAPI::RegTensor<uint32_t>*) &q_int[1];
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(q_u16[1], _reg_4);
        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(dense_u8[1], q_u16[1]);
        MicroAPI::DeInterleave(even_u8, odd_u8, dense_u8[1], dense_u8[1]);
        MicroAPI::ShiftLefts(shift_u8, odd_u8, (int16_t)4, u8_mask);
        MicroAPI::Or(packed_u8, even_u8, shift_u8, u8_mask);
        MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_NORM_B8>(q_scratch + 32, packed_u8, q4_store_mask);
        MicroAPI::Exp(row_factor, row_factor, _tmp_maskreg_0);
        MicroAPI::Mul(abs_scale, row_factor, scale, _tmp_maskreg_0);
        MicroAPI::Mul(abs_zp, row_factor, zp, _tmp_maskreg_0);
        MicroAPI::Cast<half, float, quantize_pack_4bit_d256_vf_half_cfg>(half_scale, abs_scale, _tmp_maskreg_5);
        MicroAPI::Cast<half, float, quantize_pack_4bit_d256_vf_half_cfg>(half_zp, abs_zp, _tmp_maskreg_5);
        MicroAPI::StoreAlign<half, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(scale_half_out, half_scale, _tmp_maskreg_5);
        MicroAPI::StoreAlign<half, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(zp_half_out, half_zp, _tmp_maskreg_5);
    }
}

