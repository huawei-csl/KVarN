#pragma once
#include "tensorutils.h"

static constexpr MicroAPI::CastTrait cast_log_to_half_vf_cfg = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };

__aicore__ inline void cast_log_to_half_vf(__ubuf__ float* src_log, __ubuf__ half* dst){
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg _tmp_maskreg_1 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg _tmp_maskreg_0 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<float> values[2];
        MicroAPI::RegTensor<half> half_values[2];
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(values[0], src_log);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(values[1], src_log + 64);
        MicroAPI::Exp(values[0], values[0], _tmp_maskreg_0);
        MicroAPI::Exp(values[1], values[1], _tmp_maskreg_0);
        MicroAPI::Cast<half, float, cast_log_to_half_vf_cfg>(half_values[0], values[0], _tmp_maskreg_1);
        MicroAPI::StoreAlign<half, MicroAPI::StoreDist::DIST_PACK_B32>(dst, half_values[0], _tmp_maskreg_1);
        MicroAPI::Cast<half, float, cast_log_to_half_vf_cfg>(half_values[1], values[1], _tmp_maskreg_1);
        MicroAPI::StoreAlign<half, MicroAPI::StoreDist::DIST_PACK_B32>(dst + 64, half_values[1], _tmp_maskreg_1);
    }
}

