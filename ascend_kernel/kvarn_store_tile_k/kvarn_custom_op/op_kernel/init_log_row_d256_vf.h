#pragma once
#include "tensorutils.h"

__aicore__ inline void init_log_row_d256_vf(__ubuf__ float* dst){
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg _tmp_maskreg_0 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<float> values[4];
        MicroAPI::Duplicate(values[0], 0.0f, _tmp_maskreg_0);
        MicroAPI::Duplicate(values[1], 0.0f, _tmp_maskreg_0);
        MicroAPI::Duplicate(values[2], 0.0f, _tmp_maskreg_0);
        MicroAPI::Duplicate(values[3], 0.0f, _tmp_maskreg_0);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(dst, values[0], _tmp_maskreg_0);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(dst + 64, values[1], _tmp_maskreg_0);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(dst + 128, values[2], _tmp_maskreg_0);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(dst + 192, values[3], _tmp_maskreg_0);
    }
}

