#pragma once
#include "tensorutils.h"

__aicore__ inline void init_best_imbalance_vf(__ubuf__ float* dst){
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg _tmp_maskreg_0 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<float> value;
        MicroAPI::Duplicate(value, 1e+30f, _tmp_maskreg_0);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dst, value, _tmp_maskreg_0);
    }
}

