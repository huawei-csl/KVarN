#pragma once
#include "tensorutils.h"

__aicore__ inline void update_log_col_vf(__ubuf__ float* log_scale, __ubuf__ float* std){
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg _tmp_maskreg_0 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<float> log_scale_regs[2];
        MicroAPI::RegTensor<float> std_regs[2];
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(log_scale_regs[0], log_scale);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(log_scale_regs[1], log_scale + 64);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(std_regs[0], std);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(std_regs[1], std + 64);
        MicroAPI::Maxs(std_regs[0], std_regs[0], 0.001f, _tmp_maskreg_0);
        MicroAPI::Maxs(std_regs[1], std_regs[1], 0.001f, _tmp_maskreg_0);
        MicroAPI::Mins(std_regs[0], std_regs[0], 1000.0f, _tmp_maskreg_0);
        MicroAPI::Mins(std_regs[1], std_regs[1], 1000.0f, _tmp_maskreg_0);
        MicroAPI::Log(std_regs[0], std_regs[0], _tmp_maskreg_0);
        MicroAPI::Log(std_regs[1], std_regs[1], _tmp_maskreg_0);
        MicroAPI::Add(log_scale_regs[0], log_scale_regs[0], std_regs[0], _tmp_maskreg_0);
        MicroAPI::Add(log_scale_regs[1], log_scale_regs[1], std_regs[1], _tmp_maskreg_0);
        MicroAPI::Maxs(log_scale_regs[0], log_scale_regs[0], -0.3f, _tmp_maskreg_0);
        MicroAPI::Maxs(log_scale_regs[1], log_scale_regs[1], -0.3f, _tmp_maskreg_0);
        MicroAPI::Mins(log_scale_regs[0], log_scale_regs[0], 10.0f, _tmp_maskreg_0);
        MicroAPI::Mins(log_scale_regs[1], log_scale_regs[1], 10.0f, _tmp_maskreg_0);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(log_scale, log_scale_regs[0], _tmp_maskreg_0);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(log_scale + 64, log_scale_regs[1], _tmp_maskreg_0);
    }
}

