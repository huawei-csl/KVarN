#pragma once
#include "tensorutils.h"

__aicore__ inline void update_log_row_d256_vf(__ubuf__ float* log_scale, __ubuf__ float* std){
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg _tmp_maskreg_0 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<float> log_scale_regs[4];
        MicroAPI::RegTensor<float> std_regs[4];
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(log_scale_regs[0], log_scale);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(log_scale_regs[1], log_scale + 64);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(log_scale_regs[2], log_scale + 128);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(log_scale_regs[3], log_scale + 192);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(std_regs[0], std);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(std_regs[1], std + 64);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(std_regs[2], std + 128);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(std_regs[3], std + 192);
        MicroAPI::Maxs(std_regs[0], std_regs[0], 0.001f, _tmp_maskreg_0);
        MicroAPI::Maxs(std_regs[1], std_regs[1], 0.001f, _tmp_maskreg_0);
        MicroAPI::Maxs(std_regs[2], std_regs[2], 0.001f, _tmp_maskreg_0);
        MicroAPI::Maxs(std_regs[3], std_regs[3], 0.001f, _tmp_maskreg_0);
        MicroAPI::Mins(std_regs[0], std_regs[0], 1000.0f, _tmp_maskreg_0);
        MicroAPI::Mins(std_regs[1], std_regs[1], 1000.0f, _tmp_maskreg_0);
        MicroAPI::Mins(std_regs[2], std_regs[2], 1000.0f, _tmp_maskreg_0);
        MicroAPI::Mins(std_regs[3], std_regs[3], 1000.0f, _tmp_maskreg_0);
        MicroAPI::Log(std_regs[0], std_regs[0], _tmp_maskreg_0);
        MicroAPI::Log(std_regs[1], std_regs[1], _tmp_maskreg_0);
        MicroAPI::Log(std_regs[2], std_regs[2], _tmp_maskreg_0);
        MicroAPI::Log(std_regs[3], std_regs[3], _tmp_maskreg_0);
        MicroAPI::Add(log_scale_regs[0], log_scale_regs[0], std_regs[0], _tmp_maskreg_0);
        MicroAPI::Add(log_scale_regs[1], log_scale_regs[1], std_regs[1], _tmp_maskreg_0);
        MicroAPI::Add(log_scale_regs[2], log_scale_regs[2], std_regs[2], _tmp_maskreg_0);
        MicroAPI::Add(log_scale_regs[3], log_scale_regs[3], std_regs[3], _tmp_maskreg_0);
        MicroAPI::Maxs(log_scale_regs[0], log_scale_regs[0], -0.3f, _tmp_maskreg_0);
        MicroAPI::Maxs(log_scale_regs[1], log_scale_regs[1], -0.3f, _tmp_maskreg_0);
        MicroAPI::Maxs(log_scale_regs[2], log_scale_regs[2], -0.3f, _tmp_maskreg_0);
        MicroAPI::Maxs(log_scale_regs[3], log_scale_regs[3], -0.3f, _tmp_maskreg_0);
        MicroAPI::Mins(log_scale_regs[0], log_scale_regs[0], 10.0f, _tmp_maskreg_0);
        MicroAPI::Mins(log_scale_regs[1], log_scale_regs[1], 10.0f, _tmp_maskreg_0);
        MicroAPI::Mins(log_scale_regs[2], log_scale_regs[2], 10.0f, _tmp_maskreg_0);
        MicroAPI::Mins(log_scale_regs[3], log_scale_regs[3], 10.0f, _tmp_maskreg_0);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(log_scale, log_scale_regs[0], _tmp_maskreg_0);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(log_scale + 64, log_scale_regs[1], _tmp_maskreg_0);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(log_scale + 128, log_scale_regs[2], _tmp_maskreg_0);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(log_scale + 192, log_scale_regs[3], _tmp_maskreg_0);
    }
}

