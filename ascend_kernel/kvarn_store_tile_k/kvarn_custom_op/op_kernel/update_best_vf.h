#pragma once
#include "tensorutils.h"

__aicore__ inline void update_best_vf(__ubuf__ float* col_std, __ubuf__ float* row_std, __ubuf__ float* log_scale_col, __ubuf__ float* log_scale_row, __ubuf__ float* best_log_col, __ubuf__ float* best_log_row, __ubuf__ float* best_imbalance){
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg _tmp_maskreg_1 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<float> _tmp_reg_0;
        MicroAPI::RegTensor<float> col_regs[2];
        MicroAPI::RegTensor<float> row_regs[2];
        MicroAPI::RegTensor<float> col_max;
        MicroAPI::RegTensor<float> col_min;
        MicroAPI::RegTensor<float> row_max;
        MicroAPI::RegTensor<float> row_min;
        MicroAPI::RegTensor<float> col_ratio;
        MicroAPI::RegTensor<float> row_ratio;
        MicroAPI::RegTensor<float> current;
        MicroAPI::RegTensor<float> current_broadcast;
        MicroAPI::RegTensor<float> best;
        MicroAPI::RegTensor<float> selected;
        MicroAPI::MaskReg better = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALLF>();
        MicroAPI::RegTensor<float> log_scale_col_regs[2];
        MicroAPI::RegTensor<float> log_scale_row_regs[2];
        MicroAPI::RegTensor<float> best_log_col_regs[2];
        MicroAPI::RegTensor<float> best_log_row_regs[2];
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(col_regs[0], col_std);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(col_regs[1], col_std + 64);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(row_regs[0], row_std);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(row_regs[1], row_std + 64);
        MicroAPI::Max(_tmp_reg_0, col_regs[0], col_regs[1], _tmp_maskreg_1);
        MicroAPI::ReduceMax(col_max, _tmp_reg_0, _tmp_maskreg_1);
        MicroAPI::Min(_tmp_reg_0, col_regs[0], col_regs[1], _tmp_maskreg_1);
        MicroAPI::ReduceMin(col_min, _tmp_reg_0, _tmp_maskreg_1);
        MicroAPI::Max(_tmp_reg_0, row_regs[0], row_regs[1], _tmp_maskreg_1);
        MicroAPI::ReduceMax(row_max, _tmp_reg_0, _tmp_maskreg_1);
        MicroAPI::Min(_tmp_reg_0, row_regs[0], row_regs[1], _tmp_maskreg_1);
        MicroAPI::ReduceMin(row_min, _tmp_reg_0, _tmp_maskreg_1);
        MicroAPI::Maxs(col_min, col_min, 1e-08f, _tmp_maskreg_1);
        MicroAPI::Maxs(row_min, row_min, 1e-08f, _tmp_maskreg_1);
        MicroAPI::Div(col_ratio, col_max, col_min, _tmp_maskreg_1);
        MicroAPI::Div(row_ratio, row_max, row_min, _tmp_maskreg_1);
        MicroAPI::Add(current, col_ratio, row_ratio, _tmp_maskreg_1);
        MicroAPI::Duplicate(current_broadcast, current, _tmp_maskreg_1);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(best, best_imbalance);
        MicroAPI::Duplicate(best, best, _tmp_maskreg_1);
        MicroAPI::Compare<float, CMPMODE::LE>(better, current_broadcast, best, _tmp_maskreg_1);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(log_scale_col_regs[0], log_scale_col);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(log_scale_col_regs[1], log_scale_col + 64);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(log_scale_row_regs[0], log_scale_row);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(log_scale_row_regs[1], log_scale_row + 64);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(best_log_col_regs[0], best_log_col);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(best_log_col_regs[1], best_log_col + 64);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(best_log_row_regs[0], best_log_row);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(best_log_row_regs[1], best_log_row + 64);
        MicroAPI::Select(best_log_col_regs[0], log_scale_col_regs[0], best_log_col_regs[0], better);
        MicroAPI::Select(best_log_col_regs[1], log_scale_col_regs[1], best_log_col_regs[1], better);
        MicroAPI::Select(best_log_row_regs[0], log_scale_row_regs[0], best_log_row_regs[0], better);
        MicroAPI::Select(best_log_row_regs[1], log_scale_row_regs[1], best_log_row_regs[1], better);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(best_log_col, best_log_col_regs[0], _tmp_maskreg_1);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(best_log_col + 64, best_log_col_regs[1], _tmp_maskreg_1);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(best_log_row, best_log_row_regs[0], _tmp_maskreg_1);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(best_log_row + 64, best_log_row_regs[1], _tmp_maskreg_1);
        MicroAPI::Select(selected, current_broadcast, best, better);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(best_imbalance, selected, _tmp_maskreg_1);
    }
}

