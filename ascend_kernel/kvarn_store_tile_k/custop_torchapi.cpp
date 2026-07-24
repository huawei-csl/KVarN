#include "torch/extension.h" 
#include "torch_npu/csrc/core/npu/NPUStream.h" 
#include "acl/acl.h" 
#include "aclnn_kvarn_store_tile_k_kernel.h" 
#include "aclnn_kvarn_store_tile_kd256_kernel.h" 

#define CHECK_RET(x) \
    do { \
        auto __ret = x; \
        if (__ret != ACL_SUCCESS) { \
            std::cerr << __FILE__ << ":" << __LINE__ << " Error:" << __ret << std::endl; \
        } \
    } while (0); 

#define PREPARE_OP() uint64_t workspaceSize = 0; aclOpExecutor* executor = nullptr; void* workspaceAddr = nullptr; 
#define EXECOP(opname, stream, ...) \
    CHECK_RET(opname##GetWorkspaceSize(__VA_ARGS__, &workspaceSize, &executor)); \
    if (workspaceSize>0) CHECK_RET(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST)); \
    CHECK_RET(opname(workspaceAddr, workspaceSize, executor, stream)); \
    CHECK_RET(aclDestroyAclOpExecutor(executor)); \
    if (workspaceSize>0) CHECK_RET(aclrtFree(workspaceAddr));


aclTensor* toAclTensor(at::Tensor x, aclDataType aclT){ 
    std::vector<int64_t> shape = x.sizes().vec(); 
    std::vector<int64_t> strides = x.strides().vec(); 
    return aclCreateTensor(shape.data(), shape.size(), aclT, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), (void*)(x.storage().data())); 
}

void kvarn_store_tile_k_kernel(at::Tensor k_tile_rotated, at::Tensor q_packed, at::Tensor s_col_k, at::Tensor zp_k, at::Tensor s_row_k, int bits, int sinkhorn_iters, int tile_count, int packed_group) {
    PREPARE_OP();
    int devidx = k_tile_rotated.device().index();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    aclTensor* k_tile_rotated_acl = toAclTensor(k_tile_rotated, aclDataType::ACL_FLOAT16);
    aclTensor* q_packed_acl = toAclTensor(q_packed, aclDataType::ACL_UINT8);
    aclTensor* s_col_k_acl = toAclTensor(s_col_k, aclDataType::ACL_FLOAT16);
    aclTensor* zp_k_acl = toAclTensor(zp_k, aclDataType::ACL_FLOAT16);
    aclTensor* s_row_k_acl = toAclTensor(s_row_k, aclDataType::ACL_FLOAT16);
    EXECOP(aclnnKvarnStoreTileKKernel, stream, k_tile_rotated_acl, bits, sinkhorn_iters, tile_count, packed_group, q_packed_acl, s_col_k_acl, zp_k_acl, s_row_k_acl);
    aclDestroyTensor(k_tile_rotated_acl);
    aclDestroyTensor(q_packed_acl);
    aclDestroyTensor(s_col_k_acl);
    aclDestroyTensor(zp_k_acl);
    aclDestroyTensor(s_row_k_acl);
}

void kvarn_store_tile_kd256_kernel(at::Tensor k_tile_rotated, at::Tensor q_packed, at::Tensor s_col_k, at::Tensor zp_k, at::Tensor s_row_k, int bits, int sinkhorn_iters, int tile_count, int packed_group) {
    PREPARE_OP();
    int devidx = k_tile_rotated.device().index();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    aclTensor* k_tile_rotated_acl = toAclTensor(k_tile_rotated, aclDataType::ACL_FLOAT16);
    aclTensor* q_packed_acl = toAclTensor(q_packed, aclDataType::ACL_UINT8);
    aclTensor* s_col_k_acl = toAclTensor(s_col_k, aclDataType::ACL_FLOAT16);
    aclTensor* zp_k_acl = toAclTensor(zp_k, aclDataType::ACL_FLOAT16);
    aclTensor* s_row_k_acl = toAclTensor(s_row_k, aclDataType::ACL_FLOAT16);
    EXECOP(aclnnKvarnStoreTileKd256Kernel, stream, k_tile_rotated_acl, bits, sinkhorn_iters, tile_count, packed_group, q_packed_acl, s_col_k_acl, zp_k_acl, s_row_k_acl);
    aclDestroyTensor(k_tile_rotated_acl);
    aclDestroyTensor(q_packed_acl);
    aclDestroyTensor(s_col_k_acl);
    aclDestroyTensor(zp_k_acl);
    aclDestroyTensor(s_row_k_acl);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { 
    m.def("kvarn_store_tile_k_kernel", &kvarn_store_tile_k_kernel, "kvarn_store_tile_k_kernel"); 
    m.def("kvarn_store_tile_kd256_kernel", &kvarn_store_tile_kd256_kernel, "kvarn_store_tile_kd256_kernel"); 
}
