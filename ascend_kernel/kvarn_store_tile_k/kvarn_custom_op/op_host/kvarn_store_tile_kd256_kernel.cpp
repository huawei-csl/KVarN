#include "kvarn_store_tile_kd256_kernel_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

#define SET_ATTR_TO_TILING(name, dtype, idx) \
    const auto* name##_ptr = attrs->GetAttrPointer<dtype>(idx); \
    dtype name = *name##_ptr; \
    tiling.set_##name(name);
#define GET_ATTR_BY_IDX(name, dtype, idx) \
    const auto* name##_ptr = attrs->GetAttrPointer<dtype>(idx); \
    dtype name = *name##_ptr;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // set block size
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(64);
    context->SetDynUBufSize(221184);

    KvarnStoreTileKd256KernelTilingData tiling;
    auto attrs = context->GetAttrs();
    SET_ATTR_TO_TILING(bits, int32_t, 0);
    SET_ATTR_TO_TILING(sinkhorn_iters, int32_t, 1);
    SET_ATTR_TO_TILING(tile_count, int32_t, 2);
    SET_ATTR_TO_TILING(packed_group, int32_t, 3);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t userWorkspaceSize = 0;
    size_t sysWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    GET_ATTR_BY_IDX(bits, int32_t, 0);
    GET_ATTR_BY_IDX(sinkhorn_iters, int32_t, 1);
    GET_ATTR_BY_IDX(tile_count, int32_t, 2);
    GET_ATTR_BY_IDX(packed_group, int32_t, 3);
    gert::Shape* q_packed_shape = context->GetOutputShape(0);
    q_packed_shape->AppendDim(tile_count);
    q_packed_shape->AppendDim(256);
    q_packed_shape->AppendDim(packed_group);
    gert::Shape* s_col_k_shape = context->GetOutputShape(1);
    s_col_k_shape->AppendDim(tile_count);
    s_col_k_shape->AppendDim(256);
    gert::Shape* zp_k_shape = context->GetOutputShape(2);
    zp_k_shape->AppendDim(tile_count);
    zp_k_shape->AppendDim(256);
    gert::Shape* s_row_k_shape = context->GetOutputShape(3);
    s_row_k_shape->AppendDim(tile_count);
    s_row_k_shape->AppendDim(128);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_UINT8);
    context->SetOutputDataType(1, ge::DT_FLOAT16);
    context->SetOutputDataType(2, ge::DT_FLOAT16);
    context->SetOutputDataType(3, ge::DT_FLOAT16);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class KvarnStoreTileKd256Kernel : public OpDef {
public:
    explicit KvarnStoreTileKd256Kernel(const char* name) : OpDef(name)
    {
        this->Input("k_tile_rotated")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("q_packed")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("s_col_k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("zp_k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("s_row_k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("bits")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("sinkhorn_iters")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("tile_count")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("packed_group")
            .AttrType(REQUIRED)
            .Int(0);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(KvarnStoreTileKd256Kernel);
}
