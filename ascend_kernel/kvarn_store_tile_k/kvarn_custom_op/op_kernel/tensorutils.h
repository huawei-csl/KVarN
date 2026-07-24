#pragma once
#include "kernel_operator.h"
#include <type_traits>


using namespace AscendC;

#define PIPE_FIX (pipe_t)10

#define ALLCUBE_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x0, append_to_pipe>(iiii)
#define ALLVEC_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x0, append_to_pipe>(iiii)
#define INTRACORE_ALLVEC_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x1, append_to_pipe>(iiii)

// A5/C310: cube (AIC) and the two vec sub-blocks (AIV0, AIV1) share one block, so the
// cube<->vec point-to-point handoff uses the cheap hardware intra-block sync (sync mode
// 0x4 -> set_intra_block / wait_intra_block, emitting the SET_INTRA_BLOCK / WAIT_INTRA_BLOCK
// ISA ops) instead of the FFTS cross-core path (mode 0x2 -> ffts_cross_core_sync /
// wait_flag_dev, emitting SET_CROSS_CORE / WAIT_FLAG_DEV). AscendC NotifyEventImpl /
// WaitEventImpl dispatch on the mode: modeId==4 -> intra_block, else -> FFTS
// (dav_c310/kernel_operator_sync_impl.h). The all-core barriers (ALLCUBE / ALLVEC /
// INTRACORE) stay on FFTS below: they are genuinely cross-block.
//
// Mode 4 is per-sub-block (API doc atlasascendc_api_07_0273): an AIV1-issued flagId N is
// seen by AIC as flagId N+16, while AIV0's flagId N stays N. So the CUBE side must
// signal/await BOTH sub-blocks -- the "double-send" here and the "double-wait" in WAIT_VEC,
// on flagId N and N+16 -- while each AIV side issues a single flagId N (the hardware adds
// the +16 stride for AIV1). Without the double-send only one vec sub-block is released and
// the other deadlocks. Constraint: user flag_id must be <= 10 so that N+16 (16..26) stays
// clear of the AscendC reserved flags (11..15) and their AIV1 mirror (27..31).
#ifdef __DAV_C310__
#define CUBE_READY(iiii, append_to_pipe) \
    do { \
        CrossCoreSetFlag<0x4, append_to_pipe>(iiii); \
        CrossCoreSetFlag<0x4, append_to_pipe>((iiii) + 16); \
    } while (0)
#define VEC_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x4, append_to_pipe>(iiii)
#else
#define CUBE_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x2, append_to_pipe>(iiii)
#define VEC_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x2, append_to_pipe>(iiii)
#endif

#ifdef __DAV_C310__
#define ALLCUBE_WAIT(iiii, append_to_pipe) CrossCoreWaitFlag<0x0, append_to_pipe>(iiii)
#define ALLVEC_WAIT(iiii, append_to_pipe) CrossCoreWaitFlag<0x0, append_to_pipe>(iiii)
#define INTRACORE_ALLVEC_WAIT(iiii, append_to_pipe) CrossCoreWaitFlag<0x1, append_to_pipe>(iiii)
#define WAIT_CUBE(iiii, append_to_pipe) CrossCoreWaitFlag<0x4, append_to_pipe>(iiii)
// Cube-side double-wait: await both vec sub-blocks (AIV0 on flagId N, AIV1 on N+16).
// See the CUBE_READY note above for the mode-4 per-sub-block mapping.
#define WAIT_VEC(iiii, append_to_pipe) \
    do { \
        CrossCoreWaitFlag<0x4, append_to_pipe>(iiii); \
        CrossCoreWaitFlag<0x4, append_to_pipe>((iiii) + 16); \
    } while (0)
#else
#define ALLCUBE_WAIT(iiii, append_to_pipe) CrossCoreWaitFlag(iiii)
#define ALLVEC_WAIT(iiii, append_to_pipe) CrossCoreWaitFlag(iiii)
#define INTRACORE_ALLVEC_WAIT(iiii, append_to_pipe) CrossCoreWaitFlag(iiii)
#define WAIT_CUBE(iiii, append_to_pipe) CrossCoreWaitFlag(iiii)
#define WAIT_VEC(iiii, append_to_pipe) CrossCoreWaitFlag(iiii)
#endif

constexpr uint64_t VECTORFULLMASK[2] = {(uint64_t)-1, (uint64_t)-1};
#ifdef __DAV_C310__
constexpr FixpipeConfig CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};
#endif 

__aicore__ constexpr HardEvent GetHardEventByPipe(pipe_t src, pipe_t dst){
    if (src==PIPE_MTE2){
        if (dst==PIPE_MTE1){
            return HardEvent::MTE2_MTE1;
        }else if(dst==PIPE_V){
            return HardEvent::MTE2_V;
        }else if(dst==PIPE_MTE3){
            return HardEvent::MTE2_MTE3;
        }
    }else if(src==PIPE_MTE1){
        if (dst==PIPE_MTE2){
            return HardEvent::MTE1_MTE2;
        }else if(dst==PIPE_M){
            return HardEvent::MTE1_M;
        }else if(dst==PIPE_FIX){
            return HardEvent::MTE1_FIX;
        }
    }else if(src==PIPE_M){
        if (dst==PIPE_MTE1){
            return HardEvent::M_MTE1;
        }else if (dst==PIPE_FIX){
            return HardEvent::M_FIX;
        }
    }else if(src==PIPE_FIX){
        if (dst==PIPE_M){
            return HardEvent::FIX_M;
        }else if (dst==PIPE_MTE1){
            return HardEvent::FIX_MTE1;
        }
    }else if(src==PIPE_V){
        if (dst==PIPE_MTE2){
            return HardEvent::V_MTE2;
        }else if(dst==PIPE_MTE3){
            return HardEvent::V_MTE3;
        }
    }else if(src==PIPE_MTE3){
        if (dst==PIPE_V){
            return HardEvent::MTE3_V;
        }else if(dst==PIPE_MTE2){
            return HardEvent::MTE3_MTE2;
        }
    }
    return HardEvent::MTE3_MTE2;
}

__aicore__ inline void OccupyMMTE1Events(){
    if ASCEND_IS_AIC{
        TPipe* pipe_ptr = GetTPipePtr();
        pipe_ptr->AllocEventID<HardEvent::M_MTE1>();
        pipe_ptr->AllocEventID<HardEvent::M_MTE1>();
        pipe_ptr->AllocEventID<HardEvent::M_MTE1>();
    }
}


__aicore__ constexpr int Align8B(int x){
    return (x + 7) / 8 * 8;
}

__aicore__ constexpr int Align16B(int x){
    return (x + 15) / 16 * 16;
}

__aicore__ constexpr int Align32B(int x){
    return (x + 31) / 32 * 32;
}

__aicore__ constexpr int Align64B(int x){
    return (x + 63) / 64 * 64;
}

__aicore__ constexpr int Align128B(int x){
    return (x + 127) / 128 * 128;
}

__aicore__ constexpr int Align256B(int x){
    return (x + 255) / 256 * 256;
}

__aicore__ constexpr int Align512B(int x){
    return (x + 511) / 512 * 512;
}

__aicore__ inline int CeilDiv(int a, int b){
    return (b==0) ? 0 : ( (a + b - 1) / b );
}

__aicore__ inline void SetVectorMaskByCount(int count){
    uint64_t maskLow;
    uint64_t maskHigh;
    if (count <= 0){
        maskLow = 0;
        maskHigh = 0;
    }else if (count < 64){
        maskLow = ((uint64_t)1 << count) - 1;
        maskHigh = 0;
    }else if (count == 64){
        maskLow = (uint64_t)-1;
        maskHigh = 0;
    }else if (count < 128){
        maskLow = (uint64_t)-1;
        maskHigh = ((uint64_t)1 << (count - 64)) - 1;
    }else{
        maskLow = (uint64_t)-1;
        maskHigh = (uint64_t)-1;
    }
    SetVectorMask<half, MaskMode::NORMAL>(maskHigh, maskLow);
}

template <typename T, typename T1, typename T2>
__aicore__ inline T1 shiftAddr(T1 base, uint64_t size, T2 &offset){
    auto res = base + offset;
    offset += size*sizeof(T);
    return res;
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b){
    return (a<b) ? a : b;
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b){
    return (a<b) ? b : a;
}

/* ------------- Tensor ------------- */ 
template <TPosition pos, typename T>
__aicore__ inline LocalTensor<T> AllocateLocalTensor(int len){
    TBuf<pos> tbuf;
    TPipe* ptr = GetTPipePtr();
    ptr->InitBuffer(tbuf, len * sizeof(T));
    return tbuf.template Get<T>();
}

/* ------------- Tensor ------------- */ 


/* ------------- Double Buffer ------------- */ 

template <typename T, TPosition pos>
class DBuff{
public:
    __aicore__ inline DBuff(){}
    __aicore__ inline void Init(int len){
        TPipe* ptr = GetTPipePtr();
        ptr->InitBuffer(buf1, len * sizeof(T));
        ptr->InitBuffer(buf2, len * sizeof(T));
        tsr1 = buf1.template Get<T>();
        tsr2 = buf2.template Get<T>();
    }
    
    __aicore__ inline LocalTensor<T> get(int i){
        if (i%2==0){
            return tsr1;
        }else{
            return tsr2;
        }
    }
private:
    TBuf<pos> buf1, buf2;
    LocalTensor<T> tsr1, tsr2;
};
/* ------------- Double Buffer ------------- */ 


/* ------------- Triple Buffer ------------- */ 
template <typename T, TPosition pos>
class TBuff{
public:
    __aicore__ inline TBuff(){}
    __aicore__ inline void Init(int len){
        TPipe* ptr = GetTPipePtr();
        ptr->InitBuffer(buf1, len * sizeof(T));
        ptr->InitBuffer(buf2, len * sizeof(T));
        ptr->InitBuffer(buf3, len * sizeof(T));
        tsr1 = buf1.template Get<T>();
        tsr2 = buf2.template Get<T>();
        tsr3 = buf3.template Get<T>();
    }
    
    __aicore__ inline LocalTensor<T> get(int i){
        if (i%3==0){
            return tsr1;
        }else if (i%3==1){
            return tsr2;
        }else{
            return tsr3;
        }
    }
private:
    TBuf<pos> buf1, buf2, buf3;
    LocalTensor<T> tsr1, tsr2, tsr3;
};
/* ------------- Triple Buffer ------------- */ 


/* ------------- Quadro Buffer ------------- */ 
template <typename T, TPosition pos>
class QBuff{
public:
    __aicore__ inline QBuff(){}
    __aicore__ inline void Init(int len){
        TPipe* ptr = GetTPipePtr();
        ptr->InitBuffer(buf1, len * sizeof(T));
        ptr->InitBuffer(buf2, len * sizeof(T));
        ptr->InitBuffer(buf3, len * sizeof(T));
        ptr->InitBuffer(buf4, len * sizeof(T));
        tsr1 = buf1.template Get<T>();
        tsr2 = buf2.template Get<T>();
        tsr3 = buf3.template Get<T>();
        tsr4 = buf4.template Get<T>();
    }
    
    __aicore__ inline LocalTensor<T> get(int i){
        if (i%4==0){
            return tsr1;
        }else if (i%4==1){
            return tsr2;
        }else if (i%4==2){
            return tsr3;
        }else{
            return tsr4;
        }
    }
private:
    TBuf<pos> buf1, buf2, buf3, buf4;
    LocalTensor<T> tsr1, tsr2, tsr3, tsr4;
};
/* ------------- Quadro Buffer ------------- */ 


/* ------------- Penta Buffer ------------- */ 
template <typename T, TPosition pos>
class PBuff{
public:
    __aicore__ inline PBuff(){}
    __aicore__ inline void Init(int len){
        TPipe* ptr = GetTPipePtr();
        ptr->InitBuffer(buf1, len * sizeof(T));
        ptr->InitBuffer(buf2, len * sizeof(T));
        ptr->InitBuffer(buf3, len * sizeof(T));
        ptr->InitBuffer(buf4, len * sizeof(T));
        ptr->InitBuffer(buf5, len * sizeof(T));
        tsr1 = buf1.template Get<T>();
        tsr2 = buf2.template Get<T>();
        tsr3 = buf3.template Get<T>();
        tsr4 = buf4.template Get<T>();
        tsr5 = buf5.template Get<T>();
    }
    
    __aicore__ inline LocalTensor<T> get(int i){
        if (i%5==0){
            return tsr1;
        }else if (i%5==1){
            return tsr2;
        }else if (i%5==2){
            return tsr3;
        }else if (i%5==3){
            return tsr4;
        }else{
            return tsr5;
        }
    }
private:
    TBuf<pos> buf1, buf2, buf3, buf4, buf5;
    LocalTensor<T> tsr1, tsr2, tsr3, tsr4, tsr5;
};
/* ------------- Penta Buffer ------------- */ 


/* ------------- Events ------------- */ 

template <pipe_t p1, pipe_t p2, bool preset>
class SEvent{
public:
    __aicore__ inline SEvent(){
        TPipe* pipe_ptr = GetTPipePtr();
        id1 = (event_t)pipe_ptr->AllocEventID<GetHardEventByPipe(p1, p2)>();
        if constexpr (preset){
            setall();
        }
    }
    __aicore__ inline ~SEvent(){
        if constexpr (preset){
            release();
        }
    }
    __aicore__ inline void wait(){
        wait_flag(p1, p2, id1);
    }
    __aicore__ inline void set(){
        set_flag(p1, p2, id1);
    }
    __aicore__ inline void setall(){
        set();
    }
    __aicore__ inline void release(){
        wait();
    }

private:
    event_t id1;
};



template <pipe_t p1, pipe_t p2, bool preset>
class DEvent{
public:
    __aicore__ inline DEvent(){
        TPipe* pipe_ptr = GetTPipePtr();
        id1 = (event_t)pipe_ptr->AllocEventID<GetHardEventByPipe(p1, p2)>();
        id2 = (event_t)pipe_ptr->AllocEventID<GetHardEventByPipe(p1, p2)>();
        if constexpr (preset){
            setall();
        }
    }
    __aicore__ inline ~DEvent(){
        if constexpr (preset){
            release();
        }
    }
    __aicore__ inline void wait(){
        if (wait_cnt%2==0){
            wait_flag(p1, p2, id1);
        }else{
            wait_flag(p1, p2, id2);
        }
        wait_cnt ++;
    }
    __aicore__ inline void set(){
        if (set_cnt%2==0){
            set_flag(p1, p2, id1);
        }else{
            set_flag(p1, p2, id2);
        }
        set_cnt ++;
    }
    __aicore__ inline void setall(){
        set();
        set();
    }
    __aicore__ inline void release(){
        for (int i=wait_cnt; i<set_cnt; ++i){
            wait();
        }
    }

private:
    event_t id1, id2;
    int wait_cnt = 0;
    int set_cnt = 0;
};



/* ------------- Funcs -------------- */
template<typename T>
__aicore__ inline void GM2L1_ND2NZ(LocalTensor<T> dst, GlobalTensor<T> src, int h, int w, int W, int Hdst){
    Nd2NzParams param;
    param.ndNum = 1;
    param.nValue = h;
    param.dValue = w;
    param.srcNdMatrixStride = 0;
    param.srcDValue = W;
    param.dstNzC0Stride = (Hdst + 15) / 16 * 16;
    param.dstNzNStride = 1;
    param.dstNzMatrixStride = 0;
    DataCopy(dst, src, param);
}

#ifdef __DAV_C310__
// DN2NZ: GM holds the matrix transposed (GM[n, m] -> dst NZ[m, n]); W is the GM row stride (>= h).
template<typename T>
__aicore__ inline void GM2L1_DN2NZ(LocalTensor<T> dst, GlobalTensor<T> src, int h, int w, int W, int Hdst){
    Dn2NzParams param;
    param.dnNum = 1;
    param.nValue = h;
    param.dValue = w;
    param.srcDnMatrixStride = 0;
    param.srcDValue = W;
    param.dstNzC0Stride = (Hdst + 15) / 16 * 16;
    param.dstNzNStride = 1;
    param.dstNzMatrixStride = 0;
    DataCopy(dst, src, param);
}

// Dense MX scale: GM stores logical e8m0 scale [rows, k_groups] row-major. Reinterpret
// adjacent e8m0 pairs as half, then use Dn2Nz to materialize the compact physical stream
// consumed by l1_to_l0_mx: [ceil(rows / 16), ceil(k_groups / 2), 16, 2 bytes].
template<typename T>
__aicore__ inline void GM2L1_MX_SCALE_DN2NZ(
    LocalTensor<T> dst,
    GlobalTensor<T> src,
    int rows,
    int kGroups,
    int srcKGroups)
{
    static_assert(sizeof(T) == 1 && std::is_integral<T>::value,
                  "GM2L1_MX_SCALE_DN2NZ expects uint8_t-like dst/src");
    LocalTensor<half> dstHalf = dst.template ReinterpretCast<half>();
    GlobalTensor<half> srcHalf = src.template ReinterpretCast<half>();
    int dstHalfCols = CeilDiv(kGroups, 2);
    int srcHalfCols = CeilDiv(srcKGroups, 2);
    Dn2NzParams param;
    param.dnNum = 1;
    param.nValue = dstHalfCols;
    param.dValue = rows;
    param.srcDnMatrixStride = 0;
    param.srcDValue = srcHalfCols;
    param.dstNzC0Stride = dstHalfCols;
    param.dstNzNStride = 1;
    param.dstNzMatrixStride = 0;
    DataCopy(dstHalf, srcHalf, param);
}
#endif

// NOTE: This only support float
__aicore__ inline void GM2L1_ND2ZZ(LocalTensor<float> dst, GlobalTensor<float> src, int h, int w, int W, int Hdst){
    if (W<4096){
        Nd2NzParams param;
        param.ndNum = h/16;
        param.nValue = 16;
        param.dValue = w;
        param.srcNdMatrixStride = W*16;
        param.srcDValue = W;
        param.dstNzC0Stride = 16;
        param.dstNzNStride = 1;
        param.dstNzMatrixStride = Align16B(w)*16;
        DataCopy(dst, src, param);
    }else{
        Nd2NzParams param;
        param.ndNum = 1;
        param.nValue = 16;
        param.dValue = w;
        param.srcNdMatrixStride = 0;
        param.srcDValue = W;
        param.dstNzC0Stride = 16;
        param.dstNzNStride = 1;
        param.dstNzMatrixStride = 0;
        for (int i=0;i<h/16;++i){
            DataCopy(dst[i*16*Align16B(w)], src[i*16*W], param);
        }
    }
    if (h%16!=0){
        Nd2NzParams param;
        param.ndNum = 1;
        param.nValue = h%16;
        param.dValue = w;
        param.srcNdMatrixStride = 0;
        param.srcDValue = W;
        param.dstNzC0Stride = 16;
        param.dstNzNStride = 1;
        param.dstNzMatrixStride = 0;
        int tail_start = h/16*16;
        DataCopy(dst[tail_start*Align16B(w)], src[tail_start*W], param);
    }
}

template<typename T>
__aicore__ inline void GM2L1(LocalTensor<T> dst, GlobalTensor<T> src, int nBurst, int burstLen, int srcStride, int dstStride){
    DataCopyParams param;
    param.blockCount = nBurst;
    param.blockLen = burstLen;
    param.srcStride = srcStride;
    param.dstStride = dstStride;
    DataCopy(dst, src, param);
}

template<typename T>
__aicore__ inline void GM2L1PAD(LocalTensor<T> dst, GlobalTensor<T> src, int nBurst, int burstLenByte, int srcStrideByte, int dstStride){
    DataCopyExtParams param;
    param.blockCount = nBurst;
    param.blockLen = burstLenByte;
    param.srcStride = srcStrideByte;
    param.dstStride = dstStride;

    DataCopyPadExtParams<T> padparam;
    padparam.isPad = false;
    padparam.leftPadding = 0;
    padparam.rightPadding = 0;
    DataCopyPad<T, PaddingMode::Normal>(dst, src, param, padparam);
}


// 16x16 b16 block transpose (vnchwconv). Row strides in elements (each row
// base must be 32B aligned), rep strides in 32B blocks. repeat==1 forces both
// rep strides to 0 (HW treats them as a base offset on single repeats).
template<typename T>
__aicore__ inline void TRANSDATA5HD(LocalTensor<T> dst, LocalTensor<T> src,
        int repeat, int srcRowStride, int dstRowStride, int srcRepStride, int dstRepStride){
    uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE];
    for (int i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; ++i) {
        dstList[i] = (uint64_t)dst[i * dstRowStride].GetPhyAddr();
        srcList[i] = (uint64_t)src[i * srcRowStride].GetPhyAddr();
    }
    TransDataTo5HDParams p;
    p.dstHighHalf = false;
    p.srcHighHalf = false;
    p.repeatTimes = (uint8_t)repeat;
    p.dstRepStride = repeat == 1 ? 0 : (uint16_t)dstRepStride;
    p.srcRepStride = repeat == 1 ? 0 : (uint16_t)srcRepStride;
    TransDataTo5HD<T>(dstList, srcList, p);
}

template<typename T>
__aicore__ inline void LOADL0(LocalTensor<T> dst, LocalTensor<T> src, int m, int n){
#ifdef __DAV_C220_CUBE__
    int C0 = 32 / sizeof(T);
    LoadData2DParams param;
    param.repeatTimes = Align16B(m)*((n+C0-1)/C0*C0)*sizeof(T)/32/16;
    param.srcStride = 1;
    LoadData(dst, src, param);
#elif __DAV_C310__
    int C0 = 32 / sizeof(T);
    LoadData2DParamsV2 param;
    param.mStep = 1;
    param.kStep = Align16B(m)*((n+C0-1)/C0*C0)*sizeof(T)/32/16;
    param.srcStride = 1;
    param.dstStride = 1;
    LoadData(dst, src, param);
#endif 
}

template<typename TDst, typename TSrc, typename TMx>
__aicore__ inline void LOADL0_MX(
    LocalTensor<TDst> dst,
    LocalTensor<TSrc> src,
    LocalTensor<TMx> srcMx,
    int mStep,
    int kStep,
    int srcStride,
    int dstStride,
    bool ifTranspose,
    int mxXStep,
    int mxYStep,
    int mxSrcStride,
    int mxDstStride)
{
#ifdef __DAV_C310__
    LoadData2DParamsV2 param;
    param.mStep = mStep;
    param.kStep = kStep;
    param.srcStride = srcStride;
    param.dstStride = dstStride;
    param.ifTranspose = ifTranspose;

    LoadData2DMxParams mxParam;
    mxParam.xStartPosition = 0;
    mxParam.yStartPosition = 0;
    mxParam.xStep = mxXStep;
    mxParam.yStep = mxYStep;
    mxParam.srcStride = mxSrcStride;
    mxParam.dstStride = mxDstStride;

    auto srcMxE8m0 = srcMx.template ReinterpretCast<fp8_e8m0_t>();
    LoadData(dst, src, srcMxE8m0, param, mxParam);
#else
    static_assert(sizeof(TDst) == 0, "LOADL0_MX is only supported on a5/Atlas 350");
#endif
}

template<typename TDst, typename TSrc, typename TMx>
__aicore__ inline void L0NZ2NZ_MX(
    LocalTensor<TDst> dst,
    LocalTensor<TSrc> src,
    LocalTensor<TMx> srcMx,
    int mdst,
    int ndst,
    int msrc,
    int nsrc)
{
#ifdef __DAV_C310__
    int C0 = 32 / sizeof(TSrc);
    int mStep = CeilDiv(mdst, 16);
    int kStep = CeilDiv(ndst, C0);
    int dataSrcStride = CeilDiv(msrc, 16);
    int dataDstStride = CeilDiv(mdst, 16);
    int mxKStep = CeilDiv(ndst, C0 * 2);
    int mxSrcStride = CeilDiv(nsrc, C0 * 2);
    LOADL0_MX(
        dst, src, srcMx,
        mStep, kStep, dataSrcStride, dataDstStride,
        false,
        mStep, mxKStep, mxSrcStride, mxKStep);
#else
    static_assert(sizeof(TDst) == 0, "L0NZ2NZ_MX is only supported on a5/Atlas 350");
#endif
}

template<typename TDst, typename TSrc, typename TMx>
__aicore__ inline void L0NZ2ZN_MX(
    LocalTensor<TDst> dst,
    LocalTensor<TSrc> src,
    LocalTensor<TMx> srcMx,
    int mdst,
    int ndst,
    int msrc,
    int nsrc)
{
#ifdef __DAV_C310__
    int C0 = 32 / sizeof(TSrc);
    int mStep = CeilDiv(mdst, 16);
    int kStep = CeilDiv(ndst, C0);
    int dataSrcStride = CeilDiv(msrc, 16);
    int dataDstStride = CeilDiv(ndst, 16);
    int mxXStep = CeilDiv(ndst, 16);
    int mxKStep = CeilDiv(mdst, C0 * 2);
    int mxSrcStride = CeilDiv(msrc, C0 * 2);

    LOADL0_MX(
        dst, src, srcMx,
        mStep, kStep, dataSrcStride, dataDstStride,
        true,
        mxXStep, mxKStep, mxSrcStride, mxKStep);
#else
    static_assert(sizeof(TDst) == 0, "L0NZ2ZN_MX is only supported on a5/Atlas 350");
#endif
}


// load3d img2col window: L1 NC1HWC0 fmap -> L0A native hardware layout.
// a2/b* is validated as ZZ; a5/950 is still a real-HW probe path.
// Args follow the documented Img2colView semantics in doc/conv2d.md: k/m
// start+extent on
// the virtual [M, K]; padValue fixed 0; FMATRIX/PADDING rewritten per call.
template <typename T>
__aicore__ inline void L12L0A_IMG2COL(LocalTensor<T> dst, LocalTensor<T> src,
    int h, int w, int c, int kh, int kw,
    int padL, int padR, int padT, int padB,
    int strideH, int strideW, int dilationH, int dilationW,
    int k0, int m0, int kExt, int mExt){
#if defined(__DAV_C220_CUBE__)
    // a2/V220: load3d natively materializes the full [mExt, kExt] window in one call.
    LoadData3DParamsV2<T> p;
    p.padList[0] = (uint8_t)padL; p.padList[1] = (uint8_t)padR;
    p.padList[2] = (uint8_t)padT; p.padList[3] = (uint8_t)padB;
    p.l1H = (uint16_t)h; p.l1W = (uint16_t)w;
    p.channelSize = (uint16_t)c;
    p.kExtension = (uint16_t)kExt; p.mExtension = (uint16_t)mExt;
    p.kStartPt = (uint16_t)k0; p.mStartPt = (uint16_t)m0;
    p.strideW = (uint8_t)strideW; p.strideH = (uint8_t)strideH;
    p.filterW = (uint8_t)kw; p.filterH = (uint8_t)kh;
    p.dilationFilterW = (uint8_t)dilationW; p.dilationFilterH = (uint8_t)dilationH;
    p.enTranspose = false; p.enSmallK = false;
    p.padValue = (T)0;
    p.filterSizeW = false; p.filterSizeH = false; p.fMatrixCtrl = false;
    LoadData(dst, src, p);
#elif defined(__DAV_C310__)
    // a5/C310: load3d has NO K-direction destination stride (CANN: load3d 不支持配置
    // k_dst_stride). A single LoadData3D call materializes exactly ONE C0-wide K
    // fractal; passing kExt>C0 overwrites L0A fractal-0 so only the last C0 chunk
    // survives. So loop one C0 K-fractal per call, advancing the L0A dst by one
    // K-fractal (align16(mExt)*C0 elements) and kStartPt by C0 -- the same pattern
    // CANN's own arch35 conv kernels use. A following mmad(K=kExt) then reads the
    // whole NZ L0A. Validated bit-exact on Ascend950 (bf16 C0=16, fp32 C0=8;
    // multi-K-chunk, multi-M-fractal).
    int C0 = 32 / sizeof(T);
    int dstKFracStride = ((mExt + 15) / 16 * 16) * C0;
    int nChunk = (kExt + C0 - 1) / C0;
    LoadData3DParamsV2<T> p;
    p.padList[0] = (uint8_t)padL; p.padList[1] = (uint8_t)padR;
    p.padList[2] = (uint8_t)padT; p.padList[3] = (uint8_t)padB;
    p.l1H = (uint16_t)h; p.l1W = (uint16_t)w;
    p.channelSize = (uint16_t)c;
    p.kExtension = (uint16_t)C0; p.mExtension = (uint16_t)mExt;
    p.mStartPt = (uint16_t)m0;
    p.strideW = (uint8_t)strideW; p.strideH = (uint8_t)strideH;
    p.filterW = (uint8_t)kw; p.filterH = (uint8_t)kh;
    p.dilationFilterW = (uint8_t)dilationW; p.dilationFilterH = (uint8_t)dilationH;
    p.enTranspose = false; p.enSmallK = false;
    p.padValue = (T)0;
    p.filterSizeW = false; p.filterSizeH = false; p.fMatrixCtrl = false;
    for (int i = 0; i < nChunk; i++) {
        p.kStartPt = (uint16_t)(k0 + i * C0);
        LoadData(dst[i * dstKFracStride], src, p);
    }
#endif
}


template <typename T>
__aicore__ inline void L0NZ2ZZ(LocalTensor<T> dst, LocalTensor<T> src, int mdst, int ndst, int msrc, int nsrc){
#ifdef __DAV_C220_CUBE__
    LoadData2DParams param;
    int C0 = 32 / sizeof(T);
    param.repeatTimes = (ndst+C0-1)/(C0);
    param.srcStride = (msrc+15)/16;

    for (int i=0; i<(mdst+15)/16; ++i){
        LoadData(dst[16*i*((ndst+C0-1)/C0*C0)], src[i*16*C0], param);
    }
#elif __DAV_C310__

#endif
}


template<typename T> 
__aicore__ inline void L0NZ2ZN(LocalTensor<T> dst, LocalTensor<T> src, int mdst, int ndst, int msrc, int nsrc){
#ifdef __DAV_C220_CUBE__
    if constexpr(std::is_same<T, int8_t>::value){
        LoadData2dTransposeParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = (ndst + 31) / 32;
        loadDataParams.srcStride = (msrc + 31) / 32;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;
        for (int i=0; i<(mdst+31)/32; ++i){
            LoadDataWithTranspose(dst[i*32*Align32B(ndst)], src[i*32*32], loadDataParams);
        }
    }else{
        LoadData2DParams param;
        param.repeatTimes = (ndst+32/sizeof(T)-1)/(32/sizeof(T));
        param.srcStride = (msrc+15)/16;
        param.ifTranspose = true;

        int C0 = 32 / sizeof(T);

        for (int i=0; i<(mdst+15)/16; ++i){
            LoadData(dst[16*i*((ndst+C0-1)/C0*C0)], src[i*16*C0], param);
        }
    }
    
#elif __DAV_C310__
    int C0 = 32 / sizeof(T);
    LoadData2DParamsV2 param;
    param.mStep = (mdst+15)/16;
    param.kStep = (ndst+C0-1)/C0;
    param.srcStride = (msrc+15)/16;
    param.dstStride = (ndst+15)/16;
    param.ifTranspose = true;
    LoadData(dst, src, param);
#endif 
}

template<typename T> 
__aicore__ inline void L0NZ2NZ(LocalTensor<T> dst, LocalTensor<T> src, int mdst, int ndst, int msrc, int nsrc){
#ifdef __DAV_C220_CUBE__
    if (mdst==msrc){
        LOADL0(dst, src, mdst, ndst);
    }else{
        LoadData2DParams param;
        param.repeatTimes = (mdst+15)/16;
        param.srcStride = 1;
        int C0 = 32 / sizeof(T);
        for (int i=0; i<(ndst+C0-1)/C0; ++i){
            LoadData(dst[C0*i*((mdst+15)/16*16)], src[C0*i*((msrc+15)/16*16)], param);
        }
    }
#elif __DAV_C310__
    LoadData2DParamsV2 param;
    param.mStep = (mdst+15)/16;
    param.kStep = (ndst+32/sizeof(T)-1)/(32/sizeof(T));
    param.srcStride = (msrc+15)/16;
    param.dstStride = (mdst+15)/16;
    LoadData(dst, src, param);
#endif 
}


template<typename T> 
__aicore__ inline void L0NZ2NN(LocalTensor<T> dst, LocalTensor<T> src, int mdst, int ndst, int msrc, int nsrc){
#ifdef __DAV_C220_CUBE__
    if constexpr(std::is_same<T, int8_t>::value){
        LoadData2dTransposeParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = (mdst + 31) / 32;
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = (mdst + 31) / 32 - 1;
        for (int i=0; i<(ndst+31)/32; ++i){
            LoadDataWithTranspose(dst[i*32*Align32B(mdst)], src[i*32*Align32B(msrc)], loadDataParams);
        }
    }else{
        LoadData2DParams param;
        param.repeatTimes = (mdst+15)/16;
        param.srcStride = 1;
        param.ifTranspose = true;
        int C0 = 32 / sizeof(T);
        for (int i=0; i<(ndst+C0-1)/C0; ++i){
            LoadData(dst[C0*i*((mdst+15)/16*16)], src[C0*i*((msrc+15)/16*16)], param);
        }
    }
#elif __DAV_C310__

#endif 
}


__aicore__ inline void L0ZZ2ZZ(LocalTensor<float> dst, LocalTensor<float> src, int mdst, int ndst, int msrc, int nsrc){
#ifdef __DAV_C220_CUBE__
    if ((ndst == nsrc)&&(ndst%16>8)){
        LOADL0(dst, src, mdst, ndst);
        return;
    }

    LoadData2DParams param;
    param.repeatTimes = CeilDiv(ndst, 8);
    param.srcStride = 1;

    for (int i = 0; i < CeilDiv(mdst, 16); ++i){
        LoadData(
            dst[i * 16 * Align8B(ndst)],
            src[i * 16 * Align16B(nsrc)],
            param
        );
    }
#elif __DAV_C310__

#endif
}

__aicore__ inline void L0ZZ2NZ(LocalTensor<float> dst, LocalTensor<float> src, int mdst, int ndst, int msrc, int nsrc){
#ifdef __DAV_C220_CUBE__
    LoadData2DParams param;
    param.repeatTimes = CeilDiv(mdst, 16);
    param.srcStride = Align16B(nsrc) / 8;

    for (int i = 0; i < CeilDiv(ndst, 8); ++i){
        LoadData(
            dst[i * 8 * Align16B(mdst)],
            src[i * 8 * 16],
            param
        );
    }
#elif __DAV_C310__

#endif
}

template<typename T> 
__aicore__ inline void L0ZZ2NN(LocalTensor<T> dst, LocalTensor<T> src, int mdst, int ndst, int msrc, int nsrc){
#ifdef __DAV_C220_CUBE__
    LoadData2dTransposeParams loadDataParams;
    loadDataParams.startIndex = 0;
    loadDataParams.repeatTimes = (mdst + 15) / 16;
    loadDataParams.srcStride = (nsrc + 15) / 16;
    loadDataParams.dstGap = 1;
    loadDataParams.dstFracGap = 0;
    for (int i=0; i<(ndst+15)/16; ++i){
        LoadDataWithTranspose(dst[i*16*Align16B(mdst)], src[i*16*16], loadDataParams);
    }
#elif __DAV_C310__

#endif 
}

template<typename T> 
__aicore__ inline void L0ZZ2ZN(LocalTensor<T> dst, LocalTensor<T> src, int mdst, int ndst, int msrc, int nsrc){
#ifdef __DAV_C220_CUBE__
    LoadData2dTransposeParams loadDataParams;
    loadDataParams.startIndex = 0;
    loadDataParams.repeatTimes = CeilDiv(ndst,16);
    loadDataParams.srcStride = 1;
    loadDataParams.dstGap = 0;
    loadDataParams.dstFracGap = CeilDiv(ndst, 16) - 1;
    for (int i=0; i<(mdst+15)/16;++i){
        LoadDataWithTranspose(dst[i*16*Align16B(ndst)], src[i*16*Align16B(nsrc)], loadDataParams);
    }
#elif __DAV_C310__

#endif 
}


template <typename T>
__aicore__ inline void L1_TO_BT(LocalTensor<T> dstC2, LocalTensor<T> srcL1, int n){
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C310__)
    // Load a bias row from L1 into the C2 bias-table buffer. Bias is fp32 (float
    // inputs) or int32 (int8@int8); both are 4 bytes, so a C2 entry occupies 2 slots
    // each (L0C width) and burstLen is computed in 32B units over n*oneDataLen*2
    // bytes, rounded up to an even burst count (mirrors CANN LoadBias2C2). Burst-form
    // DataCopy on the MTE1 pipe; count-form would fill only n/2 logical columns.
    // Shared, byte-identical formula on a2 (910b / C220) and a5 (Atlas 350 / C310):
    // the C2 2-slot layout for 4-byte bias is the same on both.
    constexpr int oneDataLen = (sizeof(T) == 4) ? 2 : 1;
    uint16_t lenBurst = (uint16_t)(((n * oneDataLen * 2) + 31) / 32);
    if (sizeof(T) == 4) lenBurst = (uint16_t)(((lenBurst + 1) / 2) * 2);
    DataCopy(dstC2, srcL1, {(uint16_t)1, lenBurst, (uint16_t)0, (uint16_t)0});
#endif
    // No #else body: the kernel .cpp is compiled once per core type (AIC/AIV); on the
    // a2 vector (AIV) compile __DAV_C220_CUBE__ is undefined, so this template must
    // compile to an empty no-op (it is only ever called on the cube side), exactly
    // like LOADL0 and the other cube-only wrappers. The DSL-level assert_valid_device
    // gate already rejects unsupported devices.
}


template<typename T>
__aicore__ inline void GM2UB(LocalTensor<T> dst, GlobalTensor<T> src, int nBurst, int burstLen, int srcStride, int dstStride){
    DataCopyParams param;
    param.blockCount = nBurst;
    param.blockLen = burstLen;
    param.srcStride = srcStride;
    param.dstStride = dstStride;
    DataCopy(dst, src, param);
}

template<typename T>
__aicore__ inline void GM2UBPAD(LocalTensor<T> dst, GlobalTensor<T> src, int nBurst, int burstLenByte, int srcStrideByte, int dstStride){
    DataCopyExtParams param;
    param.blockCount = nBurst;
    param.blockLen = burstLenByte;
    param.srcStride = srcStrideByte;
    param.dstStride = dstStride;

    DataCopyPadExtParams<T> padparam;
    padparam.isPad = false;
    padparam.leftPadding = 0;
    padparam.rightPadding = 0;
    DataCopyPad(dst, src, param, padparam);
}

template<typename T, uint8_t dim, const NdDmaConfig& config = kDefaultNdDmaConfig>
__aicore__ inline void GM2UB_ND_DMA(LocalTensor<T> dst, GlobalTensor<T> src, const NdDmaLoopInfo<dim>& loopInfo, T constantValue){
#ifdef __DAV_C310__
    NdDmaParams<T, dim> params{loopInfo, constantValue};
    NdDmaDci();
    DataCopy<T, dim, config>(dst, src, params);
#else
    static_assert(!std::is_same_v<T, T>, "GM2UB_ND_DMA is only supported on A5 / C310");
#endif
}

template<typename T>
__aicore__ inline void UB2GM(GlobalTensor<T> dst, LocalTensor<T> src, int nBurst, int burstLen, int srcStride, int dstStride){
    DataCopyParams param;
    param.blockCount = nBurst;
    param.blockLen = burstLen;
    param.srcStride = srcStride;
    param.dstStride = dstStride;
    DataCopy(dst, src, param);
}

template<typename T>
__aicore__ inline void UB2GMPAD(GlobalTensor<T> dst, LocalTensor<T> src, int nBurst, int burstLenByte, int srcStride, int dstStrideByte){
    DataCopyExtParams param;
    param.blockCount = nBurst;
    param.blockLen = burstLenByte;
    param.srcStride = srcStride;
    param.dstStride = dstStrideByte;

    DataCopyPad(dst, src, param);
}

template<typename T>
__aicore__ inline void UB2UB(LocalTensor<T> dst, LocalTensor<T> src, int nBurst, int burstLen, int srcStride, int dstStride){
    DataCopyParams param;
    param.blockCount = nBurst;
    param.blockLen = burstLen;
    param.srcStride = srcStride;
    param.dstStride = dstStride;
    DataCopy(dst, src, param);
}

template<typename T>
__aicore__ inline void UB2L1(LocalTensor<T> dst, LocalTensor<T> src, int nBurst, int burstLen, int srcStride, int dstStride){
    DataCopyParams param;
    param.blockCount = nBurst;
    param.blockLen = burstLen;
    param.srcStride = srcStride;
    param.dstStride = dstStride;
    DataCopy(dst, src, param);
}

template<typename T>
__aicore__ inline void UB2L1_NZ(LocalTensor<T> dst, LocalTensor<T> src, int mdst, int ndst, int msrc, int nsrc, int Msrc = -1){
    // msrc = valid rows to copy (src view span); Msrc = physical fractal-row stride of the src
    // NZ tile (src shape). srcStride = Msrc - msrc skips the per-fractal-col padding rows (e.g.
    // an odd stride used to dodge UB bank conflict). Msrc < 0 -> compact (Msrc == msrc, srcStride 0,
    // identical to the old behaviour). NZ row = C0 elems = 32B = 1 datablock, so the diff is in rows.
    const int C0 = 32 / sizeof(T);
    if (Msrc < 0) Msrc = msrc;
    DataCopyParams param;
    param.blockCount = (nsrc + C0 - 1) / C0;
    param.blockLen = msrc;
    param.srcStride = Msrc - msrc;
    param.dstStride = (mdst + 15) / 16 * 16 - msrc;
    DataCopy(dst, src, param);
}

template<typename T>
__aicore__ inline void UB2L1_ND2NZ(LocalTensor<T> dst, LocalTensor<T> src, int mdst, int ndst, int msrc, int nsrc, int Nsrc){
    // default msrc <= mdst, nsrc <= Nsrc.  nsrc is the span of columns
    // being moved; Nsrc is the row stride of the underlying src tile.
    const int C0 = 32 / sizeof(T);
    DataCopyParams param;
    param.blockCount = msrc;
    param.blockLen = 1;
    param.srcStride = (Nsrc + C0 - 1) / C0 - 1;
    param.dstStride = 0;
    for (int i=0; i<(nsrc + C0 - 1) / C0; ++i){
        DataCopy(dst[i*C0*((mdst + 15) / 16 * 16)], src[i*C0], param);
    }
}

template<typename T>
__aicore__ inline void UB2UB_ND2NZ(LocalTensor<T> dst, LocalTensor<T> src, int mdst, int ndst, int msrc, int nsrc){
    const int C0 = 32 / sizeof(T);
    DataCopyParams param;
    param.blockCount = (nsrc + C0 - 1) / C0;
    param.blockLen = 1;
    param.srcStride = 0;
    param.dstStride = (mdst + 15) / 16 * 16 - 1;
    for (int i=0; i<msrc; ++i){
        DataCopy(dst[i*C0], src[i*nsrc], param);
    }
}

template<typename T>
__aicore__ inline void UB2UB_ND2NZ_COMPACT(LocalTensor<T> dst, LocalTensor<T> src, int m, int n){
    const int C0 = 32 / sizeof(T);
    DataCopyParams param;
    param.blockCount = (n + C0 - 1) / C0;
    param.blockLen = 1;
    param.srcStride = 0;
    param.dstStride = m - 1;
    for (int i=0; i<m; ++i){
        DataCopy(dst[i*C0], src[i*n], param);
    }
}


template <typename T, typename T2>
__aicore__ inline void L0C2L1(LocalTensor<T> dst, LocalTensor<T2> src, int m, int n, int dst_M, int nz_M, bool reluEn){
#ifdef __DAV_C220_CUBE__
    // Temporarily disable the C220 float->float L0C2L1 path.
    // Keep the old branch commented for reference until support is clarified.
    /*
    if constexpr(std::is_same<T, float>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_cbuf((__cbuf__ T*)dst.GetPhyAddr(), (__cc__ T2*)src.GetPhyAddr(), 0, n, m, dst_M*2, (nz_M+15)/16*16, 0, NoQuant, (uint8_t)reluEn, false, false);
    }else
    */
    if constexpr(std::is_same<T, half>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_cbuf((__cbuf__ T*)dst.GetPhyAddr(), (__cc__ T2*)src.GetPhyAddr(), 0, n, m, dst_M, (nz_M+15)/16*16, 0, F322F16, (uint8_t)reluEn, false, false);
    }else if constexpr(std::is_same<T, bfloat16_t>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_cbuf((__cbuf__ T*)dst.GetPhyAddr(), (__cc__ T2*)src.GetPhyAddr(), 0, n, m, dst_M, (nz_M+15)/16*16, 0, F322BF16, (uint8_t)reluEn, false, false);
    }else{
        copy_matrix_cc_to_cbuf((__cbuf__ T*)dst.GetPhyAddr(), (__cc__ T2*)src.GetPhyAddr(), 0, n, m, dst_M*sizeof(T2)/2, (nz_M+15)/16*16, 0, NoQuant, (uint8_t)reluEn, false, false);
    }
#elif __DAV_C310__
    // This one does not support relu yet. Need to implement in the future 
    QuantMode_t q;
    if constexpr(std::is_same<T, float>::value && std::is_same<T2, float>::value){
        q = NoQuant; 
    }else if constexpr(std::is_same<T, half>::value && std::is_same<T2, float>::value){
        q = F322F16;
    }else if constexpr(std::is_same<T, bfloat16_t>::value && std::is_same<T2, float>::value){
        q = F322BF16;
    }else{
        q = NoQuant;
    }
    if constexpr(std::is_same<T, float>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_cbuf((__cbuf__ float*) dst.GetPhyAddr(), (__cc__ float*) src.GetPhyAddr(), 0, n, m, (dst_M+15)/16*16*8, (nz_M+15)/16*16, 0, 0, 0, NoQuant,
                                (uint8_t)reluEn, true, false, 0, 0, false, false,
                                0, false, false, false, false, false, false);
    }
    else{
        copy_matrix_cc_to_cbuf((__cbuf__ T*) dst.GetPhyAddr(), (__cc__ T2*) src.GetPhyAddr(), 0, n, m, (dst_M+15)/16*16*16, (nz_M+15)/16*16, 0, 0, 0, q,
                                (uint8_t)reluEn, false, false, 0, 0, false, false,
                                0, false, false, false, false, false, false);
    }

#endif
}


// Pack an a2/V220 fixpipe scalar-quant deqScalar (validated on Ascend910B3):
//   [31:13] scale  -- float19 (top 19 bits of the fp32 scale; low 13 bits unused)
//   [45:37] offset -- signed int9, added after the int9 round
//   [46]    sign   -- set by the caller from the dst dtype, NOT here
__aicore__ inline uint64_t PackDeqScalar(float scale, int offset){
    union { float f; uint32_t u; } cvt; cvt.f = scale;
    uint64_t deq = (uint64_t)(cvt.u & 0xFFFFE000u);            // scale  -> [31:13]
    deq |= ((uint64_t)((uint32_t)offset & 0x1FFu)) << 37;      // offset -> [45:37]
    return deq;
}

// L0C -> GM with an optional scalar dequant/requant. The fixpipe quant MODE is
// inferred from the (src, dst) dtype pair, and for 8-bit output the signed/unsigned
// saturation is taken from the dst dtype (int8_t -> signed, uint8_t -> unsigned).
// scale/offset feed the deqScalar; they are ignored by the pure-cast modes.
// scaledQuant=true (set by codegen when l0c.requant() supplies a non-default scale/offset)
// switches the fp32->float/half/bf16 casts to their a5-only scaled QF322*_PRE variants.
// Resolve the fixpipe scalar quant MODE + packed deqScalar from the (dst T, L0C-src T2) dtype
// pair. Shared by the L0C->GM and L0C->UB stores so the two stay in lock-step. scaledQuant
// routes the fp32->float/half/bf16 casts to the a5-only scaled QF322*_PRE variants; hif8Hybrid
// picks QF322HIF8_PRE_HYBRID. scale/offset feed the deqScalar (ignored by the pure-cast modes).
template <typename T, typename T2>
__aicore__ inline void ResolveFixpipeQuant(QuantMode_t& q, uint64_t& deqScalar, float scale, int offset, bool scaledQuant, bool hif8Hybrid){
    deqScalar = 0;
    if constexpr(std::is_same<T, float>::value && std::is_same<T2, float>::value){
#ifdef __DAV_C310__
        if (scaledQuant){ q = QF322F32_PRE; deqScalar = PackDeqScalar(scale, 0); }  // a5: scaled fp32 L0C -> fp32
        else q = NoQuant;
#else
        q = NoQuant;
#endif
    }else if constexpr(std::is_same<T, half>::value && std::is_same<T2, float>::value){
        if (scaledQuant){ q = QF322F16_PRE; deqScalar = PackDeqScalar(scale, 0); }   // a5: scaled fp32 L0C -> fp16
        else q = F322F16;
    }else if constexpr(std::is_same<T, bfloat16_t>::value && std::is_same<T2, float>::value){
        if (scaledQuant){ q = QF322BF16_PRE; deqScalar = PackDeqScalar(scale, 0); }  // a5: scaled fp32 L0C -> bf16
        else q = F322BF16;
    }else if constexpr(std::is_same<T, bfloat16_t>::value && std::is_integral<T2>::value){
        q = QS322BF16_PRE; deqScalar = PackDeqScalar(scale, 0);           // a5: int32 L0C -> bf16 dequant
    }else if constexpr(std::is_same<T, half>::value && std::is_integral<T2>::value){
        q = DEQF16; deqScalar = PackDeqScalar(scale, 0);                  // int32 L0C -> fp16 dequant
    }else if constexpr(sizeof(T) == 1 && std::is_integral<T>::value && std::is_same<T2, float>::value){
        q = QF322B8_PRE; deqScalar = PackDeqScalar(scale, offset);        // fp32 L0C -> 8-bit
        if constexpr(std::is_signed<T>::value) deqScalar |= (1ULL << 46); // s8 dst -> signed saturation
    }else if constexpr(sizeof(T) == 1 && std::is_integral<T>::value && std::is_integral<T2>::value){
        q = REQ8; deqScalar = PackDeqScalar(scale, offset);              // int32 L0C -> 8-bit requant
        if constexpr(std::is_signed<T>::value) deqScalar |= (1ULL << 46); // s8 dst -> signed saturation
#ifdef __DAV_C310__
    }else if constexpr(std::is_same<T, float8_e4m3_t>::value && std::is_same<T2, float>::value){
        q = QF322FP8_PRE; deqScalar = PackDeqScalar(scale, 0);            // a5: fp32 L0C -> fp8 e4m3
    }else if constexpr(std::is_same<T, hifloat8_t>::value && std::is_same<T2, float>::value){
        q = hif8Hybrid ? QF322HIF8_PRE_HYBRID : QF322HIF8_PRE;            // a5: fp32 L0C -> hif8
        deqScalar = PackDeqScalar(scale, 0);                             // (hybrid vs half-to-away rounding)
#endif
    }else{
        q = NoQuant;
    }
}

template <typename T, typename T2>
__aicore__ inline void L0C2GM_NZ2ND(GlobalTensor<T> dst, LocalTensor<T2> src, int m, int n, int N, int nz_M, bool reluEn, float scale = 1.0f, int offset = 0, bool scaledQuant = false, bool hif8Hybrid = false){
    QuantMode_t q;
    uint64_t deqScalar;
    ResolveFixpipeQuant<T, T2>(q, deqScalar, scale, offset, scaledQuant, hif8Hybrid);
#ifdef __DAV_C220_CUBE__
    FixpipeParamsV220 fixpipeParams;
    fixpipeParams.nSize = n;
    fixpipeParams.mSize = m;
    fixpipeParams.srcStride = (nz_M+15)/16*16;
    fixpipeParams.dstStride = N;
    fixpipeParams.ndNum = 1;
    fixpipeParams.srcNdStride = 1;
    fixpipeParams.dstNdStride = 1;
    fixpipeParams.quantPre = q;
    fixpipeParams.deqScalar = deqScalar;
    fixpipeParams.reluEn = reluEn;
    Fixpipe(dst, src, fixpipeParams);
#elif __DAV_C310__
    // a5/C310 reuses the shared deqScalar packing (PackDeqScalar + sign bit, above the
    // #ifdef); this assumes the a5 bit-layout matches a2/V220 -- validated on Ascend950 cannsim.
    FixpipeParamsC310 fixpipeParams;
    fixpipeParams.nSize = n;
    fixpipeParams.mSize = m;
    fixpipeParams.srcStride = (nz_M+15)/16*16;
    fixpipeParams.dstStride = N;
    fixpipeParams.quantPre = q;
    fixpipeParams.deqScalar = deqScalar;
    fixpipeParams.reluEn = reluEn;
    Fixpipe(dst, src, fixpipeParams);
#endif
}


// L0C [m, n] tile -> GM kept NZ ([C1=n/16] strips of [mPad, 16]); dst points at the
// tile's row m0 inside strip 0, strips are mPad rows apart. One native NZ fixpipe:
// needs the explicit CFG_NZ template config (the default config is NZ2ND and reads
// dstStride as an ND row width); dstStride = strip-to-strip in 32B blocks (mPad*2 for
// fp32), the copy_matrix dst_M*2 convention. quantPre dispatch shared with NZ2ND
// (ResolveFixpipeQuant). 910B3-validated over a unit scan.
template <typename T, typename T2>
__aicore__ inline void L0C2GM_NZ2NZ(GlobalTensor<T> dst, LocalTensor<T2> src, int m, int n, int mPad, int nz_M, bool reluEn = false, float scale = 1.0f, int offset = 0, bool scaledQuant = false, bool hif8Hybrid = false){
    QuantMode_t q;
    uint64_t deqScalar;
    ResolveFixpipeQuant<T, T2>(q, deqScalar, scale, offset, scaledQuant, hif8Hybrid);
#ifdef __DAV_C220_CUBE__
    FixpipeParamsV220 p;
    p.nSize = n;
    p.mSize = m;
    p.srcStride = (nz_M+15)/16*16;
    p.dstStride = mPad * 16 * sizeof(T) / 32;
    p.quantPre = q;
    p.deqScalar = deqScalar;
    p.reluEn = reluEn;
    Fixpipe<T, T2, CFG_NZ>(dst, src, p);
#elif __DAV_C310__
    // CFG_NZ = {CO2Layout::NZ}, so the params must be templated to the same NZ
    // format (the default FixpipeParamsC310 is ROW_MAJOR and won't match Fixpipe<CFG_NZ>).
    FixpipeParamsC310<CO2Layout::NZ> p;
    p.nSize = n;
    p.mSize = m;
    p.srcStride = (nz_M+15)/16*16;
    // a5/C310 Fixpipe dstStride is in ELEMENTS (mirrors L0C2GM_NZ2ND's dstStride=N),
    // not a2/V220's 32B-block unit. The NZ strip-to-strip stride is mPad*C0 elements.
    p.dstStride = mPad * 16;
    p.quantPre = q;
    p.deqScalar = deqScalar;
    p.reluEn = reluEn;
    Fixpipe<T, T2, CFG_NZ>(dst, src, p);
#endif
}


// L0C [m, n] tile -> GM transposed to DN: the NZ2DN fixpipe (CO2Layout::COLUMN_MAJOR)
// writes the transpose, out[j, i] = L0C[i, j], so the dst is an [n, m] ND plane.
// dstStride = M_dst, the row width of that [n, m] dst (>= m), in ELEMENTS -- mirrors
// L0C2GM_NZ2ND's dstStride=N (the C310 element convention), not the a2/V220 32B-block
// unit. nSize/mSize still describe the L0C source tile (n cols, m rows); only the
// write pattern is transposed. quantPre dispatch shared with NZ2ND (ResolveFixpipeQuant).
// a5/C310 only -- validated on Ascend950 cannsim.
template <typename T, typename T2>
__aicore__ inline void L0C2GM_NZ2DN(GlobalTensor<T> dst, LocalTensor<T2> src, int m, int n, int M_dst, int nz_M, bool reluEn = false, float scale = 1.0f, int offset = 0, bool scaledQuant = false, bool hif8Hybrid = false){
    QuantMode_t q;
    uint64_t deqScalar;
    ResolveFixpipeQuant<T, T2>(q, deqScalar, scale, offset, scaledQuant, hif8Hybrid);
#ifdef __DAV_C310__
    // CFG_COLUMN_MAJOR = {CO2Layout::COLUMN_MAJOR}; the params must be templated to the
    // same DN format (the default FixpipeParamsC310 is ROW_MAJOR).
    FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> p;
    p.nSize = n;
    p.mSize = m;
    p.srcStride = (nz_M+15)/16*16;
    p.dstStride = M_dst;
    // Nz2DnParams{dnNum, srcNzMatrixStride, dstDnMatrixStride, srcNzC0Stride}. The
    // default ctor leaves srcNzC0Stride=0 (innermost/loop0 source stride), which pins
    // the source m-walk to row 0 -- every output column then gets src[0, n] (verified on
    // Ascend950 cannsim). The matmul adv_api (FixpipeParamsUtil, arch 3510) sets {1,0,0,1};
    // mirror it so the transpose actually advances the source.
    p.params = {1, 0, 0, 1};
    p.quantPre = q;
    p.deqScalar = deqScalar;
    p.reluEn = reluEn;
    Fixpipe<T, T2, CFG_COLUMN_MAJOR>(dst, src, p);
#else
    static_assert(sizeof(T) == 0, "L0C2GM_NZ2DN (NZ2DN fixpipe) is only supported on a5/C310");
#endif
}


// L0C -> UB with an optional scalar dequant/requant (a5/C310 only). Mirrors L0C2GM_NZ2ND's
// quant dispatch (shared ResolveFixpipeQuant). dualMode = 0(SINGLE)/1(SPLITM)/2(SPLITN); the
// requant/relu features are only valid in SINGLE (the stub gates this), so scale/offset/
// scaledQuant/hif8Hybrid keep their plain-cast defaults for the dual-lane stores.
template <typename T, typename T2>
__aicore__ inline void L0C2UB_NZ2ND(LocalTensor<T> dst, LocalTensor<T2> src, int m, int n, int N, int nz_M, int dualMode, bool subBlkId, bool reluEn = false, float scale = 1.0f, int offset = 0, bool scaledQuant = false, bool hif8Hybrid = false){
    QuantMode_t q;
    uint64_t deqScalar;
    ResolveFixpipeQuant<T, T2>(q, deqScalar, scale, offset, scaledQuant, hif8Hybrid);
#ifdef __DAV_C310__
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = n;
    fixpipeParams.mSize = m;
    fixpipeParams.srcStride = (nz_M+15)/16*16;
    fixpipeParams.dstStride = N;
    fixpipeParams.quantPre = q;
    fixpipeParams.deqScalar = deqScalar;
    fixpipeParams.dualDstCtl = dualMode;
    fixpipeParams.subBlockId = subBlkId;
    fixpipeParams.reluEn = reluEn;
    Fixpipe<T, T2, CFG_ROW_MAJOR_UB>(dst, src, fixpipeParams);
#endif
}


template <typename T1, typename T2, typename T3>
__aicore__ inline void MMAD(LocalTensor<T1> dst, LocalTensor<T2> src0, LocalTensor<T3> src1, uint16_t m, uint16_t k, uint16_t n, bool cmatrixInitVal){
    // static_assert(std::is_same<T2, T3>::value, "MMAD requires src0/src1 to have the same datatype");
    MmadParams param;
    param.m = (m+15)/16*16;
    param.n = n;
    param.k = k;
    param.cmatrixInitVal = cmatrixInitVal;
    // param.unitFlag = unitFlag;
    Mmad(dst, src0, src1, param);
} 

template <typename T1, typename T2, typename T3>
__aicore__ inline void MMAD_MX(LocalTensor<T1> dst, LocalTensor<T2> src0, LocalTensor<T3> src1, uint16_t m, uint16_t k, uint16_t n, bool cmatrixInitVal){
#ifdef __DAV_C310__
    static_assert(std::is_same<T1, float>::value, "MMAD_MX requires float L0C destination");
    static_assert(
        (
            (std::is_same<T2, mx_fp8_e4m3_t>::value || std::is_same<T2, mx_fp8_e5m2_t>::value) &&
            (std::is_same<T3, mx_fp8_e4m3_t>::value || std::is_same<T3, mx_fp8_e5m2_t>::value)
        ) ||
        (
            (std::is_same<T2, fp4x2_e2m1_t>::value || std::is_same<T2, fp4x2_e1m2_t>::value) &&
            (std::is_same<T3, fp4x2_e2m1_t>::value || std::is_same<T3, fp4x2_e1m2_t>::value)
        ),
        "MMAD_MX requires both sources to be MX FP8 or both sources to be FP4");
    MmadParams param;
    param.m = (m+15)/16*16;
    param.n = n;
    param.k = k;
    param.cmatrixInitVal = cmatrixInitVal;
    Mmad(dst, src0, src1, param);
#else
    static_assert(sizeof(T1) == 0, "MMAD_MX is only supported on a5/Atlas 350");
#endif
}

template <typename T1, typename T2, typename T3, typename TB>
__aicore__ inline void MMAD_MX_BIAS(LocalTensor<T1> dst, LocalTensor<T2> src0, LocalTensor<T3> src1,
                                    LocalTensor<TB> bias, uint16_t m, uint16_t k, uint16_t n){
#ifdef __DAV_C310__
    // MX (scale) matmul with a C2 bias table: C = bias + scale_a*A @ (scale_b*B) on the
    // is_init K-tile. The per-block MX scales are intrinsic to the MX operand types (set up
    // by the L0AMX/L0BMX buffers), while the bias rides the 4-arg Mmad's C2 (bias-table)
    // operand -- so this is MMAD_MX's type contract plus MMAD_BIAS's cmatrixInitVal=false.
    // a5/Atlas 350 (C310) only. Later accumulate tiles must use plain MMAD_MX.
    static_assert(std::is_same<T1, float>::value, "MMAD_MX_BIAS requires float L0C destination");
    static_assert(
        (
            (std::is_same<T2, mx_fp8_e4m3_t>::value || std::is_same<T2, mx_fp8_e5m2_t>::value) &&
            (std::is_same<T3, mx_fp8_e4m3_t>::value || std::is_same<T3, mx_fp8_e5m2_t>::value)
        ) ||
        (
            (std::is_same<T2, fp4x2_e2m1_t>::value || std::is_same<T2, fp4x2_e1m2_t>::value) &&
            (std::is_same<T3, fp4x2_e2m1_t>::value || std::is_same<T3, fp4x2_e1m2_t>::value)
        ),
        "MMAD_MX_BIAS requires both sources to be MX FP8 or both sources to be FP4");
    MmadParams param;
    param.m = (m+15)/16*16;
    param.n = n;
    param.k = k;
    param.cmatrixInitVal = false;
    Mmad(dst, src0, src1, bias, param);
#else
    static_assert(sizeof(T1) == 0, "MMAD_MX_BIAS is only supported on a5/Atlas 350");
#endif
}

template <typename T1, typename T2, typename T3, typename TB>
__aicore__ inline void MMAD_BIAS(LocalTensor<T1> dst, LocalTensor<T2> src0, LocalTensor<T3> src1,
                                 LocalTensor<TB> bias, uint16_t m, uint16_t k, uint16_t n){
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C310__)
    // C = bias + A@B on the is_init K-tile. cmatrixInitVal=false initializes C from
    // the bias table; the 4-arg Mmad auto-derives cmatrixSource=true from the bias's
    // C2 (bias-table) position. Later accumulate tiles must use plain MMAD. Supported
    // on a2 (910b / C220) and a5 (Atlas 350 / C310): bias is fp32 for float/half/bf16
    // inputs and int32 for int8@int8 (Mmad datatype tables 8 & 9).
    MmadParams param;
    param.m = (m+15)/16*16;
    param.n = n;
    param.k = k;
    param.cmatrixInitVal = false;
    Mmad(dst, src0, src1, bias, param);
#endif
    // No #else body (see L1_TO_BT): on the a2 AIV compile __DAV_C220_CUBE__ is
    // undefined, so this cube-only template must compile to an empty no-op.
}


template <typename T>
__aicore__ inline void MERGESORT4(LocalTensor<T> dst, LocalTensor<T> src){
    MrgSort4Info params;
    params.elementLengths[0] = 32;
    params.elementLengths[1] = 32;
    params.elementLengths[2] = 32;
    params.elementLengths[3] = 32;
    params.ifExhaustedSuspension = false;
    params.validBit = 0b1111;
    params.repeatTimes = 1;

    MrgSortSrcList<T> srcList;
    srcList.src1 = src[0];
    srcList.src2 = src[32 * 1 * 2 * 4 / sizeof(T)];
    srcList.src3 = src[32 * 2 * 2 * 4 / sizeof(T)];
    srcList.src4 = src[32 * 3 * 2 * 4 / sizeof(T)];

    MrgSort<T>(dst, srcList, params);
} 


template <typename T>
__aicore__ inline void MERGESORT4(LocalTensor<T> dst, LocalTensor<T> src, const uint16_t length_per_seq, const uint8_t repeat){
    MrgSort4Info params;
    params.elementLengths[0] = length_per_seq;
    params.elementLengths[1] = length_per_seq;
    params.elementLengths[2] = length_per_seq;
    params.elementLengths[3] = length_per_seq;
    params.ifExhaustedSuspension = false;
    params.validBit = 0b1111;
    params.repeatTimes = repeat;

    MrgSortSrcList<T> srcList;
    srcList.src1 = src[0];
    srcList.src2 = src[length_per_seq * 1 * 2 * 4 / sizeof(T)];
    srcList.src3 = src[length_per_seq * 2 * 2 * 4 / sizeof(T)];
    srcList.src4 = src[length_per_seq * 3 * 2 * 4 / sizeof(T)];

    MrgSort<T>(dst, srcList, params);
} 

template <typename T>
__aicore__ inline void MERGESORT2SEQ(LocalTensor<T> dst, LocalTensor<T> src1, LocalTensor<T> src2, const uint16_t size1, const uint16_t size2){
    MrgSort4Info params;
    params.elementLengths[0] = size1;
    params.elementLengths[1] = size2;
    params.elementLengths[2] = 0;
    params.elementLengths[3] = 0;
    params.ifExhaustedSuspension = false;
    params.validBit = 3;
    params.repeatTimes = 1;

    MrgSortSrcList<T> srcList;
    srcList.src1 = src1;
    srcList.src2 = src2;
    srcList.src3 = src2;
    srcList.src4 = src2;

    MrgSort<T>(dst, srcList, params);
}
