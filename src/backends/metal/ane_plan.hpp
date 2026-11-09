#pragma once
#include <algorithm>
#include <vector>

namespace tqllm::mlx_backend {
// Per-layer cost of one prefill layer on this machine, measured once at load (Qwen35::ane_calibrate, cached on disk):
// milliseconds, per row unless named fixed. The plan below picks, for any prompt length m, whether the FFN splits
// between the GPU and the Neural Engine and into how many slices, from these numbers alone (no length thresholds).
struct AneCost {
    double k1=0;      // GPU-only layer, one pass of up to group rows (as the engine runs it: dequantize per pass)
    double d1=0;      // fixed: the same pass
    double u=0;       // GPU mixer, non-split layer-major path (FP16 weights dequantized once per layer)
    double u_split=0; // GPU mixer while a split layer runs (staged 4-bit projections)
    double f=0;       // GPU FFN, all channels, non-split layer-major path
    double d=0;       // fixed: the layer-major path's FP16 dequantize of the layer (once per prefill)
    double s=0;       // GPU share of the split FFN
    double g0=0;      // fixed: GPU work of a split slice (launch / sync overhead of its mixer, share and pack)
    double p=0;       // pack of the ANE input, per padded row
    double a=0;       // ANE evaluation, per padded row
    double o=0;       // fixed: ANE evaluation per slice (submit, wait, join)
    double r=0;       // fixed: requantize of the layer's ANE weights (once per prefill)
    bool valid() const {return k1>0 && u>0 && u_split>0 && f>0 && s>0 && a>0;}
};

struct AnePlan {
    bool split=false;
    int slices=1,rows=0;  // the split's slices (balanced, rows each; the last may be shorter)
    bool prefetch=false;  // requantize the next layer's ANE weights while this layer runs (ANE-bound slices)
    double t_split=0,t_gpu=0;  // predicted ms per layer
};

// The ANE function sizes ascend (512..2048); fit = the smallest size with at least n rows, or -1.
inline int ane_fit_rows(const std::vector<int>& sizes,int n) {
    for(int v:sizes)if(v>=n)return v;
    return -1;
}

// Predicted per-layer time of a split into k balanced slices of r rows (fit = the ANE function's padded rows). One slice:
// mixer, pack, then the ANE share and the GPU share side by side. Two or more: the last slice of a layer joins in the
// next layer (handoff carry), so one slice's ANE share overlaps the next slice's GPU work and the layer costs the slower
// side per slice. The requantize is GPU work: it is free only when the GPU would otherwise wait for the ANE at least
// that long (then it is prefetched into that wait), else it adds to the layer.
struct AneSplitCost {double ms=0;bool prefetch=false;};
inline AneSplitCost ane_split_ms(const AneCost& c,int k,int r,int fit) {
    const double ane=fit*c.a+c.o;
    if(k==1) {
        const double share=r*c.s,idle=std::max(0.0,ane-share);
        const bool pf=idle>=c.r;
        return {c.g0+r*c.u_split+fit*c.p+std::max(share,ane)+(pf?0.0:c.r),pf};
    }
    const double gpu=c.g0+r*(c.u_split+c.s)+fit*c.p,idle=k*std::max(0.0,ane-gpu);
    const bool pf=idle>=c.r;
    return {k*std::max(gpu,ane)+(pf?0.0:c.r),pf};
}

// GPU-only time of a layer for m rows: below dequant_rows the 4-bit kernels run directly (no dequantize), up to
// group_rows one pass (k1, d1), above it layer-major passes sharing one dequantize per layer (u, f, d)
inline double ane_gpu_ms(const AneCost& c,int m,int dequant_rows,int group_rows) {
    if(m<dequant_rows)return m*(c.u+c.f);
    if(m<=group_rows)return c.d1+m*c.k1;
    return c.d+m*(c.u+c.f);
}
// m prompt rows; sizes the ANE function sizes; align the GPU kernels' row tile (slices are whole tiles but the last)
inline AnePlan ane_plan(const AneCost& c,int m,const std::vector<int>& sizes,int align=1,int dequant_rows=0,int group_rows=0) {
    AnePlan best;
    best.t_gpu=ane_gpu_ms(c,m,dequant_rows,group_rows);
    if(!c.valid() || sizes.empty() || m<1 || align<1)return best;
    double t=0;
    for(int k=std::max(1,(m+sizes.back()-1)/sizes.back());k<=m;++k) {
        const int r=((m+k-1)/k+align-1)/align*align,fit=ane_fit_rows(sizes,r);
        if(fit<0 || (k-1)*r>=m)continue;
        const auto sc=ane_split_ms(c,k,r,fit);
        if(!best.split || sc.ms<t) {t=sc.ms;best.split=true;best.slices=k;best.rows=r;best.prefetch=sc.prefetch;}
        if(r<=sizes.front())break;  // more slices only add padding and the fixed cost per slice
    }
    best.t_split=t;
    if(!best.split || t>=best.t_gpu) {best.split=false;best.slices=1;best.rows=0;best.prefetch=false;}
    return best;
}
}  // namespace tqllm::mlx_backend
