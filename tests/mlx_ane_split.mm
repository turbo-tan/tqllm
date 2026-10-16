// Native go/no-go for the prefill FFN split: one FFN layer (5120 -> 17408 -> 5120) at M rows. The GPU (MLX FP16 GEMM,
// as the engine's prefill FFN) computes the first F - C intermediate columns while Core ML computes the last C columns
// on the Neural Engine at the same time, reading the MLX input buffer and writing into an MLX-owned output buffer.
// Prints GPU-only, ANE-only and concurrent wall times. usage: test-mlx-ane-split <ffn_slice.mlmodelc> <M> <C>
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <mlx/mlx.h>
#include <chrono>
#include <cstdio>
#include <string>

namespace mx = mlx::core;
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {return std::chrono::duration<double, std::milli>(Clock::now() - t).count();}

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc != 4) {std::fprintf(stderr, "usage: test-mlx-ane-split model.mlmodelc M C\n"); return 2;}
        const int M = std::stoi(argv[2]), C = std::stoi(argv[3]), H = 5120, F = 17408, G = F - C;
        NSError* err = nil;
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        auto t_load = Clock::now();
        MLModel* model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
                                           configuration:cfg error:&err];
        if (!model) {std::fprintf(stderr, "LOAD_FAIL %s\n", err.localizedDescription.UTF8String); return 1;}
        std::printf("ANE_SPLIT load %.0f ms\n", ms_since(t_load));
        NSString* in_name = model.modelDescription.inputDescriptionsByName.allKeys.firstObject;
        NSString* out_name = model.modelDescription.outputDescriptionsByName.allKeys.firstObject;

        // GPU share: same math as the engine's prefill FFN (FP16 GEMMs, FP16 swiglu)
        auto x = mx::astype(mx::random::normal({M, H}), mx::float16);
        auto wg = mx::astype(mx::multiply(mx::random::normal({G, H}), mx::array(0.02f)), mx::float16);
        auto wu = mx::astype(mx::multiply(mx::random::normal({G, H}), mx::array(0.02f)), mx::float16);
        auto wd = mx::astype(mx::multiply(mx::random::normal({H, G}), mx::array(0.02f)), mx::float16);
        auto wgf = mx::astype(mx::multiply(mx::random::normal({F, H}), mx::array(0.02f)), mx::float16);
        auto wuf = mx::astype(mx::multiply(mx::random::normal({F, H}), mx::array(0.02f)), mx::float16);
        auto wdf = mx::astype(mx::multiply(mx::random::normal({H, F}), mx::array(0.02f)), mx::float16);
        auto y_ane = mx::zeros({M, H}, mx::float16);
        mx::eval(x, wg, wu, wd, wgf, wuf, wdf, y_ane);
        auto gpu_ffn = [&](const mx::array& a, const mx::array& b, const mx::array& d) {
            auto g = mx::matmul(x, mx::transpose(a));
            auto h = mx::multiply(mx::multiply(g, mx::sigmoid(g)), mx::matmul(x, mx::transpose(b)));
            return mx::matmul(h, mx::transpose(d));
        };

        // zero-copy views of the MLX buffers (unified memory)
        NSArray* shape = @[@(M), @(H)], *strides = @[@(H), @1];
        MLMultiArray* xin = [[MLMultiArray alloc] initWithDataPointer:(void*)x.data<mx::float16_t>() shape:shape
                                                             dataType:MLMultiArrayDataTypeFloat16 strides:strides deallocator:nil error:&err];
        MLMultiArray* yout = [[MLMultiArray alloc] initWithDataPointer:(void*)y_ane.data<mx::float16_t>() shape:shape
                                                              dataType:MLMultiArrayDataTypeFloat16 strides:strides deallocator:nil error:&err];
        MLDictionaryFeatureProvider* features = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{in_name: [MLFeatureValue featureValueWithMultiArray:xin]} error:&err];
        MLPredictionOptions* opts = [MLPredictionOptions new];
        opts.outputBackings = @{out_name: yout};
        auto ane = [&]() {
            NSError* e = nil;
            id<MLFeatureProvider> r = [model predictionFromFeatures:features options:opts error:&e];
            if (!r) std::fprintf(stderr, "PREDICT_FAIL %s\n", e.localizedDescription.UTF8String);
        };
        auto best = [](auto f) {f(); double b = 1e30; for (int i = 0; i < 7; ++i) {auto t = Clock::now(); f(); b = std::min(b, ms_since(t));} return b;};

        const double t_full = best([&] {mx::eval(gpu_ffn(wgf, wuf, wdf));});
        const double t_gpu = best([&] {mx::eval(gpu_ffn(wg, wu, wd));});
        const double t_ane = best(ane);
        const double t_both = best([&] {
            auto g = gpu_ffn(wg, wu, wd);
            mx::async_eval(g);   // GPU share starts now
            ane();               // ANE share runs meanwhile on this thread
            mx::eval(mx::add(g, y_ane));
        });
        const double flop = 2.0 * M * 3.0 * H * F;
        std::printf("ANE_SPLIT M=%d C=%d gpu_full %.2f ms (%.1f TFLOPS) gpu_share %.2f ms ane_share %.2f ms (%.1f TFLOPS) concurrent %.2f ms -> %.2fx\n",
                    M, C, t_full, flop / t_full * 1e-9, t_gpu, t_ane, 2.0 * M * 3.0 * H * C / t_ane * 1e-9, t_both, t_full / t_both);
        std::puts("ANE_SPLIT_DONE");
    }
    return 0;
}
