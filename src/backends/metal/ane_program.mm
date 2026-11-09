// TQLLM's own private-API W8A8 Neural Engine FFN program. The private AppleNeuralEngine client flow (protocol
// signatures + type-encoding check, dlopen, compile/load/evaluate on a serial queue, the FNV-named compile cache, the
// IOSurface L008/L00h surfaces, the BLOBFILE constant blob, the MIL bufferType/strides), the MIL function body
// (int8 weights dequantized by 1/ANE_FFN_INT8_UNIT, per-token int8 input, tanh-based SiLU, the intermediate's Hadamard
// rotation as a grouped 1x1 conv + per-token int8 requantize, the FP32 join via an appended per-token scale row) are
// borrowed from Splash (Apache-2.0): runtime/ane/Program.mm and runtime/ops/AneFfn.cpp. Simplified to the one FFN shape
// (single matmul segment) and driven from the CPU like AneFfn::predict: the CPU packs the input and joins the output,
// and the evaluation is made blocking with its own Metal shared event (the GPU-side pack/join handoff is step 2).

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include "ane_program.hpp"
#include "nax_affine.hpp"

#include <mlx/mlx.h>
#include <mlx/version.h>
#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <mlx/event.h>
#include <mlx/backend/metal/event.h>

#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <objc/runtime.h>
#include <sys/qos.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <thread>
#include <utility>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// ---- ABI constants (Splash metal/abi/AneFfn.h) ----
// int8 values span +-127; the ANE dequantizes them by 1/128 so fp16 cannot overflow, which every scale carries back.
static constexpr float kInt8Peak = 127.0f;
static constexpr float kInt8Unit = 128.0f;
// The Hadamard rotation of the intermediate runs in blocks of 512; a block of n values takes the first n Hadamard rows.
static constexpr uint32_t kIntermediateBlock = 512u;
// The least peak the ANE quantizes a token's rotated intermediate row against (Splash kIntermediateFloor).
static constexpr double kIntermediateFloor = 0x1p-9;
// The input / gate / up Hadamard rotation runs in blocks of 128 (Splash ANE_FFN_INPUT_BLOCK == ANE_FFN_ROTATION_UNIT):
// v -> diag(sign) H v / sqrt(128) over each 128-channel block of the hidden input, before the per-token int8. Qwen3's
// outlier channels are the massive-activation channels of the FFN input; the rotation spreads each block's outlier
// across its 128 channels so one per-token scale no longer loses the rest of the row. Gate/up weight columns (the
// hidden K) are rotated the same way, so (R x)(R W)^T = x W^T is unchanged.
static constexpr uint32_t kInputBlock = 128u;

// =====================================================================================================================
// The private AppleNeuralEngine interface (borrowed from Splash runtime/ane/Program.mm).
// =====================================================================================================================
@protocol TqAneModel
+ (id)modelAtURL:(NSURL*)url key:(NSString*)key;
- (NSDictionary*)modelAttributes;
@end
@protocol TqAneClient
+ (id)sharedConnection;
- (BOOL)compileModel:(id)model options:(NSDictionary*)options qos:(unsigned)qos error:(NSError**)error;
- (BOOL)compiledModelExistsFor:(id)model;
- (void)purgeCompiledModel:(id)model;
- (BOOL)loadModel:(id)model options:(NSDictionary*)options qos:(unsigned)qos error:(NSError**)error;
- (BOOL)unloadModel:(id)model options:(NSDictionary*)options qos:(unsigned)qos error:(NSError**)error;
- (BOOL)evaluateWithModel:(id)model options:(NSDictionary*)options request:(id)request qos:(unsigned)qos
                    error:(NSError**)error;
@end
@protocol TqAneSurface
+ (id)objectWithIOSurface:(IOSurfaceRef)surface;
@end
@protocol TqAneRequest
+ (id)requestWithInputs:(NSArray*)inputs inputIndices:(NSArray*)inputIndices outputs:(NSArray*)outputs
          outputIndices:(NSArray*)outputIndices weightsBuffer:(id)weights perfStats:(id)stats
         procedureIndex:(NSNumber*)procedure sharedEvents:(id)events transactionHandle:(NSNumber*)transaction;
- (void)setCompletionHandler:(void (^)(BOOL success, NSError* error))handler;
@end
@protocol TqAneEvents
+ (id)waitEventWithValue:(uint64_t)value sharedEvent:(id)event eventType:(uint64_t)type;
+ (id)signalEventWithValue:(uint64_t)value symbolIndex:(unsigned)symbol eventType:(int64_t)type sharedEvent:(id)event;
+ (id)sharedEventsWithSignalEvents:(NSArray*)signals waitEvents:(NSArray*)waits;
@end

namespace tqllm::mlx_backend {
namespace {

using Clock = std::chrono::steady_clock;

// Each private method, as the framework must define it on arm64 (BOOL is B), without the frame offsets a type
// encoding carries. A method whose stripped encoding differs takes or returns other types than declared.
struct Signature final {
    const char* owner;
    char kind;
    const char* selector;
    const char* types;
};
constexpr Signature kSignatures[] = {
    {"_ANEModel", '+', "modelAtURL:key:", "@@:@@"},
    {"_ANEModel", '-', "modelAttributes", "@@:"},
    {"_ANEClient", '+', "sharedConnection", "@@:"},
    {"_ANEClient", '-', "compileModel:options:qos:error:", "B@:@@I^@"},
    {"_ANEClient", '-', "compiledModelExistsFor:", "B@:@"},
    {"_ANEClient", '-', "purgeCompiledModel:", "v@:@"},
    {"_ANEClient", '-', "loadModel:options:qos:error:", "B@:@@I^@"},
    {"_ANEClient", '-', "unloadModel:options:qos:error:", "B@:@@I^@"},
    {"_ANEClient", '-', "evaluateWithModel:options:request:qos:error:", "B@:@@@I^@"},
    {"_ANEIOSurfaceObject", '+', "objectWithIOSurface:", "@@:^{__IOSurface=}"},
    {"_ANERequest", '+',
     "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:"
     "transactionHandle:",
     "@@:@@@@@@@@@"},
    {"_ANERequest", '-', "setCompletionHandler:", "v@:@?"},
    {"_ANESharedWaitEvent", '+', "waitEventWithValue:sharedEvent:eventType:", "@@:Q@Q"},
    {"_ANESharedSignalEvent", '+', "signalEventWithValue:symbolIndex:eventType:sharedEvent:", "@@:QIq@"},
    {"_ANESharedEvents", '+', "sharedEventsWithSignalEvents:waitEvents:", "@@:@@"},
};

constexpr unsigned kQos = QOS_CLASS_DEFAULT;

std::string methodName(const Signature& s) { return std::string(1, s.kind) + "[" + s.owner + " " + s.selector + "]"; }

// Checks that the framework has each method with its declared type encoding. Throws at the first that is missing or
// whose encoding changed.
void check(std::span<const Signature> signatures) {
    for (const Signature& s : signatures) {
        Class owner = NSClassFromString(@(s.owner));
        if (!owner) throw std::runtime_error(std::string("AppleNeuralEngine lacks ") + s.owner);
        const SEL selector = sel_registerName(s.selector);
        Method method = s.kind == '+' ? class_getClassMethod(owner, selector) : class_getInstanceMethod(owner, selector);
        if (!method) throw std::runtime_error("AppleNeuralEngine lacks " + methodName(s));
        const char* encoding = method_getTypeEncoding(method);
        std::string types = encoding ? encoding : "";
        std::erase_if(types, [](char c) { return c >= '0' && c <= '9'; });
        if (types != s.types) throw std::runtime_error("AppleNeuralEngine changed " + methodName(s));
    }
}

// The resolved private interface, the shared client and the serial queue the service's work runs on.
struct Api final {
    __strong id<TqAneClient> client = nil;
    Class<TqAneModel> model = nil;
    Class<TqAneSurface> surface = nil;
    Class<TqAneEvents> events = nil, signalEvent = nil, waitEvent = nil;
    Class<TqAneRequest> request = nil;
    __strong dispatch_queue_t queue = nil;
};

Api resolve() {
    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
        const char* reason = dlerror();
        throw std::runtime_error(std::string("AppleNeuralEngine does not load: ") + (reason ? reason : "no reason"));
    }
    check(kSignatures);
    const auto named = [](const char* name) { return NSClassFromString(@(name)); };
    Api api;
    api.model = named("_ANEModel");
    api.surface = named("_ANEIOSurfaceObject");
    api.events = named("_ANESharedEvents");
    api.signalEvent = named("_ANESharedSignalEvent");
    api.waitEvent = named("_ANESharedWaitEvent");
    api.request = named("_ANERequest");
    Class client = named("_ANEClient");
    id shared = [(Class<TqAneClient>)client sharedConnection];
    if (![shared isKindOfClass:client]) throw std::runtime_error("AppleNeuralEngine lacks a shared connection");
    api.client = shared;
    api.queue = dispatch_queue_create("tqllm.ane", DISPATCH_QUEUE_SERIAL);
    return api;
}

// The interface, resolved once. A failed resolution is remembered and rethrown on every use.
const Api& api() {
    static Api* resolved = nullptr;
    static std::string failure;
    static std::once_flag once;
    std::call_once(once, [] {
        @autoreleasepool {
            try {
                resolved = new Api(resolve());
            } catch (const std::exception& e) {
                failure = e.what();
            }
        }
    });
    if (!resolved) throw std::runtime_error(failure);
    return *resolved;
}

[[noreturn]] void fail(const char* what, NSError* error) {
    throw std::runtime_error(std::string("ANE ") + what + " failed" +
                             (error ? std::string(": ") + error.description.UTF8String : std::string()));
}

// Runs `body` on the service's serial queue and blocks until it finishes; rethrows what it threw.
void runOnQueue(const Api& api, std::function<void()> body) {
    struct Outcome {
        std::mutex mutex;
        std::condition_variable changed;
        bool finished = false;
        std::exception_ptr failure;
    };
    auto outcome = std::make_shared<Outcome>();
    dispatch_async(api.queue, ^{
        std::exception_ptr failure;
        @autoreleasepool {
            try {
                body();
            } catch (...) {
                failure = std::current_exception();
            }
        }
        std::lock_guard lock(outcome->mutex);
        outcome->finished = true;
        outcome->failure = failure;
        outcome->changed.notify_all();
    });
    std::unique_lock lock(outcome->mutex);
    outcome->changed.wait(lock, [&] { return outcome->finished; });
    if (outcome->failure) std::rethrow_exception(outcome->failure);
}

// ---- the per-user compile cache (Splash cacheDirectory) ----
std::filesystem::path cacheDirectory() {
    char path[PATH_MAX];
    const size_t size = confstr(_CS_DARWIN_USER_CACHE_DIR, path, sizeof path);
    if (!size || size > sizeof path) throw std::runtime_error("macOS names no per-user cache directory");
    return std::filesystem::path(path) / "tqllm-ane";
}

constexpr uint64_t kFnvBasis = 14695981039346656037ULL;
uint64_t fnv1a(std::span<const uint8_t> bytes, uint64_t hash = kFnvBasis) {
    for (const uint8_t b : bytes) hash = (hash ^ b) * 1099511628211ULL;
    return hash;
}
std::string hex(uint64_t v) {
    char t[17];
    std::snprintf(t, sizeof t, "%016llx", static_cast<unsigned long long>(v));
    return t;
}
std::span<const uint8_t> bytesOf(std::string_view t) {
    return {reinterpret_cast<const uint8_t*>(t.data()), t.size()};
}
void writeWhole(const std::filesystem::path& file, std::span<const uint8_t> bytes) {
    const std::filesystem::path partial = file.string() + "." + std::to_string(getpid()) + ".partial";
    {
        std::ofstream out(partial, std::ios::binary);
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        out.close();
        if (!out) throw std::runtime_error("unable to write " + file.string());
    }
    std::filesystem::rename(partial, file);
}
bool holds(const std::filesystem::path& file, std::span<const uint8_t> bytes) {
    std::ifstream in(file, std::ios::binary | std::ios::ate);
    if (!in || static_cast<uint64_t>(in.tellg()) != bytes.size()) return false;
    std::vector<uint8_t> contents(bytes.size());
    in.seekg(0);
    return in.read(reinterpret_cast<char*>(contents.data()), static_cast<std::streamsize>(contents.size())) &&
           std::memcmp(contents.data(), bytes.data(), bytes.size()) == 0;
}

// =====================================================================================================================
// IOSurface + Metal shared memory (borrowed from Splash ane::Surface, with our own Metal buffer wrap).
// =====================================================================================================================
enum class Element : uint8_t { Float16, Int8 };
uint32_t elementBytes(Element e) noexcept { return e == Element::Int8 ? 1 : 2; }
const char* milType(Element e) noexcept { return e == Element::Int8 ? "int8" : "fp16"; }
uint32_t rowBytes(uint32_t width, Element e) noexcept { return (width * elementBytes(e) + 63) / 64 * 64; }
uint64_t surfaceBytes(uint32_t rows, uint32_t width, Element e) noexcept {
    const uint64_t raw = uint64_t{rowBytes(width, e)} * rows;
    return (raw + 16383) / 16384 * 16384;  // page-aligned, matches Metal allocation granularity
}

// An IOSurface of `rows` rows of `width` fp16/int8 elements, each row padded to 64 bytes, and a Metal buffer over it
// (shared with the GPU and the ANE without a copy).
struct Surface final {
    std::shared_ptr<void> surface;  // IOSurfaceRef
    id<MTLBuffer> buffer = nil;
    Element element = Element::Float16;
    uint32_t strideBytes = 0;
    uint32_t rows = 0, width = 0;

    void* contents() const { return [buffer contents]; }

    std::string strides(uint32_t r) const {
        const uint64_t stride = strideBytes / elementBytes(element), plane = uint64_t{r} * stride;
        return "[" + std::to_string(plane) + ", " + std::to_string(plane) + ", " + std::to_string(stride) + ", 1]";
    }
    std::string bufferType(uint32_t r, uint32_t w) const {
        return std::string("tensor_buffer<") + milType(element) + ", shape=[1, 1, " + std::to_string(r) + ", " +
               std::to_string(w) + "], strides=" + strides(r) + ", interleave_factors=[1, 1, 1, 1]>";
    }
};

Surface createSurface(id<MTLDevice> device, uint32_t rows, uint32_t width, Element element) {
    const uint32_t stride = rowBytes(width, element);
    const uint64_t size = surfaceBytes(rows, width, element);
    NSDictionary* properties = @{
        (id)kIOSurfaceWidth : @(width),
        (id)kIOSurfaceHeight : @(rows),
        (id)kIOSurfaceBytesPerElement : @(elementBytes(element)),
        (id)kIOSurfaceBytesPerRow : @(stride),
        (id)kIOSurfaceAllocSize : @(size),
        (id)kIOSurfacePixelFormat : @(element == Element::Int8 ? 0x4c303038 : 0x4c303068),  // 'L008', 'L00h'
    };
    IOSurfaceRef created = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
    if (!created) throw std::runtime_error("IOSurface creation failed");
    std::shared_ptr<void> owner(created, [](void* s) { CFRelease(s); });
    if (IOSurfaceGetBytesPerRow(created) != stride)
        throw std::runtime_error("IOSurface took another row stride");
    const uint64_t allocated = IOSurfaceGetAllocSize(created);
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(created)
                                                     length:allocated
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
    if (!buffer) throw std::runtime_error("Metal buffer over IOSurface failed");
    Surface result;
    result.surface = std::move(owner);
    result.buffer = buffer;
    result.element = element;
    result.strideBytes = stride;
    result.rows = rows;
    result.width = width;
    return result;
}

// The BLOBFILE constant the rotation names (Splash ane::constantBlob): one fp16 record at offset kConstantOffset.
constexpr uint64_t kConstantOffset = 64;
std::vector<uint8_t> constantBlob(std::span<const _Float16> values) {
    constexpr uint64_t kData = 128;
    std::vector<uint8_t> blob(kData + values.size_bytes());
    const auto put = [&](uint64_t offset, auto value) { std::memcpy(blob.data() + offset, &value, sizeof value); };
    put(0, uint32_t{1});
    put(4, uint32_t{2});
    put(kConstantOffset, uint32_t{0xdeadbeef});
    put(kConstantOffset + 4, uint32_t{1});  // fp16
    put(kConstantOffset + 8, uint64_t{values.size_bytes()});
    put(kConstantOffset + 16, kData);
    std::memcpy(blob.data() + kData, values.data(), values.size_bytes());
    return blob;
}

// =====================================================================================================================
// The Hadamard rotation of the intermediate (Splash rotationSigns / rotationBlob).
// =====================================================================================================================
// TQ4's stored-basis constants (tq4.cpp): the signs applied BEFORE the 128-point Hadamard (bit set = -1) and the codebook.
constexpr uint32_t kTq4SignWords[4] = {0xE46A0359u, 0x42F85949u, 0xA4C63BBBu, 0x49F250A9u};
const char* kTq4Header = "constant half TQ4_LEVEL[16] = {6.0h, 18.0h, 31.0h, 44.0h, 58.0h, 75.0h, 96.0h, 127.0h,\n"
                         "                                -6.0h, -18.0h, -31.0h, -44.0h, -58.0h, -75.0h, -96.0h, -127.0h};\n";
std::array<float, kIntermediateBlock> rotationSigns() {
    std::mt19937 generator(20260930);
    std::array<float, kIntermediateBlock> signs{};
    for (float& s : signs) s = (generator() & 1) ? -1.0f : 1.0f;
    return signs;
}
// The rotation of each of the C intermediate channels as the [C, 512, 1, 1] weight of a grouped 1x1 convolution.
std::vector<uint8_t> rotationBlob(uint32_t channels, const std::array<float, kIntermediateBlock>& signs) {
    std::vector<_Float16> values(uint64_t{channels} * kIntermediateBlock);
    const float norm = 1.0f / std::sqrt(float(kIntermediateBlock));
    for (uint32_t channel = 0; channel < channels; ++channel)
        for (uint32_t input = 0; input < kIntermediateBlock; ++input) {
            const uint32_t output = channel % kIntermediateBlock;
            const float hadamard = (std::popcount(output & input) & 1) ? -1.0f : 1.0f;
            values[uint64_t{channel} * kIntermediateBlock + input] = _Float16(signs[output] * hadamard * norm);
        }
    return constantBlob(values);
}

// =====================================================================================================================
// The MIL program text (Splash AneFfn::function, one function, one matmul segment).
// =====================================================================================================================
std::string dimensions(uint64_t rows, uint64_t width) {
    return "[1, 1, " + std::to_string(rows) + ", " + std::to_string(width) + "]";
}
std::string tensor(const char* type, uint64_t rows, uint64_t width) {
    return std::string("tensor<") + type + ", " + dimensions(rows, width) + ">";
}
std::string fp16lit(double value) {
    char text[32];
    std::snprintf(text, sizeof text, "fp16(%a)", value);
    return text;
}
std::string functionName(uint32_t rows) { return "ffn" + std::to_string(rows); }
// The row counts the program compiles a function for: 512 up to the program's rows in steps of 256 (Splash compiles one
// per 128 rows from 512; coarser here to bound the cold compile). A slice of n tokens runs the smallest function >= n, so
// the ANE evaluates at most 255 padding rows instead of always the maximum.
std::vector<uint32_t> rowVariants(uint32_t rows) {
    std::vector<uint32_t> v;
    for (uint32_t r = 512; r < rows; r += 256) v.push_back(r);
    v.push_back(rows);
    return v;
}

// Splash runs the matmuls in segments of at most this many channels (two 2560-channel matmuls run a 5120 FFN about
// 14% faster than one 5120 on the ANE). Each segment's weights are a whole separate input tensor: the ANE compiler
// rejects a matmul whose weight is a slice of a larger dequantized tensor, but accepts slicing an activation.
constexpr uint32_t kSegment = 2560;

// (begin, width) segments of `total`, segment size kSegment (the last is the remainder).
std::vector<std::pair<uint32_t, uint32_t>> segmentsOf(uint32_t total) {
    std::vector<std::pair<uint32_t, uint32_t>> segs;
    for (uint32_t begin = 0; begin < total; begin += kSegment)
        segs.emplace_back(begin, std::min(kSegment, total - begin));
    return segs;
}

// The surfaces a function binds: x per hidden-segment, tx; gate/up weights per hidden-segment and their scales;
// down weights per C-segment and their scales; the output.
struct ProgramSurfaces final {
    std::vector<Surface> x;       // per hidden-segment [segWidth, rows] int8
    Surface tx;                   // [1, rows] fp16
    std::vector<Surface> wg, wu;  // per hidden-segment [C, segWidth] int8
    Surface sg, su;               // [C, 1] fp16
    std::vector<Surface> wd;      // per C-segment [hidden, segWidth] int8
    Surface sd;                   // [hidden, 1] fp16
    Surface y;                    // [hidden+1, rows] fp16
};

// The ANE's W8A8 FFN over `rows` rows, `channels` C intermediate, `hidden` inputs, matching Splash's segmented MIL.
// Every int8 is dequantized by 1/128; the per-row weight scales and per-token input scale carry it back. gate/up run
// one matmul per hidden-segment (whole weight tensors wg<k>/wu<k> over input segments x<k>), summed; SiLU is tanh-based
// with the token scale restored before the nonlinearity; the intermediate is Hadamard-rotated (grouped conv) and
// requantized per token; down runs one matmul per C-segment (whole weight tensors wd<i> over slices hd<i> of the
// requantized intermediate), summed. The output holds down's rows and, in a last row, each token's intermediate scale,
// which the CPU join multiplies with the token's input scale in fp32.
std::string buildFunction(uint32_t rows, uint32_t channels, uint32_t hidden, const ProgramSurfaces& s) {
    const std::string unit = fp16lit(1.0 / kInt8Unit);
    const std::string c = std::to_string(channels), r = std::to_string(rows),
                      block = std::to_string(kIntermediateBlock);
    const auto inSegs = segmentsOf(hidden), downSegs = segmentsOf(channels);
    std::string body;
    const auto line = [&](const std::string& t) { body += "        " + t + ";\n"; };
    const auto f16 = [&](const std::string& name, uint64_t height, uint64_t width, const std::string& expr) {
        line(tensor("fp16", height, width) + " " + name + " = " + expr);
    };
    // parameters, in a fixed order; load each tensor_buffer into a tensor.
    struct Param { std::string name; const Surface* s; uint32_t prows, pwidth; };
    std::vector<Param> params;
    for (size_t k = 0; k < inSegs.size(); ++k)
        params.push_back({"x" + std::to_string(k), &s.x[k], inSegs[k].second, rows});
    params.push_back({"tx", &s.tx, 1, rows});
    for (size_t k = 0; k < inSegs.size(); ++k) {
        params.push_back({"wg" + std::to_string(k), &s.wg[k], channels, inSegs[k].second});
        params.push_back({"wu" + std::to_string(k), &s.wu[k], channels, inSegs[k].second});
    }
    params.push_back({"sg", &s.sg, channels, 1});
    params.push_back({"su", &s.su, channels, 1});
    for (size_t i = 0; i < downSegs.size(); ++i)
        params.push_back({"wd" + std::to_string(i), &s.wd[i], hidden, downSegs[i].second});
    params.push_back({"sd", &s.sd, hidden, 1});
    std::string parameters;
    for (const Param& p : params) {
        parameters += (parameters.empty() ? "" : ", ") + p.s->bufferType(p.prows, p.pwidth) + " " + p.name;
        line(tensor(milType(p.s->element), p.prows, p.pwidth) + " " + p.name +
             "_t = tensor_buffer_to_tensor<ios17>(input = " + p.name + ")");
    }
    // dequantize the int8 input segments to fp16 (by 1/128).
    for (size_t k = 0; k < inSegs.size(); ++k)
        f16("x" + std::to_string(k) + "_d", inSegs[k].second, rows,
            "dequantize(input = x" + std::to_string(k) + "_t, scale = " + unit + ")");
    // gate and up: one matmul per hidden-segment (whole weight tensor), summed, scaled by the per-row weight scale.
    const auto proj = [&](const char* p, const char* w, const char* sc) {
        std::string total;
        for (size_t k = 0; k < inSegs.size(); ++k) {
            const std::string wk = std::string(w) + std::to_string(k), xk = "x" + std::to_string(k);
            f16(wk + "_d", channels, inSegs[k].second, "dequantize(input = " + wk + "_t, scale = " + unit + ")");
            const std::string mk = std::string(p) + "m" + std::to_string(k);
            f16(mk, channels, rows,
                "matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + wk + "_d, y = " + xk + "_d)");
            if (k == 0) {
                total = mk;
            } else {
                const std::string sum = std::string(p) + "m_sum" + std::to_string(k);
                f16(sum, channels, rows, "add(x = " + total + ", y = " + mk + ")");
                total = sum;
            }
        }
        f16(std::string(p) + "s", channels, rows, "mul(x = " + total + ", y = " + std::string(sc) + "_t)");
    };
    proj("g", "wg", "sg");
    proj("u", "wu", "su");
    // silu(g) = (g/2)(1 + tanh(g/2)) through the ANE's tanh; restore the token scale (gt) before the nonlinearity.
    // The up output us keeps its 1/tx factor, which the CPU join restores at the end.
    f16("gt", channels, rows, "mul(x = gs, y = tx_t)");
    f16("gh", channels, rows, "mul(x = gt, y = " + fp16lit(0.5) + ")");
    f16("th", channels, rows, "tanh(x = gh)");
    f16("tp", channels, rows, "add(x = th, y = " + fp16lit(1.0) + ")");
    f16("silu", channels, rows, "mul(x = gh, y = tp)");
    f16("h", channels, rows, "mul(x = silu, y = us)");
    // Hadamard-rotate the intermediate as a grouped 1x1 conv over [1, C, 1, rows].
    line("tensor<fp16, [1, " + c + ", 1, " + r + "]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, " + c +
         ", 1, " + r + "]))");
    line("tensor<fp16, [" + c + ", " + block +
         ", 1, 1]> rotation = const()[name = string(\"rotation\"), val = tensor<fp16, [" + c + ", " + block +
         ", 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(" +
         std::to_string(kConstantOffset) + ")))]");
    line("tensor<fp16, [1, " + c + ", 1, " + r +
         "]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32(" +
         std::to_string(channels / kIntermediateBlock) +
         "), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, "
         "1]), weight = rotation, x = h4)");
    f16("hr", channels, rows, "reshape(x = hr4, shape = tensor<int32, [4]>(" + dimensions(channels, rows) + "))");
    // per-token requantize of the rotated intermediate to int8.
    f16("habs", channels, rows, "abs(x = hr)");
    f16("peak", 1, rows, "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))");
    f16("floor", 1, rows, "maximum(x = peak, y = " + fp16lit(kIntermediateFloor) + ")");
    f16("inverse", 1, rows, "real_div(x = " + fp16lit(kInt8Peak) + ", y = floor)");
    f16("hs", channels, rows, "mul(x = hr, y = inverse)");
    line(tensor("int8", channels, rows) + " hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))");
    f16("hd", channels, rows, "dequantize(input = hq, scale = " + unit + ")");
    f16("hscale", 1, rows, "mul(x = floor, y = " + fp16lit(double(kInt8Unit) / kInt8Peak) + ")");
    // down: one matmul per C-segment; the weight is a whole tensor, the intermediate hd is sliced (axis 2). All the
    // matmuls come first, then the adds chain them (Splash's op order).
    for (size_t i = 0; i < downSegs.size(); ++i) {
        const auto [begin, width] = downSegs[i];
        const std::string idx = std::to_string(i), hdi = "hd" + idx, wdi = "wd" + idx;
        f16(hdi, width, rows,
            "slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, " + std::to_string(begin) +
                ", 0]), size = tensor<int32, [4]>(" + dimensions(width, rows) + "))");
        f16(wdi + "_d", hidden, width, "dequantize(input = " + wdi + "_t, scale = " + unit + ")");
        f16("dm" + idx, hidden, rows,
            "matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + wdi + "_d, y = " + hdi + ")");
    }
    std::string dtotal = "dm0";
    for (size_t i = 1; i < downSegs.size(); ++i) {
        const std::string sum = "dm_sum" + std::to_string(i);
        f16(sum, hidden, rows, "add(x = " + dtotal + ", y = dm" + std::to_string(i) + ")");
        dtotal = sum;
    }
    f16("ds", hidden, rows, "mul(x = " + dtotal + ", y = sd_t)");
    // append the per-token intermediate scale as a last row; the CPU join multiplies it with the input scale in fp32.
    f16("yt", hidden + 1, rows, "concat(axis = int32(2), interleave = bool(false), values = (ds, hscale))");
    line(s.y.bufferType(hidden + 1, rows) +
         " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
         "strides = tensor<int64, [4]>(" + s.y.strides(hidden + 1) + "))");
    return "    func " + functionName(rows) + "<ios18>(" + parameters + ") {\n" + body + "    } -> (y);\n";
}

// =====================================================================================================================
// CPU int8 quantization of weights and input, and the fp32 join (Splash ane_ffn_weights / ane_ffn_pack / ane_ffn_join,
// done here on the CPU for step 1).
// =====================================================================================================================
float halfToFloat(uint16_t h) { return float(std::bit_cast<_Float16>(h)); }
uint16_t floatToHalf(float f) { return std::bit_cast<uint16_t>(_Float16(f)); }

// Quantize a row-major fp16 matrix [rowsM, cols] to int8 codes [rowsM, cols] (channel/row layout unchanged) with a
// per-row scale carrying 128/127 * peak, so the MIL's dequant-by-1/128 restores w ~= code * scale * 127/128 / 127 ...
// code = round(w / peak * 127); dequant(code) = code/128; so scale must be peak * 128 / 127 for code/128 * scale ~= w.
[[maybe_unused]] void quantizeRows(const uint16_t* w, int rowsM, int cols, int8_t* codes, uint16_t* scales) {
    for (int row = 0; row < rowsM; ++row) {
        const uint16_t* src = w + size_t(row) * cols;
        float peak = 0x1p-14f;  // fp16 least normal (Splash PEAK_FLOOR), so the scale is a normal half
        for (int j = 0; j < cols; ++j) peak = std::max(peak, std::fabs(halfToFloat(src[j])));
        const float invPeak = kInt8Peak / peak;  // code = round(w * 127/peak)
        int8_t* dst = codes + size_t(row) * cols;
        for (int j = 0; j < cols; ++j) {
            int v = int(std::lround(halfToFloat(src[j]) * invPeak));
            dst[j] = int8_t(std::max(-127, std::min(127, v)));
        }
        scales[row] = floatToHalf(peak * (kInt8Unit / kInt8Peak));  // dequant uses 1/128, so scale = peak * 128/127
    }
}

// Quantize a row-major float matrix [rowsM, cols] to int8 + per-row fp16 scale, same convention as quantizeRows.
void quantizeRowsFloat(const float* w, int rowsM, int cols, int8_t* codes, uint16_t* scales) {
    for (int row = 0; row < rowsM; ++row) {
        const float* src = w + size_t(row) * cols;
        float peak = 0x1p-14f;
        for (int j = 0; j < cols; ++j) peak = std::max(peak, std::fabs(src[j]));
        const float invPeak = kInt8Peak / peak;
        int8_t* dst = codes + size_t(row) * cols;
        for (int j = 0; j < cols; ++j) {
            int v = int(std::lround(src[j] * invPeak));
            dst[j] = int8_t(std::max(-127, std::min(127, v)));
        }
        scales[row] = floatToHalf(peak * (kInt8Unit / kInt8Peak));
    }
}

// Whether the input / gate / up Hadamard rotation is applied (default on). The test turns it off (ane_program_set_input_rotate)
// to measure the outlier error without it; when off, both the input pack and the gate/up weight columns skip the
// rotation so the projection is still x Wg^T (no R on either side). Read on each call so a test can toggle it between
// two programs in the same process (bind_weights and the pack/requantize kernel build each read it fresh).
std::atomic<bool> gInputRotate{true};
bool inputRotateEnabled() { return gInputRotate.load(); }

// down(hr) with rotated weights equals down(hh) with the originals (R is orthogonal: Wd . hh = (Wd R^T) . (R hh)).
// wd is row-major [hidden, C]; each row's C inputs are rotated block by block (block kIntermediateBlock=512):
// wd'[row][g*512 + o] = sum_i wd[row][g*512 + i] * signs[o] * hadamard(o, i) * norm.
//
// The Hadamard sum_i x[i] * (-1)^popcount(o & i) is computed by a natural-order fast Walsh-Hadamard transform (FWHT,
// 9 butterfly stages for block 512 = 2^9) instead of the O(512^2) matrix sum: the GPU does the same transform with the
// same stage order and the same per-position add/sub, so the two stay bit-equal. (FWHT changes the fp32 rounding
// relative to the old sequential matrix sum by ~2e-5 per element; the int8 requantize that follows dominates the error
// and the end-to-end rel L2 is unchanged.)
std::vector<float> rotateDownWeights(const uint16_t* wd, int hidden, int C,
                                     const std::array<float, kIntermediateBlock>& signs) {
    const uint32_t block = kIntermediateBlock;
    const float norm = 1.0f / std::sqrt(float(block));
    const int blocks = C / int(block);
    std::vector<float> out(size_t(hidden) * C);
    std::vector<float> buf(block);
    for (int row = 0; row < hidden; ++row) {
        const uint16_t* src = wd + size_t(row) * C;
        float* dst = out.data() + size_t(row) * C;
        for (int g = 0; g < blocks; ++g) {
            for (uint32_t i = 0; i < block; ++i) buf[i] = halfToFloat(src[g * block + i]);
            // natural-order FWHT in place: stage len = 1, 2, 4, ..., 256; butterfly (u + v, u - v).
            for (uint32_t len = 1; len < block; len <<= 1)
                for (uint32_t i = 0; i < block; i += (len << 1))
                    for (uint32_t j = i; j < i + len; ++j) {
                        const float u = buf[j], v = buf[j + len];
                        buf[j] = u + v;
                        buf[j + len] = u - v;
                    }
            for (uint32_t o = 0; o < block; ++o) dst[g * block + o] = buf[o] * (signs[o] * norm);
        }
    }
    return out;
}

// Rotate one hidden row [hidden] by the input Hadamard R in blocks of kInputBlock=128: out[g*128 + o] =
// sum_i x[g*128 + i] * signs[o] * hadamard(o, i) / sqrt(128), computed by a natural-order FWHT (7 butterfly stages for
// 128 = 2^7) so it stays bit-equal to the GPU's simd-shuffle FWHT. signs are the first 128 of the shared sign array
// (Splash rotates an input block per rotation unit, ane_ffn_rotate_block<1> reading sign[lane*4+e], i.e. signs[0..127]).
// `src` is fp16, `dst` the rotated fp32 row. hidden is a multiple of 128 (Qwen3: 5120).
void rotateInputRow(const uint16_t* src, int hidden, const std::array<float, kIntermediateBlock>& signs, float* dst) {
    const uint32_t block = kInputBlock;
    const float norm = 1.0f / std::sqrt(float(block));
    const int blocks = hidden / int(block);
    float buf[kInputBlock];
    for (int g = 0; g < blocks; ++g) {
        for (uint32_t i = 0; i < block; ++i) buf[i] = halfToFloat(src[g * block + i]);
        for (uint32_t len = 1; len < block; len <<= 1)
            for (uint32_t i = 0; i < block; i += (len << 1))
                for (uint32_t j = i; j < i + len; ++j) {
                    const float u = buf[j], v = buf[j + len];
                    buf[j] = u + v;
                    buf[j + len] = u - v;
                }
        for (uint32_t o = 0; o < block; ++o) dst[g * block + o] = buf[o] * (signs[o] * norm);
    }
}

// Rotate the hidden columns of a gate/up weight row-major [C, hidden] (fp16) by the same input Hadamard R, so that
// (R x) . (R w) = x . w (R orthogonal): gate = (R x) (R Wg)^T restores x Wg^T. Each row's hidden values are rotated in
// blocks of kInputBlock=128 (same FWHT and signs as rotateInputRow), returned as fp32 [C, hidden].
std::vector<float> rotateGateUpWeights(const uint16_t* w, int C, int hidden,
                                       const std::array<float, kIntermediateBlock>& signs) {
    const uint32_t block = kInputBlock;
    const float norm = 1.0f / std::sqrt(float(block));
    const int blocks = hidden / int(block);
    std::vector<float> out(size_t(C) * hidden);
    float buf[kInputBlock];
    for (int row = 0; row < C; ++row) {
        const uint16_t* src = w + size_t(row) * hidden;
        float* dst = out.data() + size_t(row) * hidden;
        for (int g = 0; g < blocks; ++g) {
            for (uint32_t i = 0; i < block; ++i) buf[i] = halfToFloat(src[g * block + i]);
            for (uint32_t len = 1; len < block; len <<= 1)
                for (uint32_t i = 0; i < block; i += (len << 1))
                    for (uint32_t j = i; j < i + len; ++j) {
                        const float u = buf[j], v = buf[j + len];
                        buf[j] = u + v;
                        buf[j + len] = u - v;
                    }
            for (uint32_t o = 0; o < block; ++o) dst[g * block + o] = buf[o] * (signs[o] * norm);
        }
    }
    return out;
}

// Run `body(begin, end)` over [0, rows) split into contiguous chunks across hardware threads. For the CPU pack/join
// (step 1): the ANE evaluation itself is what step 2 overlaps with the GPU; here we keep the CPU parts off the
// critical path as much as a few threads allow.
template <class Body>
void parallelRows(int rows, Body&& body) {
    unsigned hw = std::thread::hardware_concurrency();
    int threads = int(hw ? std::min(hw, 8u) : 4u);
    threads = std::max(1, std::min(threads, rows));
    if (threads == 1) {
        body(0, rows);
        return;
    }
    std::vector<std::thread> pool;
    const int chunk = (rows + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
        const int begin = t * chunk, end = std::min(rows, begin + chunk);
        if (begin >= end) break;
        pool.emplace_back([&body, begin, end] { body(begin, end); });
    }
    for (std::thread& th : pool) th.join();
}

// Pack a row-major fp16 input [rows, hidden] to the program's per-hidden-segment channel-major int8 surfaces
// x[k] [segWidth, rows] with a per-token scale tx [rows]. Each token is first rotated by the input Hadamard R in
// blocks of 128 (rotateInputRow), then quantized against its own peak over all rotated hidden channels; the MIL
// dequantizes by 1/128 and the weight scales and this token scale carry it back. The gate/up weight columns are
// rotated by the same R at bind, so (R x)(R W)^T = x W^T is preserved.
void packInput(const uint16_t* xin, int rows, int hidden, const std::vector<Surface>& xSegs, const Surface& txS,
               const std::vector<std::pair<uint32_t, uint32_t>>& inSegs,
               const std::array<float, kIntermediateBlock>& signs) {
    uint16_t* txdst = static_cast<uint16_t*>(txS.contents());
    const bool rotate = inputRotateEnabled();
    parallelRows(rows, [&](int begin, int end) {
    std::vector<float> rot(hidden);
    for (int t = begin; t < end; ++t) {
        const uint16_t* src = xin + size_t(t) * hidden;
        if (rotate) rotateInputRow(src, hidden, signs, rot.data());
        else for (int j = 0; j < hidden; ++j) rot[j] = halfToFloat(src[j]);
        float peak = 0x1p-14f;
        for (int j = 0; j < hidden; ++j) peak = std::max(peak, std::fabs(rot[j]));
        const float invPeak = kInt8Peak / peak;
        for (size_t k = 0; k < inSegs.size(); ++k) {
            int8_t* xdst = static_cast<int8_t*>(xSegs[k].contents());
            const int xStride = xSegs[k].strideBytes;  // bytes per channel row of x[k] [segWidth, rows]
            const uint32_t base = inSegs[k].first, width = inSegs[k].second;
            for (uint32_t j = 0; j < width; ++j) {
                int v = int(std::lround(rot[base + j] * invPeak));
                xdst[size_t(j) * xStride + t] = int8_t(std::max(-127, std::min(127, v)));
            }
        }
        txdst[t] = floatToHalf(peak * (kInt8Unit / kInt8Peak));
    }
    });
}

// Join the program's output [hidden+1, rows] (down rows ds [hidden, rows] plus the per-token intermediate scale in the
// last row) with the per-token input scale into y [rows, hidden] fp16, in fp32.
void joinOutput(const Surface& yS, const Surface& txS, int rows, int hidden, uint16_t* yout) {
    const uint16_t* ybuf = static_cast<const uint16_t*>(yS.contents());
    const uint16_t* txbuf = static_cast<const uint16_t*>(txS.contents());
    const int yStride = yS.strideBytes / 2;   // halves per channel row of y [hidden+1, rows]
    const uint16_t* hscale = ybuf + size_t(hidden) * yStride;  // last row: per-token intermediate scale
    parallelRows(rows, [&](int begin, int end) {
    for (int t = begin; t < end; ++t) {
        const float inscale = halfToFloat(txbuf[t]), hsc = halfToFloat(hscale[t]);
        const float tokenScale = inscale * hsc;
        const bool scaleFinite = std::isfinite(inscale) && std::isfinite(hsc);
        uint16_t* dst = yout + size_t(t) * hidden;
        for (int ch = 0; ch < hidden; ++ch) {
            const float part = halfToFloat(ybuf[size_t(ch) * yStride + t]);
            const float v = part * tokenScale;
            // Splash's non-finite guard: a non-finite partial / scale contributes 0 (the GPU share stands for the row).
            dst[ch] = (scaleFinite && std::isfinite(part) && std::isfinite(v)) ? floatToHalf(v) : floatToHalf(0.0f);
        }
    }
    });
}

// =====================================================================================================================
// The program: own Metal device + shared event, the surfaces, the compiled/loaded model, per-layer int8 weights.
// =====================================================================================================================
struct LayerWeights final {
    bool bound = false;
    // per hidden-segment gate/up int8 codes [C, segWidth]; per C-segment down int8 codes [hidden, segWidth].
    std::vector<std::vector<int8_t>> wgCodes, wuCodes, wdCodes;
    std::vector<uint16_t> sgScales, suScales, sdScales;
};

class NativeAneProgram final : public AneProgram {
public:
    NativeAneProgram(int columns, int rows, int hidden, int layers)
        : columns_(columns), rows_(rows), hidden_(hidden), layers_(layers), api_(api()) {
        if (columns <= 0 || rows <= 0 || hidden <= 0 || layers <= 0)
            throw std::invalid_argument("ANE program: bad dimensions");
        if (columns % int(kIntermediateBlock) != 0)
            throw std::invalid_argument("ANE program: columns must be a multiple of 512");
        // The IOSurface-backed MTLBuffers must be created on MLX's own Metal device: mx::allocator::make_buffer (step 2,
        // pack/join/requantize) only wraps a pointer whose MTLBuffer belongs to MLX's device. (On Apple Silicon this is
        // the one GPU; MLX just needs the buffer registered with its device object.)
        {
            MTL::Device* mtl = mlx::core::metal::device(mlx::core::Device::gpu).mtl_device();
            device_ = (__bridge id<MTLDevice>)(void*)mtl;
        }
        if (!device_) throw std::runtime_error("no Metal device");
        event_ = [device_ newSharedEvent];
        if (!event_) throw std::runtime_error("no Metal shared event");
        if (![(id)event_ respondsToSelector:NSSelectorFromString(@"eventPort")])
            throw std::runtime_error("the Neural Engine cannot share this Metal event");

        // segment layout: hidden split into inSegs_ (gate/up input + weight segments), C into downSegs_.
        inSegs_ = segmentsOf(uint32_t(hidden_));
        downSegs_ = segmentsOf(uint32_t(columns_));

        // surfaces: x[k] [segWidth, rows] int8, wg[k]/wu[k] [C, segWidth] int8, wd[i] [hidden, segWidth] int8.
        ProgramSurfaces surf;
        for (const auto& [begin, width] : inSegs_) {
            (void)begin;
            surf.x.push_back(createSurface(device_, width, rows_, Element::Int8));
            surf.wg.push_back(createSurface(device_, columns_, width, Element::Int8));
            surf.wu.push_back(createSurface(device_, columns_, width, Element::Int8));
        }
        surf.tx = createSurface(device_, 1, rows_, Element::Float16);
        surf.sg = createSurface(device_, columns_, 1, Element::Float16);
        surf.su = createSurface(device_, columns_, 1, Element::Float16);
        for (const auto& [begin, width] : downSegs_) {
            (void)begin;
            surf.wd.push_back(createSurface(device_, hidden_, width, Element::Int8));
        }
        surf.sd = createSurface(device_, hidden_, 1, Element::Float16);
        surf.y = createSurface(device_, hidden_ + 1, rows_, Element::Float16);
        surf_ = std::move(surf);

        weights_.resize(layers_);

        signs_ = rotationSigns();
        const std::vector<uint8_t> blob = rotationBlob(columns_, signs_);
        rowCounts_ = rowVariants(uint32_t(rows_));
        std::string mil = "program(1.3)\n{\n";
        for (uint32_t r : rowCounts_) mil += buildFunction(r, columns_, hidden_, surf_);
        mil += "}\n";
        compileAndLoad(mil, blob);
    }

    ~NativeAneProgram() override {
        if (!model_) return;
        try {
            runOnQueue(api_, [this] {
                NSError* error = nil;
                [api_.client unloadModel:model_ options:@{} qos:kQos error:&error];
            });
        } catch (...) {
        }
    }

    int rows() const override { return rows_; }
    std::vector<int> row_counts() const override { return std::vector<int>(rowCounts_.begin(), rowCounts_.end()); }
    int fit_rows(int n) const override {
        for (uint32_t r : rowCounts_) if (int(r) >= n) return int(r);
        return -1;
    }
    int columns() const override { return columns_; }
    int hidden() const override { return hidden_; }
    bool has(int layer) const override {
        return layer >= 0 && layer < layers_ && weights_[layer].bound;
    }
    double last_eval_ms() const override { return lastEvalMs_; }

    void bind_weights(int layer, const void* wg, const void* wu, const void* wd) override {
        if (layer < 0 || layer >= layers_) throw std::invalid_argument("ANE program: bad layer");
        LayerWeights& lw = weights_[layer];
        lw.sgScales.resize(columns_);
        lw.suScales.resize(columns_);
        lw.sdScales.resize(hidden_);
        // gate/up: rotate the hidden columns by the input Hadamard R (so (R x)(R Wg)^T = x Wg^T restores the
        // projection after the input is rotated at pack), then quantize rows [C, H] (scale over all H) and split the
        // int8 codes by hidden-segment. When the rotation is off (test), quantize the fp16 rows directly.
        if (inputRotateEnabled()) {
            const std::vector<float> wgRot = rotateGateUpWeights(static_cast<const uint16_t*>(wg), columns_, hidden_, signs_);
            const std::vector<float> wuRot = rotateGateUpWeights(static_cast<const uint16_t*>(wu), columns_, hidden_, signs_);
            quantizeSegmentedFloat(wgRot.data(), columns_, hidden_, inSegs_, lw.wgCodes, lw.sgScales.data());
            quantizeSegmentedFloat(wuRot.data(), columns_, hidden_, inSegs_, lw.wuCodes, lw.suScales.data());
        } else {
            quantizeSegmentedHalf(static_cast<const uint16_t*>(wg), columns_, hidden_, inSegs_, lw.wgCodes,
                                  lw.sgScales.data());
            quantizeSegmentedHalf(static_cast<const uint16_t*>(wu), columns_, hidden_, inSegs_, lw.wuCodes,
                                  lw.suScales.data());
        }
        // down: pre-rotate by the intermediate Hadamard R (so down(R.hh) with it equals down(hh)), then quantize rows
        // [H, C] (scale over all C) and split the int8 codes by C-segment.
        const std::vector<float> wdRot = rotateDownWeights(static_cast<const uint16_t*>(wd), hidden_, columns_, signs_);
        quantizeSegmentedFloat(wdRot.data(), hidden_, columns_, downSegs_, lw.wdCodes, lw.sdScales.data());
        lw.bound = true;
    }

    void evaluate(int layer, const void* x, void* y) override {
        if (!has(layer)) throw std::runtime_error("ANE program: layer not bound");
        // stage this layer's int8 weights into the surfaces once per layer switch (the engine stages a layer ahead;
        // here it belongs to bind, not the timed evaluation of a bound layer).
        if (layer != stagedLayer_) {
            const LayerWeights& lw = weights_[layer];
            for (size_t k = 0; k < inSegs_.size(); ++k) {
                copyRows(surf_.wg[k], lw.wgCodes[k].data(), columns_, int(inSegs_[k].second), 1);
                copyRows(surf_.wu[k], lw.wuCodes[k].data(), columns_, int(inSegs_[k].second), 1);
            }
            for (size_t i = 0; i < downSegs_.size(); ++i)
                copyRows(surf_.wd[i], lw.wdCodes[i].data(), hidden_, int(downSegs_[i].second), 1);
            copyRows(surf_.sg, lw.sgScales.data(), columns_, 1, 2);
            copyRows(surf_.su, lw.suScales.data(), columns_, 1, 2);
            copyRows(surf_.sd, lw.sdScales.data(), hidden_, 1, 2);
            stagedLayer_ = layer;
        }
        // pack the fp16 input to per-token int8 segments + scale.
        packInput(static_cast<const uint16_t*>(x), rows_, hidden_, surf_.x, surf_.tx, inSegs_, signs_);
        // one blocking ANE evaluation (timed: last_eval_ms).
        const auto evalStart = Clock::now();
        runEvaluation();
        lastEvalMs_ = std::chrono::duration<double, std::milli>(Clock::now() - evalStart).count();
        // join the per-token-scaled fp32 output back to fp16 [rows, hidden].
        joinOutput(surf_.y, surf_.tx, rows_, hidden_, static_cast<uint16_t*>(y));
    }

private:
    // Copy a dense row-major [rowsM, cols] (element `elemBytes` bytes) into a surface, padding each row to its stride.
    template <class T>
    void copyRows(const Surface& s, const T* src, int rowsM, int cols, int elemBytes) {
        uint8_t* dst = static_cast<uint8_t*>(s.contents());
        const size_t rowBytes = size_t(cols) * elemBytes;
        for (int r = 0; r < rowsM; ++r)
            std::memcpy(dst + size_t(r) * s.strideBytes, reinterpret_cast<const uint8_t*>(src) + size_t(r) * rowBytes,
                        rowBytes);
    }

    // Quantize a row-major fp16 matrix [rowsM, cols] to int8 + per-row scale (scale over all cols), then split the
    // int8 codes by column-segment into one dense [rowsM, segWidth] block per segment.
    [[maybe_unused]] static void quantizeSegmentedHalf(const uint16_t* w, int rowsM, int cols,
                                      const std::vector<std::pair<uint32_t, uint32_t>>& segs,
                                      std::vector<std::vector<int8_t>>& blocks, uint16_t* scales) {
        std::vector<int8_t> codes(size_t(rowsM) * cols);
        quantizeRows(w, rowsM, cols, codes.data(), scales);
        splitCodes(codes.data(), rowsM, cols, segs, blocks);
    }
    static void quantizeSegmentedFloat(const float* w, int rowsM, int cols,
                                       const std::vector<std::pair<uint32_t, uint32_t>>& segs,
                                       std::vector<std::vector<int8_t>>& blocks, uint16_t* scales) {
        std::vector<int8_t> codes(size_t(rowsM) * cols);
        quantizeRowsFloat(w, rowsM, cols, codes.data(), scales);
        splitCodes(codes.data(), rowsM, cols, segs, blocks);
    }
    // Split dense int8 codes [rowsM, cols] into [rowsM, segWidth] blocks by column-segment.
    static void splitCodes(const int8_t* codes, int rowsM, int cols,
                           const std::vector<std::pair<uint32_t, uint32_t>>& segs,
                           std::vector<std::vector<int8_t>>& blocks) {
        blocks.assign(segs.size(), {});
        for (size_t k = 0; k < segs.size(); ++k) {
            const auto [begin, width] = segs[k];
            blocks[k].resize(size_t(rowsM) * width);
            for (int r = 0; r < rowsM; ++r)
                std::memcpy(blocks[k].data() + size_t(r) * width, codes + size_t(r) * cols + begin, width);
        }
    }

    void compileAndLoad(const std::string& mil, const std::vector<uint8_t>& blob) {
        const std::string key = hex(fnv1a(blob, fnv1a(bytesOf(mil))));
        const std::filesystem::path cache = cacheDirectory();
        const std::filesystem::path directory = cache / key;
        std::filesystem::create_directories(directory);
        const std::span<const uint8_t> milBytes = bytesOf(mil);
        if (!holds(directory / "model.mil", milBytes)) writeWhole(directory / "model.mil", milBytes);
        if (!holds(directory / "weights.bin", blob)) writeWhole(directory / "weights.bin", blob);

        runOnQueue(api_, [this, directory, key] {
            model_ = [api_.model modelAtURL:[NSURL fileURLWithPath:@(directory.c_str()) isDirectory:YES]
                                        key:@(key.c_str())];
            if (!model_) fail("model creation", nil);
            const bool compiled = [api_.client compiledModelExistsFor:model_];
            if (!compiled) {
                NSDictionary* options = @{@"kANEFModelType" : @"kANEFModelMIL", @"kANEFNetPlistFilenameKey" : @"model.mil"};
                NSError* error = nil;
                if (![api_.client compileModel:model_ options:options qos:kQos error:&error]) fail("compilation", error);
            }
            NSError* error = nil;
            if (![api_.client loadModel:model_ options:@{} qos:kQos error:&error]) {
                // the service may hold an uncompilable compilation from another compiler: purge, compile, load again.
                [api_.client purgeCompiledModel:model_];
                NSDictionary* options = @{@"kANEFModelType" : @"kANEFModelMIL", @"kANEFNetPlistFilenameKey" : @"model.mil"};
                if (![api_.client compileModel:model_ options:options qos:kQos error:&error]) fail("compilation", error);
                error = nil;
                if (![api_.client loadModel:model_ options:@{} qos:kQos error:&error]) fail("load", error);
            }
            describe();
        });
    }

    // Read the loaded model's procedures: one function per row count (ffn<rows>), each with its input symbol
    // names/indices and output symbol index. The maximum-rows function is the CPU path's (inputNames_ and friends).
    void describe() {
        NSDictionary* attributes = [model_ modelAttributes];
        NSDictionary* description = attributes[@"ANEFModelDescription"];
        NSArray* symbols = description[@"kANEFModelInputSymbolsArrayKey"];
        NSDictionary* functions = description[@"kANEFModelProcedureNameToIDMapKey"];
        NSArray* entries = description[@"ANEFModelProcedures"];
        if (![symbols isKindOfClass:NSArray.class] || ![functions isKindOfClass:NSDictionary.class] ||
            ![entries isKindOfClass:NSArray.class] || entries.count != functions.count || entries.count != rowCounts_.size())
            throw std::runtime_error("ANE program does not describe its procedures");
        variants_.clear();
        for (NSString* function in functions) {
            const uint32_t index = [functions[function] unsignedIntValue];
            if (index >= entries.count) throw std::runtime_error("ANE program names an unknown procedure");
            NSDictionary* entry = entries[index];
            Variant v;
            v.rows = uint32_t(std::stoul(std::string(function.UTF8String).substr(3)));  // "ffn<rows>"
            v.procedure = index;
            v.outputSymbol = [[entry[@"ANEFModelOutputSymbolIndexArray"] objectAtIndex:0] unsignedIntValue];
            for (NSNumber* symbol in entry[@"ANEFModelInputSymbolIndexArray"]) {
                v.inputSymbols.push_back(symbol.unsignedIntValue);
                v.inputNames.push_back([symbols[symbol.unsignedIntValue] UTF8String]);
            }
            variants_.push_back(std::move(v));
        }
        std::sort(variants_.begin(), variants_.end(), [](const Variant& a, const Variant& b) { return a.rows < b.rows; });
        const Variant& top = variants_.back();
        if (top.rows != uint32_t(rows_)) throw std::runtime_error("ANE program lacks its maximum-rows function");
        inputNames_ = top.inputNames;
        inputSymbols_ = top.inputSymbols;
        outputSymbol_ = top.outputSymbol;
        procedure_ = top.procedure;
    }

    // Bind the surfaces to the request's inputs by name, then run one evaluation ordered by the shared event and block
    // on its completion handler (Splash enqueue, made synchronous for the CPU-driven path).
    void runEvaluation() {
        @autoreleasepool {
            // the surfaces by input name (x<k>, wg<k>, wu<k>, wd<i>, tx, sg, su, sd).
            const auto surfaceFor = [&](const std::string& name) -> const Surface& {
                if (name == "tx") return surf_.tx;
                if (name == "sg") return surf_.sg;
                if (name == "su") return surf_.su;
                if (name == "sd") return surf_.sd;
                if (name.size() >= 2 && name[0] == 'x') return surf_.x[std::stoul(name.substr(1))];
                if (name.size() >= 3 && name[0] == 'w' && name[1] == 'g') return surf_.wg[std::stoul(name.substr(2))];
                if (name.size() >= 3 && name[0] == 'w' && name[1] == 'u') return surf_.wu[std::stoul(name.substr(2))];
                if (name.size() >= 3 && name[0] == 'w' && name[1] == 'd') return surf_.wd[std::stoul(name.substr(2))];
                throw std::runtime_error("ANE program has an unexpected input " + name);
            };
            NSMutableArray* objects = [NSMutableArray arrayWithCapacity:inputNames_.size()];
            NSMutableArray* indices = [NSMutableArray arrayWithCapacity:inputNames_.size()];
            const auto object = [&](const Surface& s) {
                id result = [api_.surface objectWithIOSurface:(IOSurfaceRef)s.surface.get()];
                if (!result) fail("surface object creation", nil);
                return result;
            };
            for (size_t i = 0; i < inputNames_.size(); ++i) {
                [objects addObject:object(surfaceFor(inputNames_[i]))];
                [indices addObject:@(inputSymbols_[i])];
            }
            NSArray* outputs = @[ object(surf_.y) ];
            NSArray* outputIndices = @[ @(outputSymbol_) ];

            id native = (id)event_;
            const uint64_t wait = ++eventValue_, signal = ++eventValue_;
            [event_ setSignaledValue:wait];  // start at once (CPU-driven; no GPU work precedes it)
            id signalEvent = [api_.signalEvent signalEventWithValue:signal symbolIndex:0 eventType:0 sharedEvent:native];
            id waitEvent = [api_.waitEvent waitEventWithValue:wait sharedEvent:native eventType:0];
            if (!signalEvent || !waitEvent) fail("event creation", nil);
            id events = [api_.events sharedEventsWithSignalEvents:@[ signalEvent ] waitEvents:@[ waitEvent ]];
            if (!events) fail("event creation", nil);
            id request = [api_.request requestWithInputs:objects
                                            inputIndices:indices
                                                 outputs:outputs
                                           outputIndices:outputIndices
                                           weightsBuffer:nil
                                               perfStats:nil
                                          procedureIndex:@(procedure_)
                                            sharedEvents:events
                                       transactionHandle:nil];
            if (!request) fail("request creation", nil);

            auto done = std::make_shared<std::pair<std::mutex, std::condition_variable>>();
            auto state = std::make_shared<std::pair<bool, bool>>(false, false);  // {finished, success}
            void (^report)(BOOL, NSError*) = ^(BOOL success, NSError*) {
                std::lock_guard lock(done->first);
                state->first = true;
                state->second = success;
                done->second.notify_all();
            };
            [request setCompletionHandler:report];
            NSError* error = nil;
            if (![api_.client evaluateWithModel:model_ options:@{} request:request qos:kQos error:&error])
                fail("evaluation", error);
            std::unique_lock lock(done->first);
            done->second.wait(lock, [&] { return state->first; });
            if (!state->second) throw std::runtime_error("ANE evaluation reported failure");
        }
    }

public:
    // =================================================================================================================
    // Step 2: GPU pack / join / weight requantize into the IOSurfaces, and the asynchronous evaluate.
    // =================================================================================================================
    void reserve_slots(int slots) override {
        if (slots < 1) throw std::invalid_argument("ANE program: slots must be >= 1");
        if (!altReady_) {
            for (const auto& [begin, width] : inSegs_) {
                (void)begin;
                alt_.x.push_back(createSurface(device_, width, rows_, Element::Int8));
            }
            alt_.tx = createSurface(device_, 1, rows_, Element::Float16);
            alt_.y = createSurface(device_, hidden_ + 1, rows_, Element::Float16);
            altReady_ = true;
        }
        if (slots <= int(slotWeights_.size())) return;
        // Each slot owns its own weight surfaces (wg[k]/wu[k]/wd[i] int8 + sg/su/sd fp16), so layer l+1 can be
        // requantized into one slot while layer l evaluates from the other (Splash's double-buffered weight surfaces).
        for (int s = int(slotWeights_.size()); s < slots; ++s) {
            SlotSurfaces ss;
            for (const auto& [begin, width] : inSegs_) {
                (void)begin;
                ss.wg.push_back(createSurface(device_, columns_, width, Element::Int8));
                ss.wu.push_back(createSurface(device_, columns_, width, Element::Int8));
            }
            ss.sg = createSurface(device_, columns_, 1, Element::Float16);
            ss.su = createSurface(device_, columns_, 1, Element::Float16);
            for (const auto& [begin, width] : downSegs_) {
                (void)begin;
                ss.wd.push_back(createSurface(device_, hidden_, width, Element::Int8));
            }
            ss.sd = createSurface(device_, hidden_, 1, Element::Float16);
            ss.staged = false;
            slotWeights_.push_back(std::move(ss));
        }
    }

    mlx::core::array requantize_weights_gpu(int slot, const NaxAffine& wg_full, const NaxAffine& wu_full,
                                            const NaxAffine& wd_full, int share) override {
        namespace mx = mlx::core;
        if (slot < 0 || slot >= int(slotWeights_.size())) throw std::invalid_argument("ANE program: bad slot");
        if (share != columns_) throw std::invalid_argument("ANE program: share must equal columns()");
        if (wg_full.columns != hidden_ || wu_full.columns != hidden_)
            throw std::invalid_argument("ANE program: gate/up columns must be hidden");
        if (wd_full.rows != hidden_) throw std::invalid_argument("ANE program: down rows must be hidden");
        if (wg_full.rows < share || wu_full.rows < share || wd_full.columns < share)
            throw std::invalid_argument("ANE program: NaxAffine too small for the ANE share");
        SlotSurfaces& ss = slotWeights_[slot];
        if (wg_full.tq4 != wu_full.tq4 || wg_full.tq4 != wd_full.tq4)
            throw std::invalid_argument("ANE program: gate/up/down must all be TQ4 or all plain affine");
        inputTq4_ = wg_full.tq4;   // the packs queued from now on (this layer's slices) rotate x in this format's basis

        // The ANE share is the last `share` intermediate channels: gate/up rows [F-share, F) and down columns
        // [F-share, F). F - share and share are multiples of 512 (so of 32, 16 and the group), so the slice aligns to
        // 32-row blocks and 16-column steps and the decode of each kept block/step is bit-identical to the matching
        // region of the full dequant. The whole requantize reads the 4-bit codes directly and writes int8 + scales in
        // one kernel per matrix: no fp16/fp32 weight copy is materialised. Every kernel reads the 4-bit codes directly
        // and writes int8 + per-row scales into the slot surfaces.
        if (!signsArr_) {
            std::vector<float> sv(signs_.begin(), signs_.end());
            signsArr_ = mx::array(sv.data(), {int(kIntermediateBlock)}, mx::float32);
            mx::eval(*signsArr_);
        }
        const mx::StreamOrDevice gpu = mx::Device(mx::Device::gpu);
        std::vector<mx::array> deps;
        // gate / up: decode the last C rows [C, H] straight to int8 + per-row scale, split by hidden-segment.
        decodeQuantizeRowsGpu(wg_full, (wg_full.rows - share) / 32, share, inSegs_, ss.wg, ss.sg, deps, gpu);
        decodeQuantizeRowsGpu(wu_full, (wu_full.rows - share) / 32, share, inSegs_, ss.wu, ss.su, deps, gpu);
        // down: decode the last C columns [H, C], rotate each 512-block by the intermediate Hadamard (FWHT) and
        // quantize rows to int8 + per-row scale, split by C-segment, all in one kernel.
        decodeRotateQuantizeDownGpu(wd_full, wd_full.columns - share, share, downSegs_, ss.wd, ss.sd, deps, gpu);
        ss.staged = true;
        return mergeDeps(deps);
    }

    mlx::core::array pack_input_gpu(const mlx::core::array& x16, int parity,
                                    const mlx::core::array* after) override {
        namespace mx = mlx::core;
        if (x16.ndim() != 2 || x16.shape(1) != hidden_ ||
            std::find(rowCounts_.begin(), rowCounts_.end(), uint32_t(x16.shape(0))) == rowCounts_.end())
            throw std::invalid_argument("ANE program: x16 must be [n, hidden] with n one of row_counts()");
        if (parity < 0 || parity > 1 || (parity == 1 && !altReady_))
            throw std::invalid_argument("ANE program: bad parity (reserve_slots first for parity 1)");
        curRows_[parity] = x16.shape(0);
        auto x = x16.dtype() == mx::float16 ? mx::contiguous(x16) : mx::contiguous(mx::astype(x16, mx::float16));
        // write each input segment's int8 codes and the per-token scale (tx) into the IOSurfaces, on the GPU.
        std::vector<mx::array> deps;
        packInputGpu(x, deps, parity, after);
        return mergeDeps(deps);
    }

    mlx::core::array join_output_gpu(const mlx::core::array& token_scale, int parity) override {
        (void)token_scale;  // the join reads the parity's tx and y surfaces directly; the handle only orders the eval.
        return joinOutputGpu(parity);
    }

    void ready_after(const mlx::core::array& evaluated_pack) override {
        // async_eval attaches an event per stream to the arrays it schedules and signals it at the end of the command
        // buffer that ran them (mlx/transforms.cpp eval_impl); a copy keeps its MTLSharedEvent alive.
        const mlx::core::Event& e = evaluated_pack.event();
        if (!e.valid()) throw std::runtime_error("ANE program: ready_after needs an async_eval'd, unfinished pack");
        ready_ = e;
    }

    void enqueue(int slot, bool after_pack, int parity) override {
        if (slot < 0 || slot >= int(slotWeights_.size()) || !slotWeights_[slot].staged)
            throw std::runtime_error("ANE program: slot not staged");
        if (parity < 0 || parity > 1 || (parity == 1 && !altReady_))
            throw std::invalid_argument("ANE program: bad parity");
        // after_pack=true: the ANE waits on the event of ready_after()'s command buffer (the CPU did not wait for the
        // pack). false: the caller evaluated the pack; the ANE starts at once.
        runEnqueue(slot, parity, after_pack);
    }

    void wait(int parity) override { runWait(parity); }

    void wait_gpu(int parity) override {
        Pending& pd = pend_[parity];
        if (!pd.done || !aneDone_) throw std::runtime_error("ANE program: wait_gpu() without enqueue()");
        // the GPU stream waits for the request's done value; later GPU work (the join) runs after the ANE wrote y.
#if MLX_VERSION_NUMERIC >= 32004
        // stock MLX 0.32.4 does not export Event(Stream) / Event::wait(Stream): encode the wait on the stream's command
        // buffer as Event::wait does (close the open compute encoder, then encodeWait)
        auto& enc = mlx::core::metal::get_command_encoder(mlx::core::to_stream(mlx::core::Device(mlx::core::Device::gpu)));
        enc.end_encoding();
        MTL::CommandBuffer* cb = enc.get_command_buffer();
        if (!cb) throw std::runtime_error("ANE program: no open MLX command buffer to wait in");
        cb->encodeWait((MTL::Event*)(__bridge void*)aneDone_, pd.doneValue);
#else
        mlx::core::Event e = *aneDone_;
        e.set_value(pd.doneValue);
        e.wait(mlx::core::to_stream(mlx::core::Device(mlx::core::Device::gpu)));
#endif
    }

    void debug_copy_pack(std::vector<uint16_t>& tx, std::vector<int8_t>& x0, int& x0_rows, int& x0_stride) override {
        tx.assign(rows_, 0);
        std::memcpy(tx.data(), surf_.tx.contents(), size_t(rows_) * 2);
        x0_rows = int(surf_.x[0].rows);
        x0_stride = int(surf_.x[0].strideBytes);
        x0.assign(size_t(x0_rows) * x0_stride, 0);
        std::memcpy(x0.data(), surf_.x[0].contents(), x0.size());
    }

    // Test-only: dump the staged weight scales (sg/su/sd first few) and first wg/wd codes from either the CPU path
    // (shared surf_) or a GPU slot, to compare GPU requantize to CPU bind.
    void debug_copy_weights(int slot, std::vector<uint16_t>& sg, std::vector<uint16_t>& sd, std::vector<int8_t>& wg0,
                            std::vector<int8_t>& wd0, int& wg0_rows, int& wg0_stride, int& wd0_rows, int& wd0_stride) override {
        const Surface& sgS = slot < 0 ? surf_.sg : slotWeights_[slot].sg;
        const Surface& sdS = slot < 0 ? surf_.sd : slotWeights_[slot].sd;
        const Surface& wgS = slot < 0 ? surf_.wg[0] : slotWeights_[slot].wg[0];
        const Surface& wdS = slot < 0 ? surf_.wd[0] : slotWeights_[slot].wd[0];
        sg.assign(columns_, 0);
        { const uint16_t* p = static_cast<const uint16_t*>(sgS.contents()); const int st = sgS.strideBytes / 2;
          for (int r = 0; r < columns_; ++r) sg[r] = p[size_t(r) * st]; }
        sd.assign(hidden_, 0);
        { const uint16_t* p = static_cast<const uint16_t*>(sdS.contents()); const int st = sdS.strideBytes / 2;
          for (int r = 0; r < hidden_; ++r) sd[r] = p[size_t(r) * st]; }
        wg0_rows = int(wgS.rows); wg0_stride = int(wgS.strideBytes);
        wg0.assign(size_t(wg0_rows) * wg0_stride, 0); std::memcpy(wg0.data(), wgS.contents(), wg0.size());
        wd0_rows = int(wdS.rows); wd0_stride = int(wdS.strideBytes);
        wd0.assign(size_t(wd0_rows) * wd0_stride, 0); std::memcpy(wd0.data(), wdS.contents(), wd0.size());
    }

private:
    // ---- GPU helpers (metal_kernel) ----

    // An MLX array over an IOSurface-backed MTLBuffer (zero-copy): make_buffer wraps the buffer's contents pointer, and
    // the array holds a release-only deleter. The array covers `shape` elements of `dtype` from the surface base; the
    // surface's padded row stride is passed to the kernels separately. Writing through it needs a non-const cast in the
    // kernel source (MLX declares inputs const).
    mlx::core::array surfaceArray(const Surface& s, std::vector<int> shape, mlx::core::Dtype dtype) {
        namespace mx = mlx::core;
        const uint64_t allocSize = IOSurfaceGetAllocSize((IOSurfaceRef)s.surface.get());
        mx::allocator::Buffer buf = mx::allocator::make_buffer([s.buffer contents], allocSize);
        if (!buf.ptr()) throw std::runtime_error("ANE program: MLX could not wrap the IOSurface buffer");
        mx::Shape sh(shape.begin(), shape.end());
#if MLX_VERSION_NUMERIC >= 32004  // MLX 0.32.4 wraps the buffer and its deleter in allocator::Data
        return mx::array(mx::allocator::Data(buf, [](mx::allocator::Buffer b) { mx::allocator::release(b); }), std::move(sh), dtype);
#else
        return mx::array(buf, std::move(sh), dtype, [](mx::allocator::Buffer b) { mx::allocator::release(b); });
#endif
    }

    // Merge dependency handles into one small MLX array the caller evals once to run all the surface writes. Each dep
    // (a write kernel's dummy output, or a written-through surface array) is reduced to a scalar and summed, so the
    // returned handle structurally depends on every write without the caller tracking them.
    mlx::core::array mergeDeps(const std::vector<mlx::core::array>& deps) {
        namespace mx = mlx::core;
        int z = 0;
        mx::array acc = mx::array(&z, {1}, mx::int32);
        for (const auto& d : deps) {
            auto r = mx::sum(mx::astype(mx::flatten(d), mx::int32));
            acc = mx::add(acc, mx::reshape(r, {1}));
        }
        return acc;
    }

    // Pack the FP16 input x [rows, hidden] into the per-hidden-segment int8 x-surfaces and the tx scale surface, on the
    // GPU, matching the CPU packInput bit for bit: each token's hidden row is first rotated by the input Hadamard R in
    // blocks of 128 (scalar natural-order FWHT + signs/sqrt(128), the same op order as rotateInputRow), then the
    // per-token peak over all rotated channels (floor 0x1p-14), invPeak = 127/peak, code = clamp(round(rot*invPeak),
    // -127, 127), tx = half(peak*128/127). One thread per token. The FWHT is recomputed per block in each pass (peak,
    // then write); it is deterministic, so both passes give the same floats and the result equals the CPU bit for bit.
    void packInputGpu(const mlx::core::array& x, std::vector<mlx::core::array>& deps, int parity = 0,
                      const mlx::core::array* extra = nullptr) {
        namespace mx = mlx::core;
        if (!signsArr_) {
            std::vector<float> sv(signs_.begin(), signs_.end());
            signsArr_ = mx::array(sv.data(), {int(kIntermediateBlock)}, mx::float32);
            mx::eval(*signsArr_);
        }
        std::vector<Surface>& xs = xOf(parity);
        Surface& txs = txOf(parity);
        // bind the x-segment surfaces and tx as write-through inputs.
        std::vector<mx::array> segArrays;
        segArrays.reserve(inSegs_.size());
        std::vector<uint32_t> segBegin(inSegs_.size()), segWidth(inSegs_.size()), segStride(inSegs_.size());
        for (size_t k = 0; k < inSegs_.size(); ++k) {
            segArrays.push_back(surfaceArray(xs[k], {int(uint64_t(xs[k].strideBytes) * xs[k].rows)}, mx::int8));
            segBegin[k] = inSegs_[k].first;
            segWidth[k] = inSegs_[k].second;
            segStride[k] = xs[k].strideBytes;
        }
        auto txArr = surfaceArray(txs, {int(uint64_t(txs.strideBytes) / 2 * txs.rows)}, mx::float16);

        // rotate the block at hidden offset `off` of token t into blk[0..127] (scalar natural-order FWHT, signs[0..127]).
        // When the rotation is off (test), blk is just the raw fp16 block.
        const bool tq4in = inputTq4_;   // the layer's weights are TQ4: its stored basis is H*D*(x) per 128-block
        const bool doRotate = inputRotateEnabled() || tq4in;
        const std::string rotate_tq4 =
            "    for (uint i = 0; i < 128u; ++i)\n"
            "      blk[i] = float(xin[t * H + off + i]) * (((TW[i >> 5u] >> (i & 31u)) & 1u) ? -1.0f : 1.0f);\n"
            "    for (uint len = 1u; len < 128u; len <<= 1)\n"
            "      for (uint i = 0; i < 128u; i += (len << 1))\n"
            "        for (uint j = i; j < i + len; ++j) {\n"
            "          float u = blk[j], vv = blk[j + len];\n"
            "          blk[j] = u + vv; blk[j + len] = u - vv;\n"
            "        }\n"
            "    for (uint o = 0; o < 128u; ++o) blk[o] = blk[o] * (1.0f / sqrt(128.0f));\n";
        const std::string rotate_on =
            "    for (uint i = 0; i < 128u; ++i) blk[i] = float(xin[t * H + off + i]);\n"
            "    for (uint len = 1u; len < 128u; len <<= 1)\n"
            "      for (uint i = 0; i < 128u; i += (len << 1))\n"
            "        for (uint j = i; j < i + len; ++j) {\n"
            "          float u = blk[j], vv = blk[j + len];\n"
            "          blk[j] = u + vv; blk[j + len] = u - vv;\n"
            "        }\n"
            "    for (uint o = 0; o < 128u; ++o) blk[o] = blk[o] * (sgn[o] * (1.0f / sqrt(128.0f)));\n";
        const std::string rotate_off =
            "    for (uint i = 0; i < 128u; ++i) blk[i] = float(xin[t * H + off + i]);\n";
        const std::string rotate = tq4in ? rotate_tq4 : doRotate ? rotate_on : rotate_off;
        const std::string kernelSuffix = tq4in ? "tq4_" : doRotate ? "rot_" : "raw_";
        const std::string src =
            "            uint t = thread_position_in_grid.x;   // one thread per token\n"
            "            if (t >= rows[0]) return;\n"
            "            const uint H = hidden[0];\n"
            "            device const half* xin = (device const half*)x;   // [rows, H] row-major\n"
            "            device const float* sgn = (device const float*)signs;\n"
            "            constexpr uint TW[4] = {0xE46A0359u, 0x42F85949u, 0xA4C63BBBu, 0x49F250A9u};\n"
            "            float blk[128];\n"
            "            float peak = 0x1p-14f;\n"
            "            for (uint off = 0; off < H; off += 128u) {\n" + rotate +
            "              for (uint o = 0; o < 128u; ++o) peak = max(peak, fabs(blk[o]));\n"
            "            }\n"
            "            float invPeak = 127.0f / peak;\n";
        // build the per-segment write loops: re-rotate each 128-block of the segment, then write int8.
        std::string segLoops;
        for (size_t k = 0; k < inSegs_.size(); ++k) {
            const std::string ks = std::to_string(k);
            segLoops += "{ device char* xs = (device char*)x" + ks + "; const uint base = " +
                        std::to_string(segBegin[k]) + "u, width = " + std::to_string(segWidth[k]) + "u, stride = " +
                        std::to_string(segStride[k]) + "u;\n" +
                        "  for (uint bo = 0; bo < width; bo += 128u) {\n"
                        "    uint off = base + bo;\n" + rotate +
                        "    for (uint o = 0; o < 128u; ++o) {\n"
                        "      float fv = blk[o] * invPeak;\n"
                        "      float r = round(fv);\n"               // round half away from zero == std::lround
                        "      int c = int(r); c = max(-127, min(127, c));\n"
                        "      xs[(bo + o) * stride + t] = char(c);\n"
                        "    }\n"
                        "  } }\n";
        }
        const std::string tail =
            "            device half* txs = (device half*)tx;\n"
            "            txs[t] = half(peak * (128.0f / 127.0f));\n";
        const std::string source = src + segLoops + tail;

        std::vector<std::string> names = {"x", "rows", "hidden", "signs", "tx", "after"};
        // `after`: the parity's last join output (unread). MLX orders this kernel after the kernel that wrote it, so
        // the pack never overwrites tx/x while the previous join of this parity still reads them.
        // and `extra` (e.g. the slot's requantize): MLX orders the pack, and so the ANE's ready event, after both.
        mx::array after = joinDone_[parity] ? *joinDone_[parity] : scalarU32(0u);
        if (extra) after = mx::add(mx::sum(mx::astype(mx::flatten(after), mx::int32)),
                                   mx::sum(mx::astype(mx::flatten(*extra), mx::int32)));
        std::vector<mx::array> args = {x, scalarU32(uint32_t(curRows_[parity])), scalarU32(uint32_t(hidden_)), *signsArr_, txArr,
                                       after};
        for (size_t k = 0; k < inSegs_.size(); ++k) {
            names.push_back("x" + std::to_string(k));
            args.push_back(segArrays[k]);
        }
        auto fn = mx::fast::metal_kernel("tqllm_ane_pack2_" + kernelSuffix + std::to_string(inSegs_.size()), names,
                                         {"dummy"}, source, "", false, false);
        const int grid = ((curRows_[parity] + 63) / 64) * 64;
        auto outs = fn(args, {{1}}, {mx::int8}, {grid, 1, 1}, {64, 1, 1}, {}, std::nullopt, false,
                       mx::Device(mx::Device::gpu));
        // Only the kernel's dummy output needs to reach the returned handle: evaluating it runs the kernel, whose
        // writes to the x-segment and tx surfaces complete when the command buffer does. Summing the full int8
        // surfaces here (tens of MB) would add a large reduction for no ordering benefit.
        deps.push_back(outs[0]);
    }

    // Join the output surface y [hidden+1, rows] with the per-token scale (tx surface) into y_out [rows, hidden] FP32,
    // matching the CPU joinOutput: tokenScale = tx[t] * hscale[t] (hscale = row `hidden`), y_out[t][ch] =
    // ybuf[ch*yStride + t] * tokenScale, computed in fp32. One thread per (token, channel) via a 2D grid.
    mlx::core::array joinOutputGpu(int parity = 0) {
        namespace mx = mlx::core;
        Surface& ys = yOf(parity);
        Surface& txs = txOf(parity);
        auto yArr = surfaceArray(ys, {int(uint64_t(ys.strideBytes) / 2 * ys.rows)}, mx::float16);
        auto txArr = surfaceArray(txs, {int(uint64_t(txs.strideBytes) / 2 * txs.rows)}, mx::float16);
        const uint32_t yStride = ys.strideBytes / 2;
        const std::string source =
            "uint t = thread_position_in_grid.x;\n"
            "uint ch = thread_position_in_grid.y;\n"
            "const uint R = rows[0], H = hidden[0], ys = ystride[0];\n"
            "if (t >= R || ch >= H) return;\n"
            "device const half* yb = (device const half*)y;\n"
            "device const half* txb = (device const half*)tx;\n"
            // Splash's join flags a non-finite ANE value or token scale and drops its contribution (an infinity in the
            // ANE output must not poison the row). Match that: if the partial, the intermediate scale or the input
            // scale of this token is not finite, contribute 0 for it (the GPU share alone stands for the row). For
            // finite inputs this is identical to the plain join, so GPU == CPU stays bitwise on every real slice.
            "float part = float(yb[ch * ys + t]);\n"
            "float hscale = float(yb[H * ys + t]);\n"
            "float inscale = float(txb[t]);\n"
            "float tokenScale = inscale * hscale;\n"
            "float v = part * tokenScale;\n"
            "bool finite = isfinite(part) && isfinite(hscale) && isfinite(inscale) && isfinite(v);\n"
            "device float* out = (device float*)yout;\n"
            "out[t * H + ch] = finite ? v : 0.0f;\n"
            "if (t == 0 && ch == 0) done[0] = 1;\n";
        // second output `done`: a 1-byte handle of this kernel; the next pack of this parity takes it as an input.
        auto fn = mx::fast::metal_kernel("tqllm_ane_join2", {"y", "tx", "rows", "hidden", "ystride"}, {"yout", "done"},
                                         source, "", false, false);
        const int n = curRows_[parity] ? curRows_[parity] : rows_;   // rows of the last pack into this parity
        auto outs = fn({yArr, txArr, scalarU32(uint32_t(n)), scalarU32(uint32_t(hidden_)), scalarU32(yStride)},
                       {{n, hidden_}, {1}}, {mx::float32, mx::int8}, {n, hidden_, 1}, {32, 8, 1}, {},
                       std::nullopt, false, mx::Device(mx::Device::gpu));
        joinDone_[parity] = outs[1];
        return outs[0];
    }

    // Fused gate/up path: decode the last `C` rows of a 4-bit-affine NaxAffine [F, H] straight to int8 + per-row fp16
    // scale in the slot's hidden-segment surfaces, without materialising the fp16 weights. One threadgroup per output
    // row r (of C), 256 threads: each thread decodes its strided columns of row r from the NAX codes exactly as
    // nax_affine_dequant would into a threadgroup float row; then the hidden columns are rotated by the input Hadamard
    // R in blocks of 128 (scalar natural-order FWHT + signs/sqrt(128), the same op order as rotateGateUpWeights, one
    // thread per 128-block); then the threadgroup reduces the row's peak over the rotated values and each thread writes
    // int8. One TG per row keeps the GPU busy (C threadgroups) and the 32x row-share of each code word is served from
    // the L2 cache. Numerics equal nax_affine_dequant + rotateGateUpWeights + quantizeRows(float) bit for bit: every
    // decoded value is the same half, the FWHT is the same scalar transform, and peak/round/scale follow quantizeRows.
    //
    // NAX layout for row r = blkF*32 + rr (blkF = brow + r/32, rr = r % 32; F - C and C are multiples of 512): a
    // column c decodes from ks = c/16; col16 = c & 15; fm = col16 & 7; jj = col16 >> 3; p = (rr & 15) & 3;
    // fn = (rr & 15) & 12; A = fn/4; lane = (A & 1) | ((fm & 3) << 1) | ((A & 2) << 2) | ((fm & 4) << 2) (inverse of
    // kSource); code nibble i = p + 4*jj of word.x (rr < 16) or word.y (rr >= 16); scale/bias at
    // S[(blkF*GPRF + g)*32 + rr], g = (ks*16)/GS.
    void decodeQuantizeRowsGpu(const NaxAffine& w, int brow, int C, const std::vector<std::pair<uint32_t, uint32_t>>& segs,
                               std::vector<Surface>& wsurf, Surface& ssurf, std::vector<mlx::core::array>& deps,
                               mlx::core::StreamOrDevice stream) {
        namespace mx = mlx::core;
        if (w.bits != 4 || w.q6) throw std::invalid_argument("ANE program: only 4-bit affine or TQ4 weights");
        const bool tq4 = w.tq4;   // TQ4: stored already rotated (128-blocks, TQ4 signs), value = level * fp16 scale
        const int H = w.columns, STEPSF = H / 16, GPRF = H / w.group;
        if (H % int(kInputBlock) != 0) throw std::invalid_argument("ANE program: hidden must be a multiple of 128");
        if (C % 4 != 0) throw std::invalid_argument("ANE program: C must be a multiple of 4");
        if (!signsArr_) {
            std::vector<float> sv(signs_.begin(), signs_.end());
            signsArr_ = mx::array(sv.data(), {int(kIntermediateBlock)}, mx::float32);
            mx::eval(*signsArr_);
        }
        std::vector<mx::array> segArrays;
        for (size_t k = 0; k < segs.size(); ++k)
            segArrays.push_back(surfaceArray(wsurf[k], {int(uint64_t(wsurf[k].strideBytes) * wsurf[k].rows)}, mx::int8));
        auto sArr = surfaceArray(ssurf, {int(uint64_t(ssurf.strideBytes) / 2 * ssurf.rows)}, mx::float16);

        // One SIMDGROUP per 4 consecutive output rows (a 32-bit NAX code word holds 4 rows x 2 columns, and the 32 scales
        // of a 32-row block are contiguous, so 4 rows share every word and scale load). Lane l owns columns 4l..4l+3 of
        // each 128-block for all 4 rows. The natural-order FWHT keeps rotateInputRow's stage order (len 1, 2 inside the
        // lane's registers, 4..64 by simd_shuffle_xor over the lane bits) with the same u+v / u-v operands, so the result
        // is bit-equal to the CPU bind; the row peak is a max reduction (order-free). Two passes recompute decode + FWHT
        // instead of keeping 4 rows x 5120 values: pass 1 finds the peaks, pass 2 quantizes and stores the 4 int8 of a
        // lane as one coalesced 32-bit store.
        const bool rotate = inputRotateEnabled() && !tq4;   // TQ4's stored basis IS the ANE input basis (see pack tq4)
        const std::string load =
            "      const uint ks = (gb << 3u) + (lane >> 2u);\n"
            "      const uint jj = (lane >> 1u) & 1u;\n"
            "      const uint g = (ks * 16u) / " + std::to_string(w.group) + "u;\n"
            "      const size_t sbi = ((size_t)blkF * " + std::to_string(GPRF) + "u + g) * 32u + rr0;\n"
            "      const uint2 sraw = *((const device uint2*)((const device ushort*)scales + sbi));\n"
            + (tq4 ? std::string(
            "      const float sc[4] = {float(as_type<half>(ushort(sraw.x & 0xFFFFu))), float(as_type<half>(ushort(sraw.x >> 16))),\n"
            "                           float(as_type<half>(ushort(sraw.y & 0xFFFFu))), float(as_type<half>(ushort(sraw.y >> 16)))};\n")
            : std::string(
            "      const uint2 braw = *((const device uint2*)((const device ushort*)biases + sbi));\n"
            "      const float sc[4] = {as_type<float>(sraw.x << 16), as_type<float>(sraw.x & 0xFFFF0000u),\n"
            "                           as_type<float>(sraw.y << 16), as_type<float>(sraw.y & 0xFFFF0000u)};\n"
            "      const float bi[4] = {as_type<float>(braw.x << 16), as_type<float>(braw.x & 0xFFFF0000u),\n"
            "                           as_type<float>(braw.y << 16), as_type<float>(braw.y & 0xFFFF0000u)};\n")) +
            "      for (uint r = 0u; r < 4u; ++r) {\n"
            "        const uint fm = ((lane << 2u) + r) & 7u;\n"
            "        const uint lanew = (A & 1u) | ((fm & 3u) << 1u) | ((A & 2u) << 2u) | ((fm & 4u) << 2u);\n"
            "        const uint word = ((const device uint*)words)[(((size_t)blkF * " + std::to_string(STEPSF) +
                "u + ks) * 32u + lanew) * 2u + wsel];\n"
            "        for (uint p = 0u; p < 4u; ++p) {\n"
            "          const uint code = (word >> (4u * (p + 4u * jj))) & 15u;\n"
            + (tq4 ?
            std::string("          v[p][r] = float(half(float(TQ4_LEVEL[code]) * sc[p]));\n") :
            std::string("          v[p][r] = float(half(fma(float(code), sc[p], bi[p])));\n")) +
            "        }\n"
            "      }\n";
        const std::string fwht = !rotate ? std::string("") : std::string(
            "      for (uint p = 0u; p < 4u; ++p) {\n"
            "        { float a = v[p][0] + v[p][1], b = v[p][0] - v[p][1]; v[p][0] = a; v[p][1] = b;\n"
            "          a = v[p][2] + v[p][3]; b = v[p][2] - v[p][3]; v[p][2] = a; v[p][3] = b; }\n"
            "        { float a = v[p][0] + v[p][2], b = v[p][0] - v[p][2]; v[p][0] = a; v[p][2] = b;\n"
            "          a = v[p][1] + v[p][3]; b = v[p][1] - v[p][3]; v[p][1] = a; v[p][3] = b; }\n"
            "        for (uint bit = 1u; bit < 32u; bit <<= 1u)\n"
            "          for (uint r = 0u; r < 4u; ++r) {\n"
            "            const float other = simd_shuffle_xor(v[p][r], ushort(bit));\n"
            "            v[p][r] = ((lane & bit) == 0u) ? (v[p][r] + other) : (other - v[p][r]);\n"
            "          }\n"
            "        for (uint r = 0u; r < 4u; ++r) v[p][r] = v[p][r] * so[r];\n"
            "      }\n");
        std::string store;
        for (size_t k = 0; k < segs.size(); ++k) {
            const std::string ks = std::to_string(k);
            store += std::string(k == 0 ? "        if" : "        else if") + " (gb * 128u >= " + std::to_string(segs[k].first) +
                     "u && gb * 128u < " + std::to_string(segs[k].first + segs[k].second) + "u) {\n"
                     "          *((device uint*)((device char*)w" + ks + " + (row0 + p) * " + std::to_string(wsurf[k].strideBytes) +
                     "u + (gb * 128u - " + std::to_string(segs[k].first) + "u) + (lane << 2u))) = packed;\n"
                     "        }\n";
        }
        const std::string source =
            "const uint lane = thread_index_in_simdgroup;\n"
            "const uint sgi = thread_position_in_grid.x >> 5u;          // 4-row group index\n"
            "const uint row0 = sgi * 4u;                                // first output row in [0, C)\n"
            "const uint NB = " + std::to_string(H / int(kInputBlock)) + "u;\n"
            "const uint blkF = " + std::to_string(brow) + "u + (row0 >> 5u);\n"
            "const uint rr0 = row0 & 31u;\n"
            "const uint A = (rr0 & 15u) >> 2u;\n"
            "const bool wsel = rr0 >= 16u;\n"
            "device const float* sgn = (device const float*)signs;\n"
            "float so[4];\n"
            "for (uint r = 0u; r < 4u; ++r) so[r] = sgn[(lane << 2u) + r] * (1.0f / sqrt(128.0f));\n"
            "float v[4][4];\n"
            "float peak[4] = {0x1p-14f, 0x1p-14f, 0x1p-14f, 0x1p-14f};\n"
            "for (uint gb = 0u; gb < NB; ++gb) {\n" + load + fwht +
            "      for (uint p = 0u; p < 4u; ++p)\n"
            "        for (uint r = 0u; r < 4u; ++r) peak[p] = max(peak[p], fabs(v[p][r]));\n"
            "}\n"
            "for (uint p = 0u; p < 4u; ++p) peak[p] = simd_max(peak[p]);\n"
            "for (uint gb = 0u; gb < NB; ++gb) {\n" + load + fwht +
            "      for (uint p = 0u; p < 4u; ++p) {\n"
            "        const float invPeak = 127.0f / peak[p];\n"
            "        uint packed = 0u;\n"
            "        for (uint r = 0u; r < 4u; ++r) {\n"
            "          int code8 = int(round(v[p][r] * invPeak)); code8 = max(-127, min(127, code8));\n"
            "          packed |= (uint(code8) & 255u) << (8u * r);\n"
            "        }\n" + store +
            "      }\n"
            "}\n"
            "if (lane < 4u) { device half* ss = (device half*)scale; ss[(row0 + lane) * " + std::to_string(ssurf.strideBytes / 2) +
            "u] = half(peak[lane] * (128.0f / 127.0f)); }\n";

        std::vector<std::string> names = {"words", "scales", "biases", "signs", "scale"};
        std::vector<mx::array> args = {w.words, w.scales, w.biases, *signsArr_, sArr};
        for (size_t k = 0; k < segs.size(); ++k) {
            names.push_back("w" + std::to_string(k));
            args.push_back(segArrays[k]);
        }
        auto fn = mx::fast::metal_kernel("tqllm_ane_decquant3_" + std::string(tq4 ? "tq4_" : rotate ? "rot_" : "raw_") +
                                         "g" + std::to_string(w.group) + "_" + std::to_string(segs.size()),
                                         names, {"dummy"}, source, tq4 ? kTq4Header : "", false, false);
        auto outs = fn(args, {{1}}, {mx::int8}, {(C / 4) * 32, 1, 1}, {128, 1, 1}, {}, std::nullopt, false, stream);
        deps.push_back(outs[0]);
    }

    // Fused down path: decode the last `C` columns of a 4-bit-affine NaxAffine [H, Fd] straight from the NAX codes,
    // rotate each 512-block by the intermediate Hadamard (FWHT), and quantize the rotated row to int8 + per-row fp16
    // scale into the slot's C-segment surfaces, in one kernel with no fp16/fp32 intermediate. One threadgroup per
    // hidden row (H), 512 threads: thread o owns lane o of every 512-block; for each block g it decodes column
    // (Fd - C) + g*512 + o of its row, runs the 9-stage shuffle/threadgroup FWHT, keeps the rotated value for that
    // block in a register, and tracks the row peak; then the threadgroup reduces the peak and each thread writes its
    // blocks' int8 codes. The decoded half equals nax_affine_dequant, the FWHT equals rotateDownWeights (same stage
    // order), and the per-row peak/round/scale equal quantizeRows, so the output is bit-equal to the CPU bind.
    // kSegment is a multiple of 512, so each 512-block lies wholly in one C-segment.
    void decodeRotateQuantizeDownGpu(const NaxAffine& w, int coff, int C,
                                     const std::vector<std::pair<uint32_t, uint32_t>>& segs, std::vector<Surface>& wsurf,
                                     Surface& ssurf, std::vector<mlx::core::array>& deps,
                                     mlx::core::StreamOrDevice stream) {
        namespace mx = mlx::core;
        if (w.bits != 4 || w.q6) throw std::invalid_argument("ANE program: only 4-bit affine or TQ4 weights");
        const bool tq4 = w.tq4;   // stored in 128-blocks rotated by TQ4's H*D: un-rotate (D*H), then the ANE's 512 rotation
        if (C % int(kIntermediateBlock) != 0) throw std::invalid_argument("ANE program: C must be a multiple of 512");
        if (coff % int(kIntermediateBlock) != 0) throw std::invalid_argument("ANE program: down offset must be a multiple of 512");
        const int BL = int(kIntermediateBlock), blocks = C / BL, H = w.rows;
        const int STEPSF = w.columns / 16, GPRF = w.columns / w.group;
        if (H % 4 != 0) throw std::invalid_argument("ANE program: hidden must be a multiple of 4");
        if (!signsArr_) {
            std::vector<float> sv(signs_.begin(), signs_.end());
            signsArr_ = mx::array(sv.data(), {int(kIntermediateBlock)}, mx::float32);
            mx::eval(*signsArr_);
        }
        for (int g = 0; g < blocks; ++g) {
            const uint32_t col0 = uint32_t(g) * BL;
            bool inside = false;
            for (const auto& [b, wd] : segs) inside = inside || (col0 >= b && col0 + BL <= b + wd);
            if (!inside) throw std::runtime_error("ANE program: a 512-block crosses a down C-segment boundary");
        }
        std::vector<mx::array> segArrays;
        for (size_t k = 0; k < segs.size(); ++k)
            segArrays.push_back(surfaceArray(wsurf[k], {int(uint64_t(wsurf[k].strideBytes) * wsurf[k].rows)}, mx::int8));
        auto sArr = surfaceArray(ssurf, {int(uint64_t(ssurf.strideBytes) / 2 * ssurf.rows)}, mx::float16);

        // One threadgroup of 4 simdgroups per 4 consecutive hidden rows (a NAX code word holds 4 rows x 2 columns, the 32
        // scales of a 32-row block are contiguous). Simdgroup sg owns columns 128 sg .. 128 sg + 127 of every 512-block,
        // lane l columns 4l..4l+3 of those, for all 4 rows. TQ4: the stored 128-blocks are un-rotated first (natural FWHT
        // inside the simdgroup, then the TQ4 sign and 1/sqrt(128)). The 512-point rotation then runs in the same stage order
        // as rotateDownWeights: len 1, 2 in registers, 4..64 by simd_shuffle_xor, 128 and 256 across the 4 simdgroups
        // through two ping-pong threadgroup buffers (u + v / u - v from the same operands, so the result is bit-equal to
        // the CPU bind). Two passes recompute decode + rotation: pass 1 reduces the 4 row peaks, pass 2 quantizes and
        // stores 4 int8 per lane per row as one 32-bit store.
        const std::string decode =
            "    {\n"
            "      const uint ks = " + std::to_string(coff / 16) + "u + g * 32u + sg * 8u + (lane >> 2u);\n"
            "      const uint jj = (lane >> 1u) & 1u;\n"
            "      const uint gi = (ks * 16u) / " + std::to_string(w.group) + "u;\n"
            "      const size_t sbi = ((size_t)blkF * " + std::to_string(GPRF) + "u + gi) * 32u + rr0;\n"
            "      const uint2 sraw = *((const device uint2*)((const device ushort*)scales + sbi));\n"
            + (tq4 ? std::string(
            "      const float sc[4] = {float(as_type<half>(ushort(sraw.x & 0xFFFFu))), float(as_type<half>(ushort(sraw.x >> 16))),\n"
            "                           float(as_type<half>(ushort(sraw.y & 0xFFFFu))), float(as_type<half>(ushort(sraw.y >> 16)))};\n")
            : std::string(
            "      const uint2 braw = *((const device uint2*)((const device ushort*)biases + sbi));\n"
            "      const float sc[4] = {as_type<float>(sraw.x << 16), as_type<float>(sraw.x & 0xFFFF0000u),\n"
            "                           as_type<float>(sraw.y << 16), as_type<float>(sraw.y & 0xFFFF0000u)};\n"
            "      const float bi[4] = {as_type<float>(braw.x << 16), as_type<float>(braw.x & 0xFFFF0000u),\n"
            "                           as_type<float>(braw.y << 16), as_type<float>(braw.y & 0xFFFF0000u)};\n")) +
            "      for (uint e = 0u; e < 4u; ++e) {\n"
            "        const uint fm = ((lane << 2u) + e) & 7u;\n"
            "        const uint lanew = (A & 1u) | ((fm & 3u) << 1u) | ((A & 2u) << 2u) | ((fm & 4u) << 2u);\n"
            "        const uint word = ((const device uint*)words)[(((size_t)blkF * " + std::to_string(STEPSF) +
                "u + ks) * 32u + lanew) * 2u + wsel];\n"
            "        for (uint p = 0u; p < 4u; ++p) {\n"
            "          const uint code = (word >> (4u * (p + 4u * jj))) & 15u;\n"
            + (tq4 ?
            std::string("          v[p][e] = float(half(float(TQ4_LEVEL[code]) * sc[p]));\n") :
            std::string("          v[p][e] = float(half(fma(float(code), sc[p], bi[p])));\n")) +
            "        }\n"
            "      }\n"
            "    }\n";
        const std::string fwht128 =
            "    for (uint p = 0u; p < 4u; ++p) {\n"
            "      { float a = v[p][0] + v[p][1], b = v[p][0] - v[p][1]; v[p][0] = a; v[p][1] = b;\n"
            "        a = v[p][2] + v[p][3]; b = v[p][2] - v[p][3]; v[p][2] = a; v[p][3] = b; }\n"
            "      { float a = v[p][0] + v[p][2], b = v[p][0] - v[p][2]; v[p][0] = a; v[p][2] = b;\n"
            "        a = v[p][1] + v[p][3]; b = v[p][1] - v[p][3]; v[p][1] = a; v[p][3] = b; }\n"
            "      for (uint bit = 1u; bit < 32u; bit <<= 1u)\n"
            "        for (uint e = 0u; e < 4u; ++e) {\n"
            "          const float other = simd_shuffle_xor(v[p][e], ushort(bit));\n"
            "          v[p][e] = ((lane & bit) == 0u) ? (v[p][e] + other) : (other - v[p][e]);\n"
            "        }\n"
            "    }\n";
        const std::string tq4Unrotate = !tq4 ? std::string("") : fwht128 + std::string(
            "    for (uint e = 0u; e < 4u; ++e) {\n"
            "      const uint i7 = (lane << 2u) + e;\n"
            "      const float sgnv = (((TW[i7 >> 5u] >> (i7 & 31u)) & 1u) ? -1.0f : 1.0f) * 0.08838834764831845f;\n"
            "      for (uint p = 0u; p < 4u; ++p) v[p][e] = v[p][e] * sgnv;\n"
            "    }\n");
        // stages 128 and 256 across the four simdgroups
        const std::string cross =
            "    for (uint stage = 0u; stage < 2u; ++stage) {\n"
            "      threadgroup float* buf = stage == 0u ? xA : xB;\n"
            "      for (uint p = 0u; p < 4u; ++p) for (uint e = 0u; e < 4u; ++e) buf[(p * 4u + e) * 128u + tid] = v[p][e];\n"
            "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "      const uint partner = (sg ^ (1u << stage)) * 32u + lane;\n"
            "      const bool low = ((sg >> stage) & 1u) == 0u;\n"
            "      for (uint p = 0u; p < 4u; ++p) for (uint e = 0u; e < 4u; ++e) {\n"
            "        const float other = buf[(p * 4u + e) * 128u + partner];\n"
            "        v[p][e] = low ? (v[p][e] + other) : (other - v[p][e]);\n"
            "      }\n"
            "    }\n";
        const std::string block =
            decode + tq4Unrotate + fwht128 + cross +
            "    for (uint e = 0u; e < 4u; ++e) for (uint p = 0u; p < 4u; ++p) v[p][e] = v[p][e] * so[e];\n";
        std::string store;
        for (size_t k = 0; k < segs.size(); ++k) {
            const std::string ks = std::to_string(k);
            store += std::string(k == 0 ? "        if" : "        else if") + " (g * 512u >= " + std::to_string(segs[k].first) +
                     "u && g * 512u < " + std::to_string(segs[k].first + segs[k].second) + "u) {\n"
                     "          *((device uint*)((device char*)w" + ks + " + (row0 + p) * " + std::to_string(wsurf[k].strideBytes) +
                     "u + (g * 512u - " + std::to_string(segs[k].first) + "u) + sg * 128u + (lane << 2u))) = packed;\n"
                     "        }\n";
        }
        const std::string source =
            "const uint lane = thread_index_in_simdgroup;\n"
            "const uint sg = simdgroup_index_in_threadgroup;\n"
            "const uint tid = sg * 32u + lane;\n"
            "const uint row0 = threadgroup_position_in_grid.x * 4u;      // first hidden row of this group\n"
            "const uint blocks = " + std::to_string(blocks) + "u;\n"
            "const uint blkF = row0 >> 5u;\n"
            "const uint rr0 = row0 & 31u;\n"
            "const uint A = (rr0 & 15u) >> 2u;\n"
            "const bool wsel = rr0 >= 16u;\n"
            "constexpr uint TW[4] = {0xE46A0359u, 0x42F85949u, 0xA4C63BBBu, 0x49F250A9u};\n"
            "device const float* sgn = (device const float*)signs;\n"
            "float so[4];\n"
            "for (uint e = 0u; e < 4u; ++e) so[e] = sgn[sg * 128u + (lane << 2u) + e] * (1.0f / sqrt(512.0f));\n"
            "threadgroup float xA[16 * 128];\n"
            "threadgroup float xB[16 * 128];\n"
            "threadgroup float pk[16];\n"
            "float v[4][4];\n"
            "float peak[4] = {0x1p-14f, 0x1p-14f, 0x1p-14f, 0x1p-14f};\n"
            "for (uint g = 0u; g < blocks; ++g) {\n" + block +
            "    for (uint p = 0u; p < 4u; ++p) for (uint e = 0u; e < 4u; ++e) peak[p] = max(peak[p], fabs(v[p][e]));\n"
            "}\n"
            "for (uint p = 0u; p < 4u; ++p) peak[p] = simd_max(peak[p]);\n"
            "if (lane == 0u) for (uint p = 0u; p < 4u; ++p) pk[sg * 4u + p] = peak[p];\n"
            "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "for (uint p = 0u; p < 4u; ++p) peak[p] = max(max(pk[p], pk[4u + p]), max(pk[8u + p], pk[12u + p]));\n"
            "for (uint g = 0u; g < blocks; ++g) {\n" + block +
            "    for (uint p = 0u; p < 4u; ++p) {\n"
            "      const float invPeak = 127.0f / peak[p];\n"
            "      uint packed = 0u;\n"
            "      for (uint e = 0u; e < 4u; ++e) {\n"
            "        int code8 = int(round(v[p][e] * invPeak)); code8 = max(-127, min(127, code8));\n"
            "        packed |= (uint(code8) & 255u) << (8u * e);\n"
            "      }\n" + store +
            "    }\n"
            "}\n"
            "if (sg == 0u && lane < 4u) { device half* ss = (device half*)scale; ss[(row0 + lane) * " +
            std::to_string(ssurf.strideBytes / 2) + "u] = half(peak[lane] * (128.0f / 127.0f)); }\n";

        std::vector<std::string> names = {"words", "scales", "biases", "signs", "scale"};
        std::vector<mx::array> args = {w.words, w.scales, w.biases, *signsArr_, sArr};
        for (size_t k = 0; k < segs.size(); ++k) {
            names.push_back("w" + std::to_string(k));
            args.push_back(segArrays[k]);
        }
        auto fn = mx::fast::metal_kernel("tqllm_ane_decrotquant_down2_" + std::string(tq4 ? "tq4_" : "") + "g" +
                                         std::to_string(w.group) + "_" + std::to_string(segs.size()), names, {"dummy"},
                                         source, tq4 ? kTq4Header : "", false, false);
        auto outs = fn(args, {{1}}, {mx::int8}, {(H / 4) * 128, 1, 1}, {128, 1, 1}, {}, std::nullopt, false, stream);
        deps.push_back(outs[0]);
    }

    mlx::core::array scalarU32(uint32_t v) {
        namespace mx = mlx::core;
        return mx::array(&v, {1}, mx::uint32);
    }

    // Bind the slot's weight surfaces + the shared x/tx/y surfaces to a request and run one evaluation, started at once
    // (the wait event is pre-signalled; the caller evaluated the pack first). Non-blocking: records the completion
    // state for runWait(). (Mirrors runEvaluation but for the GPU path and returns without waiting.)
    void runEnqueue(int slot, int parity, bool useReady) {
        @autoreleasepool {
            SlotSurfaces& ss = slotWeights_[slot];
            const auto surfaceFor = [&](const std::string& name) -> const Surface& {
                if (name == "tx") return txOf(parity);
                if (name == "sg") return ss.sg;
                if (name == "su") return ss.su;
                if (name == "sd") return ss.sd;
                if (name.size() >= 2 && name[0] == 'x') return xOf(parity)[std::stoul(name.substr(1))];
                if (name.size() >= 3 && name[0] == 'w' && name[1] == 'g') return ss.wg[std::stoul(name.substr(2))];
                if (name.size() >= 3 && name[0] == 'w' && name[1] == 'u') return ss.wu[std::stoul(name.substr(2))];
                if (name.size() >= 3 && name[0] == 'w' && name[1] == 'd') return ss.wd[std::stoul(name.substr(2))];
                throw std::runtime_error("ANE program has an unexpected input " + name);
            };
            const Variant& var = variantFor(uint32_t(curRows_[parity]));   // the function of the packed slice's rows
            NSMutableArray* objects = [NSMutableArray arrayWithCapacity:var.inputNames.size()];
            NSMutableArray* indices = [NSMutableArray arrayWithCapacity:var.inputNames.size()];
            const auto object = [&](const Surface& s) {
                id result = [api_.surface objectWithIOSurface:(IOSurfaceRef)s.surface.get()];
                if (!result) fail("surface object creation", nil);
                return result;
            };
            for (size_t i = 0; i < var.inputNames.size(); ++i) {
                [objects addObject:object(surfaceFor(var.inputNames[i]))];
                [indices addObject:@(var.inputSymbols[i])];
            }
            NSArray* outputs = @[ object(yOf(parity)) ];
            NSArray* outputIndices = @[ @(var.outputSymbol) ];

            // The ANE signals our own event_ when done (runWait blocks on the completion handler). It waits either on
            // the event MLX signals after the pack's command buffer (useReady: the GPU ordered the pack, the CPU did
            // not wait), or on event_ pre-signalled here (the caller evaluated the pack first).
            Pending& pd = pend_[parity];
            if (pd.done) throw std::runtime_error("ANE program: parity already has a request in flight");
            id native = (id)event_;
            id waitEvent;
            if (useReady) {
                if (!ready_) throw std::runtime_error("ANE program: enqueue(after_pack) without ready_after()");
                auto* mtl = ready_->cast<mlx::core::metal::EventImpl>().mtl_event();  // MTL::SharedEvent*
                id<MTLSharedEvent> mlxShared = (__bridge id<MTLSharedEvent>)(void*)mtl;
                waitEvent = [api_.waitEvent waitEventWithValue:ready_->value() sharedEvent:mlxShared eventType:0];
                pd.ready = std::move(ready_);
                ready_.reset();
            } else {
                const uint64_t wait = ++eventValue_;
                [event_ setSignaledValue:wait];
                waitEvent = [api_.waitEvent waitEventWithValue:wait sharedEvent:native eventType:0];
            }
#if MLX_VERSION_NUMERIC >= 32004
            if (!aneDone_) aneDone_ = [device_ newSharedEvent];
            id<MTLSharedEvent> doneShared = aneDone_;
#else
            if (!aneDone_) aneDone_.emplace(mlx::core::to_stream(mlx::core::Device(mlx::core::Device::gpu)));
            id<MTLSharedEvent> doneShared =
                (__bridge id<MTLSharedEvent>)(void*)aneDone_->cast<mlx::core::metal::EventImpl>().mtl_event();
#endif
            const uint64_t signal = ++aneDoneValue_;
            id signalEvent = [api_.signalEvent signalEventWithValue:signal symbolIndex:0 eventType:0 sharedEvent:doneShared];
            if (!signalEvent || !waitEvent) fail("event creation", nil);
            id events = [api_.events sharedEventsWithSignalEvents:@[ signalEvent ] waitEvents:@[ waitEvent ]];
            if (!events) fail("event creation", nil);
            id request = [api_.request requestWithInputs:objects
                                            inputIndices:indices
                                                 outputs:outputs
                                           outputIndices:outputIndices
                                           weightsBuffer:nil
                                               perfStats:nil
                                          procedureIndex:@(var.procedure)
                                            sharedEvents:events
                                       transactionHandle:nil];
            if (!request) fail("request creation", nil);

            pd.done = std::make_shared<std::pair<std::mutex, std::condition_variable>>();
            pd.state = std::make_shared<std::pair<bool, bool>>(false, false);
            auto done = pd.done;
            auto state = pd.state;
            pd.doneValue = signal;
            void (^report)(BOOL, NSError*) = ^(BOOL success, NSError* cerr) {
                if (!success && cerr) std::fprintf(stderr, "ANE_PROGRAM evaluation error: %s\n", cerr.description.UTF8String);
                // a failed request does not signal: raise the value here so a GPU wait on it passes (the CPU still sees
                // the failure in wait()). The event never decreases, so a late ANE signal is harmless.
                if (!success && doneShared.signaledValue < signal) doneShared.signaledValue = signal;
                std::lock_guard lock(done->first);
                state->first = true;
                state->second = success;
                done->second.notify_all();
            };
            [request setCompletionHandler:report];
            pd.start = Clock::now();
            NSError* error = nil;
            if (![api_.client evaluateWithModel:model_ options:@{} request:request qos:kQos error:&error]) {
                if (doneShared.signaledValue < signal) doneShared.signaledValue = signal;
                pd = Pending{};
                fail("evaluation", error);
            }
        }
    }

    void runWait(int parity) {
        Pending& pd = pend_[parity];
        if (!pd.done) throw std::runtime_error("ANE program: wait() without enqueue()");
        {
            std::unique_lock lock(pd.done->first);
            pd.done->second.wait(lock, [&] { return pd.state->first; });
        }
        lastEvalMs_ = std::chrono::duration<double, std::milli>(Clock::now() - pd.start).count();
        const bool ok = pd.state->second;
        pd = Pending{};  // releases the ready event the request waited on
        if (!ok) throw std::runtime_error("ANE evaluation reported failure");
    }

    int columns_, rows_, hidden_, layers_;
    const Api& api_;
    id<MTLDevice> device_ = nil;
    id<MTLSharedEvent> event_ = nil;
    uint64_t eventValue_ = 0;
    std::vector<std::pair<uint32_t, uint32_t>> inSegs_, downSegs_;
    ProgramSurfaces surf_;
    __strong id model_ = nil;
    std::vector<std::string> inputNames_;
    std::vector<uint32_t> inputSymbols_;
    uint32_t outputSymbol_ = 0;
    uint32_t procedure_ = 0;  // the maximum-rows function (the CPU path's)
    // one entry per compiled function (row count); the GPU path binds the one fitting the packed slice
    struct Variant {
        uint32_t rows = 0, procedure = 0, outputSymbol = 0;
        std::vector<std::string> inputNames;
        std::vector<uint32_t> inputSymbols;
    };
    std::vector<Variant> variants_;
    std::vector<uint32_t> rowCounts_;
    std::array<int, 2> curRows_{0, 0};  // rows packed into each parity's surfaces by the last pack_input_gpu
    const Variant& variantFor(uint32_t n) const {
        for (const Variant& v : variants_) if (v.rows >= n) return v;
        throw std::invalid_argument("ANE program: no function for " + std::to_string(n) + " rows");
    }
    std::array<float, kIntermediateBlock> signs_{};
    int stagedLayer_ = -1;
    double lastEvalMs_ = 0;
    std::vector<LayerWeights> weights_;

    // ---- step 2: GPU-path state ----
    // Per-slot weight surfaces (double-buffered). Separate from the step-1 weights_ (CPU path).
    struct SlotSurfaces final {
        std::vector<Surface> wg, wu, wd;
        Surface sg, su, sd;
        bool staged = false;
    };
    std::vector<SlotSurfaces> slotWeights_;
    std::optional<mlx::core::array> signsArr_;  // rotation signs on the GPU, lazy
    bool inputTq4_ = false;  // the last requantized layer is TQ4: packs rotate x in TQ4's basis
    // Second set of activation surfaces (x segments, tx, y) for slice parity 1, so slice k's ANE reads/writes one set
    // while slice k-1's join and slice k+1's pack use the other (Splash double-buffers its slices the same way).
    // Parity 0 is surf_'s set. Created in reserve_slots (the GPU path only).
    struct ActSurfaces final {
        std::vector<Surface> x;
        Surface tx, y;
    };
    ActSurfaces alt_;
    bool altReady_ = false;
    std::vector<Surface>& xOf(int p) { return p ? alt_.x : surf_.x; }
    Surface& txOf(int p) { return p ? alt_.tx : surf_.tx; }
    Surface& yOf(int p) { return p ? alt_.y : surf_.y; }
    // The event MLX signals at the end of the command buffer that ran the last ready_after() pack (eval_impl attaches
    // one per stream and signals it after the buffer's work). The next enqueue(after_pack=true) makes the ANE wait on
    // it, so the ANE starts once the GPU finished the pack, without the CPU waiting.
    std::optional<mlx::core::Event> ready_;
    // A small output of each parity's last join kernel: the next pack into that parity takes it as an input, so MLX
    // orders the pack after the join (the surfaces are written through input wrappers MLX does not track).
    std::array<std::optional<mlx::core::array>, 2> joinDone_;
    // One in-flight ANE request per parity.
    struct Pending final {
        std::shared_ptr<std::pair<std::mutex, std::condition_variable>> done;
        std::shared_ptr<std::pair<bool, bool>> state;  // (finished, succeeded)
        Clock::time_point start{};
        std::optional<mlx::core::Event> ready;          // kept alive until the request completes
        uint64_t doneValue = 0;                         // value the request raises aneDone_ to
    };
    std::array<Pending, 2> pend_;
    // The ANE signals this persistent MLX event (one MTLSharedEvent) to a fresh value per request, so the GPU can wait
    // for an evaluation on its own stream (wait_gpu) instead of the CPU blocking: Splash's graph.wait(done).
#if MLX_VERSION_NUMERIC >= 32004
    id<MTLSharedEvent> aneDone_ = nil;  // on MLX's Metal device
#else
    std::optional<mlx::core::Event> aneDone_;
#endif
    uint64_t aneDoneValue_ = 0;
};

}  // namespace

void ane_program_set_input_rotate(bool on) { gInputRotate.store(on); }

std::unique_ptr<AneProgram> AneProgram::open(int columns, int rows, int hidden, int layers) {
    return std::make_unique<NativeAneProgram>(columns, rows, hidden, layers);
}

}  // namespace tqllm::mlx_backend
