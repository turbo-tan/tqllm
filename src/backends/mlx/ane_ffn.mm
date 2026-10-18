#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include "ane_ffn.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace tqllm::mlx_backend {
namespace {
class CoreMlFfn final : public AneFfn {
public:
    CoreMlFfn(const std::string& dir,int layers) {
        // dir/ane_ffn.txt: "<rows> <columns>"
        std::ifstream meta(dir+"/ane_ffn.txt");
        if(!(meta>>rows_>>columns_) || rows_<1 || columns_<1)throw std::runtime_error("ANE FFN: missing or bad "+dir+"/ane_ffn.txt");
        models_.resize(layers);
        MLModelConfiguration* cfg=[MLModelConfiguration new];
        cfg.computeUnits=MLComputeUnitsCPUAndNeuralEngine;
        int loaded=0;
        for(int l=0;l<layers;++l) {
            NSString* path=[NSString stringWithFormat:@"%s/ffn_l%d.mlmodelc",dir.c_str(),l];
            if(![[NSFileManager defaultManager] fileExistsAtPath:path])continue;
            NSError* err=nil;
            MLModel* m=[MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path] configuration:cfg error:&err];
            if(!m)throw std::runtime_error(std::string("ANE FFN: load failed for layer ")+std::to_string(l)+": "+err.localizedDescription.UTF8String);
            models_[l]=m;++loaded;
        }
        std::fprintf(stderr,"ANE_FFN loaded %d of %d layers, rows %d, columns %d\n",loaded,layers,rows_,columns_);
    }
    bool has(int layer) const override {return layer>=0 && layer<int(models_.size()) && models_[layer]!=nil;}
    int rows() const override {return rows_;}
    int columns() const override {return columns_;}
    void predict(int layer,const void* x,void* y,int hidden) override {
        @autoreleasepool {
            MLModel* m=models_.at(layer);
            NSError* err=nil;
            NSArray* shape=@[@(rows_),@(hidden)],*strides=@[@(hidden),@1];
            MLMultiArray* in=[[MLMultiArray alloc] initWithDataPointer:const_cast<void*>(x) shape:shape
                dataType:MLMultiArrayDataTypeFloat16 strides:strides deallocator:nil error:&err];
            MLMultiArray* out=[[MLMultiArray alloc] initWithDataPointer:y shape:shape
                dataType:MLMultiArrayDataTypeFloat16 strides:strides deallocator:nil error:&err];
            if(!in || !out)throw std::runtime_error("ANE FFN: buffer wrap failed");
            NSString* in_name=m.modelDescription.inputDescriptionsByName.allKeys.firstObject;
            NSString* out_name=m.modelDescription.outputDescriptionsByName.allKeys.firstObject;
            MLDictionaryFeatureProvider* features=[[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{in_name:[MLFeatureValue featureValueWithMultiArray:in]} error:&err];
            MLPredictionOptions* opts=[MLPredictionOptions new];
            opts.outputBackings=@{out_name:out};
            id<MLFeatureProvider> r=[m predictionFromFeatures:features options:opts error:&err];
            if(!r)throw std::runtime_error(std::string("ANE FFN: predict failed: ")+err.localizedDescription.UTF8String);
            // Core ML may ignore the backing and return its own buffer; copy in that case
            MLMultiArray* got=[r featureValueForName:out_name].multiArrayValue;
            if(got && got.dataPointer!=y) {
                if(got.dataType!=MLMultiArrayDataTypeFloat16 || got.count!=NSInteger(rows_)*hidden)
                    throw std::runtime_error("ANE FFN: unexpected output type or size");
                [got getBytesWithHandler:^(const void* bytes,NSInteger size){std::memcpy(y,bytes,size_t(size));}];
            }
        }
    }
private:
    int rows_=0,columns_=0;
    std::vector<MLModel*> models_;
};
}
std::unique_ptr<AneFfn> AneFfn::open(const std::string& dir,int layers) {return std::make_unique<CoreMlFfn>(dir,layers);}
}
