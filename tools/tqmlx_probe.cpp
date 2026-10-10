#include "qwen35.hpp"
#include "tqllm/tokenizer.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace mx=mlx::core;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
int main(int argc,char** argv) {
    try {
        std::string model,prompt,tokens_path,logits_path;
        int count=32,repeat=1,warmup=0,chunk=64,context=32768;
        bool ignore_eos=false,all_logits=false;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            auto value=[&]() {if(++i==argc)throw std::invalid_argument("missing argument value");return std::string(argv[i]);};
            if(arg=="--model") model=value();
            else if(arg=="--prompt") prompt=value();
            else if(arg=="--tokens-file") tokens_path=value();
            else if(arg=="--logits") logits_path=value();
            else if(arg=="--generate") count=std::stoi(value());
            else if(arg=="--repeat") repeat=std::stoi(value());
            else if(arg=="--warmup") warmup=std::stoi(value());
            else if(arg=="--prefill-chunk") chunk=std::stoi(value());
            else if(arg=="--context") context=std::stoi(value());
            else if(arg=="--ignore-eos") ignore_eos=true;
            else if(arg=="--all-logits") all_logits=true;
            else throw std::invalid_argument("unknown argument: "+arg);
        }
        if(model.empty()||count<1||repeat<1||warmup<0||chunk<1||context<1)
            throw std::invalid_argument("usage: tqmlx-probe --model file.gguf --prompt text [--generate N --repeat N --warmup N]");
        tqllm::Tokenizer tokenizer;tokenizer.load(model);
        std::vector<uint32_t> input;
        if(tokens_path.empty()) input=tokenizer.encode(prompt);
        else {std::ifstream in(tokens_path);uint32_t token;while(in>>token)input.push_back(token);if(!in.eof())throw std::runtime_error("invalid tokens file");}
        if(input.empty()||input.size()+size_t(count)>size_t(context))throw std::invalid_argument("prompt or output exceeds context");
        auto started=Clock::now();tqllm::mlx_backend::Qwen35 engine(model,context);
        std::cerr<<"MLX_MODEL_LOADED seconds="<<elapsed(started)<<" vocabulary="<<engine.vocabulary()<<'\n';
        std::vector<uint32_t> baseline;
        for(int run=-warmup;run<repeat;++run) {
            engine.reset();started=Clock::now();std::optional<mx::array> logits;
            for(size_t p=0;p<input.size();p+=size_t(chunk)) {
                const size_t n=std::min(size_t(chunk),input.size()-p);
                auto y=engine.forward({input.data()+p,n},all_logits);
                if(all_logits && !logits_path.empty()) {
                    std::ofstream out(logits_path,(p?std::ios::app:std::ios::trunc)|std::ios::binary);
                    out.write(reinterpret_cast<const char*>(y.data<float>()),std::streamsize(y.size()*sizeof(float)));
                    if(!out)throw std::runtime_error("cannot write logits file");
                }
                logits=mx::reshape(mx::slice(y,{y.shape(0)-1,0},{y.shape(0),y.shape(1)}),{y.shape(1)});
            }
            const double prefill=elapsed(started);
            uint32_t next=mx::argmax(*logits).item<uint32_t>();
            std::vector<uint32_t> output={next};
            const auto first=Clock::now();
            for(int i=1;i<count && (ignore_eos || next!=tokenizer.eos_id());++i) {
                logits=engine.forward({&next,1});next=mx::argmax(*logits).item<uint32_t>();output.push_back(next);
            }
            const double decode=elapsed(first);
            if(run<0)continue;
            if(baseline.empty())baseline=output;
            if(output!=baseline)throw std::runtime_error("reset changed greedy output across repeats");
            std::cout<<"{\"kind\":\"native_mlx\",\"run\":"<<run<<",\"prompt_tokens\":"<<input.size()
                     <<",\"output_tokens\":"<<output.size()<<",\"prefill_tok_s\":"<<input.size()/prefill
                     <<",\"ttft_ms\":"<<prefill*1000<<",\"decode_tok_s\":"<<(output.size()-1)/decode<<",\"token_ids\":[";
            for(size_t i=0;i<output.size();++i)std::cout<<(i?",":"")<<output[i];
            std::cout<<"]}"<<std::endl;
            std::cerr<<"GENERATED_TEXT "<<tokenizer.decode(output)<<'\n';
        }
    } catch(const std::exception& e) {std::cerr<<"MLX_PROBE_ERROR "<<e.what()<<'\n';return 1;}
}
