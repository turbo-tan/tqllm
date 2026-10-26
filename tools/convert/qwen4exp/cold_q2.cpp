#include "cold_expert_store.hpp"
#include "hot_cold_policy.hpp"
#include "gsq_q2_host.hpp"
#include "gguf_mapped.hpp"
#include <chrono>
#include <iostream>
#include <memory>

int main(int argc,char ** argv) {
    try {
        if (argc==4 && std::string(argv[1])=="--fast-checksums") {
            const std::string root=argv[2]; size_t pos=0;const auto piece=std::stoull(argv[3],&pos);
            if(pos!=std::string(argv[3]).size())throw std::runtime_error("invalid piece size");
            std::ifstream identity(std::filesystem::path(root)/"identity",std::ios::binary);
            const std::string id{std::istreambuf_iterator<char>(identity),{}};
            tqllm::ColdExpertStore store(root,piece,id);size_t n=0;
            for(const auto & file:std::filesystem::directory_iterator(root)) {
                if(file.path().extension()!=".q2")continue;
                const auto name=file.path().stem().string();const auto dash=name.find('-');
                if(dash==std::string::npos)throw std::runtime_error("invalid cold record name");
                size_t a=0,b=0;const int layer=std::stoi(name.substr(0,dash),&a),expert=std::stoi(name.substr(dash+1),&b);
                if(a!=dash || b!=name.size()-dash-1)throw std::runtime_error("invalid cold record name");
                store.prepare_fast(layer,expert);++n;
                if(n%512==0)std::cout<<"FAST_PROGRESS records="<<n<<std::endl;
            }
            if(!n)throw std::runtime_error("no cold records");
            store.finish_fast();std::cout<<"FAST_DONE records="<<n<<std::endl;return 0;
        }
        if (argc < 5) throw std::runtime_error("usage: cold-q2 MAP OUTPUT MAX_NEW_EXPERTS GGUF [GGUF ...]; 0 = all remaining");
        const std::string map_path=argv[1], output=argv[2];
        if (argv[3][0] == '-') throw std::runtime_error("invalid negative limit");
        size_t pos=0; const auto limit=std::stoull(argv[3],&pos);
        if (pos!=std::string(argv[3]).size()) throw std::runtime_error("invalid limit");
        constexpr int layers=48, experts=512, hidden=2560, ff=640;
        const size_t count=size_t(hidden)*ff, piece=count/64*18;
        std::vector<int32_t> map(layers*experts); std::ifstream mf(map_path,std::ios::binary);
        if (!mf.read(reinterpret_cast<char *>(map.data()),map.size()*4) || mf.peek()!=EOF)
            throw std::runtime_error("precision map size mismatch");
        tqllm::validate_hot_map(map,layers,experts);
        std::vector<std::string> sources; std::vector<std::unique_ptr<tqllm::MappedGguf>> ggufs;
        for (int i=4;i<argc;++i) { sources.emplace_back(argv[i]); ggufs.push_back(std::make_unique<tqllm::MappedGguf>(argv[i],tqllm::GgufDialect::GsqRco)); }
        tqllm::ColdExpertStore store(output,piece,tqllm::ColdExpertStore::identity(sources,map,piece),true);
        auto start=std::chrono::steady_clock::now(); size_t written=0, existing=0;
        for (int l=0;l<layers;++l) for (int e=0;e<experts;++e) {
            if (tqllm::expert_is_hot(map,l,experts,e)) continue;
            if (store.contains(l,e)) { store.load(l,e); ++existing; continue; }
            if (limit && written>=limit) { std::cout<<"PILOT_DONE new="<<written<<" existing="<<existing<<std::endl; return 0; }
            if (std::filesystem::space(output).available < 3*piece+(size_t(2)<<30)) throw std::runtime_error("disk reserve below 2 GiB");
            std::array<std::vector<char>,3> raw;
            for (int p=0;p<3;++p) {
                const std::string name="blk."+std::to_string(l)+".ffn_"+(p==0?"gate":p==1?"up":"down")+"_exps.weight";
                tqllm::MappedGguf * source=nullptr;
                for (auto & g:ggufs) if (g->file().tensors.count(name)) {
                    if(source) throw std::runtime_error("duplicate expert tensor"); source=g.get();
                }
                if (!source) throw std::runtime_error("missing tensor "+name);
                const auto & t=source->file().tensors.at(name);
                if (t.ne!=std::vector<uint64_t>{uint64_t(p==2?ff:hidden),uint64_t(p==2?hidden:ff),experts})
                    throw std::runtime_error("expert tensor shape mismatch");
                auto view=source->expert(name,e); uint64_t block=0,bytes=0;
                if (!tqllm::gguf_block_layout(t.type,block,bytes) || !block || count%block || count/block*bytes!=view.size)
                    throw std::runtime_error("expert layout mismatch");
                raw[p]=tqllm::requantize_gsq_q2_parallel(t.type,view.data,count,block,bytes,4);
            }
            store.save(l,e,raw); ++written;
            if(written%16==0 || written==1) {
                const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                std::cout<<"PROGRESS layer="<<l<<" expert="<<e<<" new="<<written<<" seconds="<<seconds<<" experts_per_second="<<written/seconds<<std::endl;
            }
        }
        store.finish(); std::cout<<"CONVERSION_DONE new="<<written<<" existing="<<existing<<std::endl;
    } catch (const std::exception & e) { std::cerr<<"cold-q2: "<<e.what()<<std::endl; return 1; }
}
