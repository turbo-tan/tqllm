#include "cold_expert_store.hpp"
#include "cold_expert_memo.hpp"
#include <cassert>
#include <iostream>
template<class F> void rejects(F f) { bool hit=false; try { f(); } catch(const std::exception &) { hit=true; } assert(hit); }
int main() {
    const auto root=std::filesystem::temp_directory_path()/("cold-store-test-"+std::to_string(getpid()));
    std::filesystem::create_directories(root);
    struct Cleanup { std::filesystem::path p; ~Cleanup(){std::filesystem::remove_all(p);} } cleanup{root};
    tqllm::ColdExpertStore writer(root.string(),18,"source-map-v1",true);
    std::array<std::vector<char>,3> raw; for(int i=0;i<3;++i)raw[i]=std::vector<char>(18,char(i+1));
    writer.save(1,2,raw); assert(writer.load(1,2)==raw);
    rejects([&]{tqllm::ColdExpertStore r(root.string(),18,"source-map-v1");});
    rejects([&]{tqllm::ColdExpertStore r(root.string(),18,"different",true);});
    writer.finish(); tqllm::ColdExpertStore reader(root.string(),18,"source-map-v1");
    reader.profile(true);
    assert(reader.load(1,2)==raw);
    assert(reader.stats().loads==1 && reader.stats().bytes==54);
    assert(reader.stats().read_ns>0 && reader.stats().checksum_ns>0);
    reader.profile(false); assert(reader.load(1,2)==raw); assert(reader.stats().loads==1);
    rejects([&]{reader.load(1,3);});
    std::filesystem::copy_file(writer.path(1,2),writer.path(1,3)); rejects([&]{reader.load(1,3);});
    { std::fstream f(writer.path(1,2),std::ios::in|std::ios::out|std::ios::binary); f.seekp(33); f.put(99); }
    rejects([&]{reader.load(1,2);});
    writer.save(1,2,raw); std::filesystem::resize_file(writer.path(1,2),40); rejects([&]{reader.load(1,2);});
    rejects([&]{tqllm::ColdExpertStore r(root.string(),0,"source-map-v1",true);});
    writer.save(1,2,raw);
    rejects([&]{reader.fast_verification(true);});
    writer.prepare_fast(1,2);writer.finish_fast();reader.fast_verification(true);
    assert(reader.load(1,2)==raw);
    { std::fstream f(writer.path(1,2),std::ios::in|std::ios::out|std::ios::binary); f.seekp(33); f.put(99); }
    rejects([&]{reader.load(1,2);});
    writer.save(1,2,raw);
    { std::fstream f(writer.path(1,2).string()+".xxh64",std::ios::in|std::ios::out|std::ios::binary);f.seekp(8);f.put(99); }
    rejects([&]{reader.load(1,2);});
    writer.prepare_fast(1,2);
    std::filesystem::remove(writer.path(1,2).string()+".xxh64");rejects([&]{reader.load(1,2);});
    auto as_raw=[](const tqllm::ColdExpertStore::View & view) {
        std::array<std::vector<char>,3> out;
        for(int i=0;i<3;++i)out[i].assign(view.parts[i].data(),view.parts[i].data()+view.parts[i].size());
        return out;
    };
    writer.save(3,4,raw);writer.prepare_fast(3,4);
    auto mapped=reader.load_mapped(3,4);assert(as_raw(mapped)==raw);
    assert(reader.load_mapped(3,4).owner.get()==mapped.owner.get());
    {std::fstream f(writer.path(3,4),std::ios::in|std::ios::out|std::ios::binary);f.seekp(33);f.put(99);}
    rejects([&]{reader.load_mapped(3,4);});
    auto changed=raw;changed[0][0]=9;writer.save(3,4,changed);writer.prepare_fast(3,4);
    auto replaced=reader.load_mapped(3,4);assert(as_raw(replaced)==changed);
    assert(replaced.owner.get()!=mapped.owner.get());
    std::filesystem::resize_file(writer.path(3,4),40);rejects([&]{reader.load_mapped(3,4);});
    writer.save(6,7,raw);
    tqllm::ColdExpertStore::View retained_view;
    {tqllm::ColdExpertStore short_lived(root.string(),18,"source-map-v1");retained_view=short_lived.load_mapped(6,7);}
    assert(as_raw(retained_view)==raw);
    int loads=0;
    auto load=[&]{++loads;return raw;};
    tqllm::ColdExpertMemo memo(108);
    auto retained=memo.get({0,0},load);
    memo.get({47,0},load);
    memo.get({0,0},load);  // refresh the earlier layer
    memo.get({20,0},load); // evict layer 47, not layer 0
    assert(loads==3 && memo.bytes()==108);
    memo.get({0,0},load); assert(loads==3);
    memo.get({47,0},load); assert(loads==4);
    assert(*retained==raw); // in-flight payload survives memo eviction
    tqllm::ColdExpertMemo tiny(1);
    assert(*tiny.get({0,0},load)==raw && tiny.bytes()==0);
    auto before=memo.bytes();
    rejects([&]{memo.get({10,0},[]()->tqllm::ColdExpertMemo::Payload{throw std::runtime_error("read failed");});});
    assert(memo.bytes()==before);
    memo.set_capacity(0);
    assert(memo.bytes()==0 && *retained==raw);
    before=loads;
    for(int i=0;i<2;++i)assert(*memo.get({0,0},load)==raw);
    assert(loads==before+2 && memo.bytes()==0);
    memo.set_capacity(54);
    before=loads;
    memo.get({0,0},load);memo.get({0,0},load);
    assert(loads==before+1 && memo.bytes()==54);
    std::cout<<"cold_expert_store: PASS\n";
}
