#include "mlx_floatx.hpp"
#include "mlx_kv.hpp"
#include "inkling2_mlx_ops.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <tuple>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
int main() {
    mx::set_default_device(mx::Device::gpu);
    int failures = 0;
    {
        constexpr int H=2,N=3,S=5,D=32;
        std::vector<float> q(H*N*D),k(S*D),v(S*D),mask(H*N*S),sinks(H*N);
        for(size_t i=0;i<q.size();++i) q[i]=float(int(i%13)-6)/8;
        for(size_t i=0;i<k.size();++i) {k[i]=float(int(i%7)-3)/4;v[i]=float(int(i%11)-5)/8;}
        for(int h=0;h<H;++h) for(int t=0;t<N;++t) {
            sinks[h*N+t]=float(h+t)-1;
            for(int s=0;s<S;++s) mask[(h*N+t)*S+s]=s<=t+2?float(s-t)/8:-INFINITY;
        }
        auto qa=mx::astype(mx::array(q.data(),{1,H,N,D}),mx::bfloat16);
        auto ka=mx::astype(mx::array(k.data(),{1,1,S,D}),mx::bfloat16);
        auto va=mx::astype(mx::array(v.data(),{1,1,S,D}),mx::bfloat16);
        auto out=mx::astype(inkling2_masked_attention(qa,ka,va,mx::array(mask.data(),{1,H,N,S}),mx::array(sinks.data(),{1,H,N,1}),1/std::sqrt(float(D))),mx::float32);
        mx::eval(out);double err=0,mag=0;
        for(int h=0;h<H;++h) for(int t=0;t<N;++t) {
            double score[S],peak=sinks[h*N+t];
            for(int s=0;s<S;++s) {score[s]=mask[(h*N+t)*S+s];for(int d=0;d<D;++d) score[s]+=double(q[(h*N+t)*D+d])*k[s*D+d]/std::sqrt(double(D));peak=std::max(peak,score[s]);}
            double z=std::exp(sinks[h*N+t]-peak);for(auto s:score) z+=std::exp(s-peak);
            for(int d=0;d<D;++d) {double expected=0;for(int s=0;s<S;++s) expected+=std::exp(score[s]-peak)/z*v[s*D+d];double delta=out.data<float>()[(h*N+t)*D+d]-expected;err+=delta*delta;mag+=expected*expected;}
        }
        const double relative=std::sqrt(err/(mag+1e-30));failures+=relative>.006;
        std::printf("ATTENTION causal GQA with per-query sinks F64 oracle relative_rms %.8f\n",relative);
    }
    {
        const float data[8]={1,2,3,4,5,6,7,8},taps[8]={.25f,.5f,.75f,1.f,1.f,.75f,.5f,.25f};
        auto x=mx::astype(mx::array(data,{4,2}),mx::bfloat16);
        auto w=mx::array(taps,{2,4});
        auto empty=mx::zeros({3,2},mx::float32);
        auto [batch,hist]=inkling2_static_convolution(x,w,empty);
        float oracle[8]={};
        for(int t=0;t<4;++t) for(int c=0;c<2;++c) {
            float value=data[t*2+c];
            for(int tap=0;tap<4;++tap) if(t-3+tap>=0) value+=taps[c*4+tap]*data[(t-3+tap)*2+c];
            oracle[t*2+c]=value/std::sqrt(5.f);
        }
        auto expected=mx::astype(mx::array(oracle,{4,2}),mx::bfloat16);
        failures+=!mx::array_equal(batch,expected).item<bool>();
        auto history=empty;
        for(int t=0;t<4;++t) {
            auto [y,next]=inkling2_static_convolution(mx::slice(x,{t,0},{t+1,2}),w,history);
            failures+=!mx::array_equal(y,mx::slice(batch,{t,0},{t+1,2})).item<bool>();
            history=next;
        }
        failures+=!mx::array_equal(hist,history).item<bool>();
        std::puts("STATIC_CONV scalar oracle and chunk/state checks complete");
    }
    constexpr int E=2, O=4, K=64;
    const float e2m1[16]={0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6};
    for (int bits : {4,8}) {
        const int group=bits==4?16:32, packed=K*bits/8;
        std::vector<uint8_t> bytes(E*O*packed), scales(E*O*K/group);
        std::vector<float> dense(E*O*K);
        const uint8_t fp8[4]={0x38,0x40,0xb8,0xc0};
        const float f8[4]={1,2,-1,-2};
        for (int e=0;e<E;++e) for (int r=0;r<O;++r) for (int c=0;c<K;++c) {
            const int code=(r+c)%16;
            if (bits==4) bytes[(e*O+r)*packed+c/2] |= uint8_t(code << (4*(c%2)));
            else bytes[(e*O+r)*packed+c]=fp8[code%4];
            dense[(e*O+r)*K+c]=(bits==4?e2m1[code]:f8[code%4])*(e+1);
        }
        for (int e=0;e<E;++e) for (int i=0;i<O*K/group;++i) scales[e*O*K/group+i]=bits==4?0x38:uint8_t(127+e);
        const float outer[2]={2688,5376};
        std::optional<mx::array> global;
        if (bits==4) global=mx::array(outer,{E},mx::float32);
        FloatxExperts w(mx::array(bytes.data(),{E,O,packed},mx::uint8),mx::array(scales.data(),{E,O,K/group},mx::uint8),global,bits);
        std::vector<float> input(2*K);
        for (int i=0;i<int(input.size());++i) input[i]=float(i%7-3)/8;
        const uint32_t ids[4]={1,0,0,1};
        auto x=mx::astype(mx::array(input.data(),{2,1,1,K}),mx::bfloat16);
        auto y=mx::astype(mx::squeeze(w(x,mx::array(ids,{2,2},mx::uint32)),2),mx::float32);
        mx::eval(y);
        double worst=0;
        for (int t=0;t<2;++t) for (int j=0;j<2;++j) for (int r=0;r<O;++r) {
            float expected=0;
            for (int c=0;c<K;++c) expected+=input[t*K+c]*dense[(ids[t*2+j]*O+r)*K+c];
            const float got=y.data<float>()[(t*2+j)*O+r];
            worst=std::max(worst,double(std::abs(got-expected)));
        }
        std::printf("FLOATX bits%d low-nibble/interleave/expert-scale worst %.8g\n",bits,worst);
        failures+=worst>0.02;
        // Prefill path: the same unsorted routes grouped by expert, then restored to route order.
        constexpr int T=4,TOPK=2;
        const uint32_t routes[T*TOPK]={1,0,0,1,1,1,0,0};
        std::vector<float> tokens(T*K);
        for (int i=0;i<int(tokens.size());++i) tokens[i]=float(i%5-2)/4;
        auto xt=mx::astype(mx::array(tokens.data(),{T,K}),mx::bfloat16);
        auto ra=mx::array(routes,{T,TOPK},mx::uint32);
        ExpertGroups groups(ra);
        auto grouped=mx::astype(groups.scatter(w(groups.gather(xt,TOPK),groups.ids,true),T,TOPK),mx::float32);
        auto per_row=mx::astype(mx::squeeze(w(mx::reshape(xt,{T,1,1,K}),ra),2),mx::float32);
        mx::eval({grouped,per_row});
        double grouped_worst=0,row_worst=0;
        for (int t=0;t<T;++t) for (int j=0;j<TOPK;++j) for (int r=0;r<O;++r) {
            float expected=0;
            for (int c=0;c<K;++c) expected+=tokens[t*K+c]*dense[(routes[t*TOPK+j]*O+r)*K+c];
            const int at=(t*TOPK+j)*O+r;
            grouped_worst=std::max(grouped_worst,double(std::abs(grouped.data<float>()[at]-expected)));
            row_worst=std::max(row_worst,double(std::abs(per_row.data<float>()[at]-expected)));
        }
        std::printf("FLOATX bits%d grouped-by-expert worst %.8g per-row worst %.8g\n",bits,grouped_worst,row_worst);
        failures+=grouped.shape()!=mx::Shape{T,TOPK,O} || grouped_worst>0.02 || row_worst>0.02;
    }
    {
        // Windowed and full caches against a host model of the visible rows, across buffer growth and
        // window compaction; every returned view ends with the zero sink row.
        for (int window : {0,3}) {
            constexpr int HEADS=2,DIM=4;
            MlxKvCache cache(700,window);
            std::vector<std::vector<float>> rows;  // every appended row, [HEADS*DIM] each
            int next=0,bad=0;
            for (int n : {5,2,1,1,3,1,300,1,1,1,1}) {
                std::vector<float> data(HEADS*n*DIM);
                for (int h=0;h<HEADS;++h) for (int r=0;r<n;++r) for (int d=0;d<DIM;++d)
                    data[(h*n+r)*DIM+d]=float((next+r)*100+h*10+d+1);
                auto chunk=mx::array(data.data(),{1,HEADS,n,DIM});
                const int before=window ? std::max(0,next-window) : 0;  // first visible position
                auto [k,v]=cache.append(chunk,chunk);
                auto kk=mx::contiguous(k);mx::eval(kk);
                const int visible=next+n-before;
                bad+=k.shape(2)!=visible+1 || cache.start()!=before;
                for (int h=0;h<HEADS;++h) for (int r=0;r<=visible;++r) for (int d=0;d<DIM;++d) {
                    const float want=r==visible ? 0.f : float((before+r)*100+h*10+d+1);
                    bad+=kk.data<float>()[(h*(visible+1)+r)*DIM+d]!=want;
                }
                next+=n;cache.retain_window();cache.settle();
            }
            failures+=bad!=0 || cache.position()!=next;
            std::printf("KV window %d in-place rows, growth/compaction, zero sink row mismatches %d\n",window,bad);
        }
        {
            // Matrix-unit NVFP4 expert rows (verify blocks) against a dense oracle: 3 experts, 32 outputs, 64 inputs,
            // runs of 1-3 rows per expert, rows reading any input row.
            constexpr int E=3,O=32,K=64,T=4;
            const float e2m1[16]={0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6};
            const uint8_t fp8[4]={0x30,0x38,0x3c,0x40};  // E4M3 0.5, 1, 1.5, 2
            const float fp8v[4]={.5f,1.f,1.5f,2.f};
            std::vector<uint8_t> bytes(E*O*K/2,0),scales(E*O*K/16);
            std::vector<float> dense(E*O*K);
            for (int e=0;e<E;++e) for (int o=0;o<O;++o) {
                for (int g=0;g<K/16;++g) scales[(e*O+o)*(K/16)+g]=fp8[(e+o+g)%4];
                for (int k=0;k<K;++k) {
                    const int code=(3*o+5*k+7*e)%16;
                    bytes[(e*O+o)*(K/2)+k/2] |= uint8_t(code<<(4*(k%2)));
                    dense[(e*O+o)*K+k]=e2m1[code]*fp8v[(e+o+k/16)%4]*(e+1);
                }
            }
            const float outer[E]={2688,5376,8064};
            FloatxExperts w(mx::array(bytes.data(),{E,O,K/2},mx::uint8),mx::array(scales.data(),{E,O,K/16},mx::uint8),
                            mx::array(outer,{E},mx::float32),4);
            w.pack_nax();
            std::vector<float> input(T*K);
            for (int i=0;i<T*K;++i) input[i]=float(i%9-4)/8;
            const uint32_t ids[6]={0,1,1,2,2,2}, xrow[6]={3,0,2,1,3,0};
            auto x=mx::astype(mx::array(input.data(),{T,K}),mx::bfloat16);
            auto y=mx::astype(w.rows(x,mx::array(xrow,{6},mx::uint32),mx::array(ids,{6},mx::uint32)),mx::float32);
            mx::eval(y);
            double worst=0;
            for (int r=0;r<6;++r) for (int o=0;o<O;++o) {
                float expected=0;
                for (int k=0;k<K;++k) expected+=input[xrow[r]*K+k]*dense[(ids[r]*O+o)*K+k];
                worst=std::max(worst,double(std::abs(y.data<float>()[r*O+o]-expected)/(std::abs(expected)+1)));
            }
            std::printf("FLOATX nvfp4 matrix-unit expert rows relative worst %.8g\n",worst);
            failures+=worst>0.01;
        }
        {
            // Speculative decoding: a verify block of 4 rows, 2 kept (truncate before retain_window), then the next
            // append; and peek(), which shows rows without keeping them.
            constexpr int DIM=4;
            auto rows=[](int from,int n) {return mx::reshape(mx::arange(float(from*DIM),float((from+n)*DIM),mx::float32),{1,1,n,DIM});};
            int bad=0;
            for (int window : {0,3}) {
                MlxKvCache cache(64,window);
                cache.append(rows(0,5),rows(0,5));cache.retain_window();
                cache.append(rows(5,4),rows(5,4));  // verify block: positions 5..8
                cache.truncate(7);cache.retain_window();  // keep 5, 6
                auto [k,v]=cache.append(rows(100,1),rows(100,1));  // position 7 holds row 100
                auto kk=mx::contiguous(k);mx::eval(kk);
                const int first=window ? 8-window-1 : 0, visible=8-first;  // the window kept before the append, plus it
                bad+=cache.position()!=8 || k.shape(2)!=visible+1;
                for (int r=0;r<=visible;++r) for (int d=0;d<DIM;++d) {
                    const int pos=first+r;
                    const float want=r==visible ? 0.f : float((pos==7 ? 100 : pos)*DIM+d);
                    bad+=kk.data<float>()[r*DIM+d]!=want;
                }
                auto [pk,pv]=cache.peek(rows(200,2),rows(200,2));
                auto pc=mx::contiguous(pk);mx::eval(pc);
                bad+=pk.shape(2)!=visible+2 || pc.data<float>()[visible*DIM]!=200*DIM || cache.position()!=8;
                auto [k2,v2]=cache.append(rows(300,1),rows(300,1));  // overwrites the peeked rows
                auto c2=mx::contiguous(k2);mx::eval(c2);
                bad+=c2.data<float>()[(k2.shape(2)-2)*DIM]!=300*DIM;
            }
            failures+=bad!=0;
            std::printf("KV truncate / peek mismatches %d\n",bad);
        }
        MlxKvCache cache(8,3);std::vector<mx::array> pending;
        cache.collect(pending);failures+=!pending.empty();
        auto a=mx::reshape(mx::arange(0,20,mx::float32),{1,1,5,4});
        cache.append(a,a);cache.collect(pending);failures+=pending.size()!=2;
        cache.reset();failures+=cache.position()!=0 || cache.start()!=0;
    }
    {
        // Fused decode kernels against MLX-op or host references.
        uint32_t seed=12345;
        auto noise=[&](mx::Shape shape,float spread) {
            size_t count=1;for(auto d:shape) count*=size_t(d);
            std::vector<float> v(count);
            for(auto& x:v) {seed=seed*1664525u+1013904223u;x=(float(seed>>8)/float(1<<24)*2-1)*spread;}
            return mx::array(v.data(),shape);
        };
        auto relative=[](const mx::array& got,const mx::array& want) {
            auto g=mx::astype(got,mx::float32),w=mx::astype(want,mx::float32);
            auto e=mx::sqrt(mx::divide(mx::sum(mx::square(mx::subtract(g,w))),mx::add(mx::sum(mx::square(w)),mx::array(1e-30f))));
            return double(e.item<float>());
        };
        {
            // Norm: F32 / BF16 out, weighted or not, D below and above one threadgroup's width.
            double worst=0;
            for (int D : {96,2048}) for (bool weighted : {false,true}) for (auto out : {mx::float32,mx::bfloat16}) {
                auto x=mx::astype(noise({3,D},2),mx::bfloat16);
                std::optional<mx::array> w;
                if (weighted) w=mx::add(noise({D},.5),mx::array(1.f));
                auto want=mx::astype(mx::fast::rms_norm(mx::astype(x,mx::float32),w,1e-6f),out);
                auto got=inkling2_norm(x,w,out);
                worst=std::max(worst,relative(got,want));
                failures+=got.dtype()!=out || got.shape()!=x.shape();
            }
            std::printf("NORM fused rms_norm vs MLX relative worst %.8g\n",worst);
            failures+=worst>4e-3;  // BF16 output rounding; F32 cases are ~1e-7
        }
        {
            // Clip: no scale, one scale [d], per-head scale [heads, d]; vectors both under and over the limit.
            double worst=0;
            constexpr int H=4;
            for (int d : {64,128}) for (int mode : {0,1,2}) {
                auto x=noise({5,H,d},mode==0?.3f:1.5f);
                std::optional<mx::array> s;
                if (mode==1) s=noise({d},1);
                if (mode==2) s=noise({H,d},1);
                auto v=s?mx::multiply(x,*s):x;
                auto size=mx::sqrt(mx::sum(mx::square(v),-1,true));
                auto want=mx::astype(mx::multiply(v,mx::minimum(mx::divide(mx::array(2.f),mx::add(size,mx::array(1e-6f))),mx::array(1.f))),mx::bfloat16);
                auto got=inkling2_clip(x,2.f,s);
                worst=std::max(worst,relative(got,want));
                failures+=got.dtype()!=mx::bfloat16;
            }
            std::printf("CLIP fused L2 clip vs MLX relative worst %.8g\n",worst);
            failures+=worst>4e-3;
        }
        {
            // Route: ids exactly as a host sort (largest score + bias, lowest expert on ties), routes normalised and scaled.
            constexpr int N=5,E=64,TOPK=6;
            auto logits=noise({N,E},3);
            std::vector<float> b(E);for(int e=0;e<E;++e) b[e]=float(e%4)*.25f;
            auto bias=mx::array(b.data(),{E});
            auto lg=mx::astype(logits,mx::float32);mx::eval(lg);
            // force ties: row 0 has identical logits and bias for experts 8..11
            std::vector<float> lv(lg.data<float>(),lg.data<float>()+N*E);
            for(int e=8;e<12;++e) {lv[e]=5.f;b[e]=0;}
            logits=mx::array(lv.data(),{N,E});bias=mx::array(b.data(),{E});
            auto [ids,routes]=inkling2_route(logits,bias,TOPK,2.5f);
            mx::eval({ids,routes});
            int bad=0;double worst=0;
            for(int r=0;r<N;++r) {
                std::vector<int> order(E);for(int e=0;e<E;++e) order[e]=e;
                std::vector<float> sc(E);for(int e=0;e<E;++e) sc[e]=1.f/(1.f+std::exp(-lv[r*E+e]));
                std::stable_sort(order.begin(),order.end(),[&](int a,int c){return sc[a]+b[a]>sc[c]+b[c];});
                float total=0;for(int j=0;j<TOPK;++j) total+=sc[order[j]];
                for(int j=0;j<TOPK;++j) {
                    bad+=int(ids.data<uint32_t>()[r*TOPK+j])!=order[j];
                    worst=std::max(worst,double(std::abs(routes.data<float>()[r*TOPK+j]-sc[order[j]]/total*2.5f)));
                }
            }
            std::printf("ROUTE fused top-k ids mismatches %d routes worst %.8g\n",bad,worst);
            failures+=bad!=0 || worst>1e-5;
        }
        {
            // Conv: matches the original static convolution kernel; per-token runs carry the history exactly; the MoE
            // combine equals the plain conv over bf16(sum experts x routes).
            constexpr int T=4,C=300,TOPK=3;
            auto x=mx::astype(noise({T,C},2),mx::bfloat16);
            auto taps=noise({C,4},1);
            auto history=noise({3,C},1);
            auto [old_y,old_next]=inkling2_static_convolution(x,taps,history);
            auto [y,next,xin]=inkling2_conv(x,std::nullopt,std::nullopt,history,taps);
            const double conv_err=relative(y,old_y),hist_err=relative(next,old_next);
            failures+=conv_err>4e-3 || hist_err!=0 || !mx::array_equal(xin,x).item<bool>();
            auto h=history;int bad=0;
            for(int t=0;t<T;++t) {
                auto [yt,nt,xt]=inkling2_conv(mx::slice(x,{t,0},{t+1,C}),std::nullopt,std::nullopt,h,taps);
                bad+=!mx::array_equal(yt,mx::slice(y,{t,0},{t+1,C})).item<bool>();
                h=nt;
            }
            bad+=!mx::array_equal(h,next).item<bool>();
            auto experts=mx::astype(noise({T,TOPK,C},2),mx::bfloat16);
            auto routes=noise({T,TOPK},1);
            auto combined=mx::astype(mx::sum(mx::multiply(mx::astype(experts,mx::float32),mx::expand_dims(routes,-1)),1),mx::bfloat16);
            auto [cy,cnext,cx]=inkling2_conv(std::nullopt,experts,routes,history,taps);
            auto [py,pnext,px]=inkling2_conv(combined,std::nullopt,std::nullopt,history,taps);
            const double combine_err=relative(cx,combined);
            failures+=bad!=0 || combine_err>4e-3 || relative(cy,py)>4e-3;
            std::printf("CONV fused vs static conv relative %.8g history %.8g chunk mismatches %d combine relative %.8g\n",
                        conv_err,hist_err,bad,combine_err);
        }
        {
            // Mask: host oracle, full and windowed, keys past the bias extent, sink logit in the last column.
            constexpr int N=3,H=2,E=4;
            auto bias=mx::astype(noise({N,H,E},1),mx::bfloat16),sinks=mx::astype(noise({N,H},1),mx::bfloat16);
            auto bf=mx::astype(bias,mx::float32),sf=mx::astype(sinks,mx::float32);mx::eval({bf,sf});
            int bad=0;
            for (int window : {0,3}) {
                const int base=10,start=window?8:0,s=base+N-start;
                auto m=mx::astype(inkling2_mask(bias,sinks,base,start,s,window),mx::float32);mx::eval(m);
                bad+=m.shape()!=mx::Shape{1,H,N,s+1};
                for(int h=0;h<H;++h) for(int r=0;r<N;++r) for(int c=0;c<=s;++c) {
                    float want;
                    if(c==s) want=sf.data<float>()[r*H+h];
                    else {
                        const int d=base+r-(start+c);
                        const bool visible=d>=0 && (window==0 || d<window);
                        want=!visible ? -INFINITY : d<E ? bf.data<float>()[(r*H+h)*E+d] : 0.f;
                    }
                    bad+=m.data<float>()[(h*N+r)*(s+1)+c]!=want;
                }
            }
            std::printf("MASK fused relative-bias mask mismatches %d\n",bad);
            failures+=bad!=0;
        }
        {
            // Depth: 0..4 finished blocks, 1 and 3 queries, against stack / matmul / softmax.
            constexpr int N=3,D=512;
            double worst=0;
            for (int nb : {0,2,4}) for (int q : {1,3}) {
                std::vector<mx::array> blocks;
                for(int i=0;i<nb;++i) blocks.push_back(noise({N,D},2));
                auto partial=noise({N,D},3),query=noise({q,D},.2f);
                auto normed=mx::fast::rms_norm(partial,std::nullopt,1e-6f);
                auto values=blocks;values.push_back(normed);
                auto states=mx::transpose(mx::stack(values,0),{1,0,2});  // [N, S, D]
                auto scores=mx::multiply(mx::sum(mx::multiply(mx::expand_dims(states,2),mx::reshape(query,{1,1,q,D})),-1),
                                         mx::array(1.f/std::sqrt(float(D))));  // [N, S, Q]; no matmul, so TF32 cannot apply
                auto w=mx::softmax(scores,1);
                auto want=mx::sum(mx::multiply(mx::expand_dims(w,-1),mx::expand_dims(states,2)),1);  // [N, Q, D]
                auto [mix,got_normed]=inkling2_depth(blocks,partial,query);
                worst=std::max({worst,relative(mix,want),relative(got_normed,normed)});
                failures+=mix.shape()!=mx::Shape{N,q,D};
            }
            std::printf("DEPTH fused depth attention vs MLX relative worst %.8g\n",worst);
            failures+=worst>1e-5;
            // the normalised rows written by the depth kernel equal inkling2_norm of each mix row, bit for bit
            int unequal=0;
            for (int nb : {0,3}) {
                std::vector<mx::array> blocks;
                for(int i=0;i<nb;++i) blocks.push_back(noise({N,D},2));
                auto partial=noise({N,D},3),query=noise({3,D},.2f);
                std::vector<mx::array> weights_{noise({D},1),noise({D},1),noise({D},1)};
                auto r=inkling2_depth_normed(blocks,partial,query,weights_);
                auto plain=inkling2_depth(blocks,partial,query);
                unequal+=!mx::array_equal(r[0],plain.first).item<bool>() || !mx::array_equal(r[1],plain.second).item<bool>();
                for(int j=0;j<3;++j) {
                    auto mix_row=mx::squeeze(mx::slice(r[0],{0,j,0},{N,j+1,D}),1);
                    auto ready_row=mx::squeeze(mx::slice(r[2],{0,j,0},{N,j+1,D}),1);
                    unequal+=!mx::array_equal(ready_row,inkling2_norm(mix_row,weights_[size_t(j)],mx::bfloat16)).item<bool>();
                }
            }
            std::printf("DEPTH normalised rows vs inkling2_norm unequal %d\n",unequal);
            failures+=unequal!=0;
        }
        {
            // Conv with the residual add folded in: F32 and BF16 sums equal the separate add, bit for bit.
            constexpr int T=3,C=512,TOPK=4;
            auto taps=noise({C,4},1),history=noise({3,C},1);
            auto x=mx::astype(noise({T,C},2),mx::bfloat16),h=mx::astype(noise({T,C},2),mx::bfloat16),res=noise({T,C},2);
            auto plain=inkling2_conv(x,std::nullopt,std::nullopt,history,taps);
            auto added=inkling2_conv(x,std::nullopt,std::nullopt,history,taps,h,mx::float32);
            auto alone=inkling2_conv(x,std::nullopt,std::nullopt,history,taps,std::nullopt,mx::float32);
            int unequal=!mx::array_equal(added[0],mx::add(mx::astype(h,mx::float32),mx::astype(plain[0],mx::float32))).item<bool>();
            unequal+=!mx::array_equal(alone[0],mx::astype(plain[0],mx::float32)).item<bool>() || added[0].dtype()!=mx::float32;
            unequal+=!mx::array_equal(added[1],plain[1]).item<bool>() || !mx::array_equal(added[2],plain[2]).item<bool>();
            auto experts=mx::astype(noise({T,TOPK,C},2),mx::bfloat16);
            auto routes=noise({T,TOPK},1);
            auto combined=inkling2_conv(std::nullopt,experts,routes,history,taps);
            auto summed=inkling2_conv(std::nullopt,experts,routes,history,taps,res,mx::bfloat16);
            unequal+=!mx::array_equal(summed[0],mx::astype(mx::add(res,mx::astype(combined[0],mx::float32)),mx::bfloat16)).item<bool>();
            std::printf("CONV residual fold unequal %d\n",unequal);
            failures+=unequal!=0;
        }
        {
            // Exact greedy head from a bounded 6-bit one (as the model builds it): the same token as argmax(cap * tanh(bf16 logits / cap)) over the
            // first V rows of the BF16 head, including exact ties (duplicated rows), near ties and padded rows.
            constexpr int ROWS=2048,V=2000,D=512;
            constexpr float CAP=30.f;
            auto wf=noise({ROWS,D},.05f);
            // rows 7 and 900 duplicate row 1500 (ties go to the lowest index); 1501 is 1500 nudged; padding rows are large
            auto base=mx::slice(wf,{1500,0},{1501,D});
            wf=mx::slice_update(wf,base,{7,0},{8,D});
            wf=mx::slice_update(wf,base,{900,0},{901,D});
            wf=mx::slice_update(wf,mx::add(base,mx::array(1e-4f)),{1501,0},{1502,D});
            wf=mx::slice_update(wf,mx::multiply(base,mx::array(4.f)),{V,0},{V+1,D});
            auto w=mx::astype(wf,mx::bfloat16);
            auto w32=mx::astype(w,mx::float32);
            auto q=mx::quantize(w32,64,6);
            auto error=mx::subtract(w32,mx::dequantize(q[0],q[1],q[2],64,6));
            auto bound=mx::add(mx::multiply(mx::sqrt(mx::sum(mx::square(error),-1)),mx::array(1.0001f)),
                               mx::multiply(mx::sqrt(mx::sum(mx::square(w32),-1)),mx::array(3e-4f)));
            mx::eval({w,q[0],q[1],q[2],bound});
            int wrong=0,ties=0;double candidates=0;
            for (int trial=0;trial<40;++trial) {
                auto x=mx::astype(trial%4==0 ? mx::multiply(base,mx::array(40.f)) : noise({1,D},3),mx::bfloat16);
                auto xf=mx::astype(x,mx::float32);
                auto approx=mx::quantized_matmul(xf,q[0],q[1],q[2],true,64,6);
                auto best=mx::argmax(mx::slice(approx,{0,0},{1,V}),-1);
                auto norm=mx::sqrt(mx::sum(mx::square(xf)));
                auto sel=inkling2_head_select(x,w,approx,bound,norm,best,V,CAP);
                auto lg=mx::slice(mx::astype(mx::matmul(x,mx::transpose(w)),mx::float32),{0,0},{1,V});
                auto ref=mx::multiply(mx::tanh(mx::divide(lg,mx::array(CAP))),mx::array(CAP));
                auto got=mx::argmax(sel,-1),want=mx::argmax(ref,-1);
                auto count=mx::sum(mx::isfinite(sel));
                mx::eval({got,want,count});
                wrong+=got.item<uint32_t>()!=want.item<uint32_t>() || sel.shape()!=mx::Shape{1,V};
                ties+=trial%4==0 && got.item<uint32_t>()==7;
                candidates+=count.item<int>();
            }
            // a verify block: several inputs in one call, each row its own bound and best
            auto xs=mx::astype(mx::concatenate({mx::multiply(base,mx::array(40.f)),noise({2,D},3)},0),mx::bfloat16);
            auto xsf=mx::astype(xs,mx::float32);
            auto approx=mx::quantized_matmul(xsf,q[0],q[1],q[2],true,64,6);
            auto best=mx::argmax(mx::slice(approx,{0,0},{3,V}),-1);
            auto sel=inkling2_head_select(xs,w,approx,bound,mx::sqrt(mx::sum(mx::square(xsf),-1)),best,V,CAP);
            auto lg=mx::slice(mx::astype(mx::matmul(xs,mx::transpose(w)),mx::float32),{0,0},{3,V});
            auto ref=mx::multiply(mx::tanh(mx::divide(lg,mx::array(CAP))),mx::array(CAP));
            const bool batch_ok=sel.shape()==mx::Shape{3,V} && mx::array_equal(mx::argmax(sel,-1),mx::argmax(ref,-1)).item<bool>();
            std::printf("HEAD bounded 6-bit greedy head wrong %d of 40, tie cases to row 7 %d of 10, mean rows recomputed %.1f, "
                        "3-row block %s\n",wrong,ties,candidates/40,batch_ok?"equal":"DIFFERS");
            failures+=wrong!=0 || ties!=10 || !batch_ok;
        }
        {
            // One-token expert product with cached zero row indices equals the per-row product.
            constexpr int E=2,O=4,K=64;
            std::vector<uint8_t> bytes(E*O*K/2),scales(E*O*K/16,0x38);
            for(size_t i=0;i<bytes.size();++i) bytes[i]=uint8_t((i*37+11)&0xff);
            const float outer[E]={2688,5376};
            FloatxExperts w(mx::array(bytes.data(),{E,O,K/2},mx::uint8),mx::array(scales.data(),{E,O,K/16},mx::uint8),
                            mx::array(outer,{E},mx::float32),4);
            auto x=mx::astype(noise({1,1,1,K},1),mx::bfloat16);
            const uint32_t ids[2]={1,0};
            auto ia=mx::array(ids,{1,2},mx::uint32);
            auto one=w(x,ia);
            auto two=w(mx::concatenate({x,x},0),mx::array(std::vector<uint32_t>{1,0,1,0}.data(),{2,2},mx::uint32));
            failures+=one.shape()!=mx::Shape{1,2,1,O} || !mx::array_equal(one,mx::slice(two,{0,0,0,0},{1,2,1,O})).item<bool>();
            failures+=&FloatxExperts::zero_rows(2)!=&FloatxExperts::zero_rows(2);
            std::puts("FLOATX one-token zero-row indices match per-row product");
        }
    }
    {
        // Prompt attention on the matrix unit against the mask + MLX attention path: full and sliding layers, a chunk
        // after 40 cached rows, 70 rows (a partial threadgroup), the cache views read in place through their strides.
        uint32_t seed=777;
        auto noise=[&](mx::Shape shape,float spread) {
            size_t count=1;for(auto d:shape) count*=size_t(d);
            std::vector<float> v(count);
            for(auto& x:v) {seed=seed*1664525u+1013904223u;x=(float(seed>>8)/float(1<<24)*2-1)*spread;}
            return mx::array(v.data(),shape);
        };
        constexpr int H=6,G=2,HD=128,N=70;
        double worst=0;
        // (window, cached rows, bias extent): full; a window smaller than a threadgroup's rows; a window with whole tiles
        // inside it for every row (the unmasked path)
        for (auto [window,PRE,E] : {std::tuple{0,40,16},std::tuple{50,40,16},std::tuple{200,300,256}}) {
            MlxKvCache cache(512,window);
            auto k0=mx::astype(noise({1,G,PRE,HD},1),mx::bfloat16),v0=mx::astype(noise({1,G,PRE,HD},1),mx::bfloat16);
            cache.append(k0,v0);cache.retain_window();cache.settle();
            const int base=cache.position();
            auto k1=mx::astype(noise({1,G,N,HD},1),mx::bfloat16),v1=mx::astype(noise({1,G,N,HD},1),mx::bfloat16);
            auto [kc,vc]=cache.append(k1,v1);
            const int s=kc.shape(2)-1,start=cache.start();
            auto q=mx::astype(noise({1,H,N,HD},.3f),mx::bfloat16);
            auto bias=mx::astype(noise({N,H,E},1),mx::bfloat16),sinks=mx::astype(noise({N,H},1),mx::bfloat16);
            const float scale=1.f/std::sqrt(float(HD));
            auto want=inkling2_sink_attention(q,kc,vc,inkling2_mask(bias,sinks,base,start,s,window),scale);
            auto got=inkling2_prompt_attention(q,kc,vc,bias,sinks,base,start,s,window,scale);
            auto e=mx::sqrt(mx::divide(mx::sum(mx::square(mx::subtract(mx::astype(got,mx::float32),mx::astype(want,mx::float32)))),
                                       mx::sum(mx::square(mx::astype(want,mx::float32)))));
            worst=std::max(worst,double(e.item<float>()));
            failures+=got.shape()!=want.shape();
        }
        std::printf("PROMPT_ATTENTION matrix unit vs mask + MLX attention relative worst %.8g\n",worst);
        failures+=worst>1e-2;
    }
    {
        // Prompt-chunk message gate: merge + norm in one kernel against the MLX ops (tanh, merge, inkling2_norm).
        uint32_t seed=4242;
        auto noise=[&](mx::Shape shape,float spread) {
            size_t count=1;for(auto d:shape) count*=size_t(d);
            std::vector<float> v(count);
            for(auto& x:v) {seed=seed*1664525u+1013904223u;x=(float(seed>>8)/float(1<<24)*2-1)*spread;}
            return mx::array(v.data(),shape);
        };
        constexpr int N=37,D=512;
        auto fg=noise({N,2*D},2),h=mx::astype(noise({N,D},3),mx::bfloat16),m=noise({N,D},3),w=noise({D},1);
        auto f=mx::tanh(mx::slice(fg,{0,0},{N,D})),g=mx::tanh(mx::slice(fg,{0,D},{N,2*D}));
        auto merged=mx::add(mx::multiply(mx::subtract(mx::array(1.f),f),mx::astype(h,mx::float32)),mx::multiply(g,m));
        auto want=inkling2_norm(merged,w,mx::bfloat16);
        auto got=inkling2_gate_merge(fg,h,m,w);
        auto e=mx::sqrt(mx::divide(mx::sum(mx::square(mx::subtract(mx::astype(got,mx::float32),mx::astype(want,mx::float32)))),
                                   mx::sum(mx::square(mx::astype(want,mx::float32)))));
        const double rel=e.item<float>();
        std::printf("GATE_MERGE fused merge + norm vs MLX ops relative %.8g\n",rel);
        failures+=rel>4e-3 || got.dtype()!=mx::bfloat16 || got.shape()!=h.shape();
    }
    {
        // Prompt-chunk K / V preparation (conv + clip or norm in one kernel) equals the separate kernels bit for bit,
        // static and dynamic; the clip's row scale equals bf16(f32(clip) * scale); the combine read through an order
        // equals the combine over rows restored first.
        uint32_t seed=99;
        auto noise=[&](mx::Shape shape,float spread) {
            size_t count=1;for(auto d:shape) count*=size_t(d);
            std::vector<float> v(count);
            for(auto& x:v) {seed=seed*1664525u+1013904223u;x=(float(seed>>8)/float(1<<24)*2-1)*spread;}
            return mx::array(v.data(),shape);
        };
        constexpr int N=20,G=2,C=G*128;
        int unequal=0;
        auto x=mx::astype(noise({N,C},2),mx::bfloat16);
        auto hist=noise({3,C},2),taps=noise({C,4},1);
        // static: inkling2_conv then clip / norm
        auto ref=inkling2_conv(x,std::nullopt,std::nullopt,hist,taps);
        for (bool clip : {true,false}) {
            auto [out,next]=inkling2_kv_prep(x,hist,taps,std::nullopt,clip,1.5f);
            auto want=clip?inkling2_clip(mx::reshape(ref[0],{N,G,128}),1.5f,std::nullopt)
                          :inkling2_norm(mx::reshape(ref[0],{N,G,128}),std::nullopt,mx::bfloat16);
            unequal+=!mx::array_equal(mx::reshape(out,{N,G,128}),want).item<bool>() || !mx::array_equal(next,ref[1]).item<bool>();
        }
        // dynamic: the MLX form (as the model's convolution())
        auto dyn=noise({N,4*G},.5f);
        auto values=mx::concatenate({hist,mx::astype(x,mx::float32)},0);
        auto tl=mx::add(mx::reshape(taps,{1,G,128,4}),mx::reshape(dyn,{N,G,1,4}));
        std::optional<mx::array> sum;
        for (int lag=0;lag<4;++lag) {
            auto v=mx::reshape(mx::slice(values,{3-lag,0},{3-lag+N,C}),{N,G,128});
            auto term=mx::reshape(mx::multiply(v,mx::squeeze(mx::slice(tl,{0,0,0,lag},{N,G,128,lag+1}),3)),{N,C});
            sum=sum?mx::add(*sum,term):term;
        }
        auto dref=mx::astype(mx::divide(mx::add(*sum,mx::astype(x,mx::float32)),mx::array(3.f)),mx::bfloat16);
        auto [dout,dnext]=inkling2_kv_prep(x,hist,taps,dyn,true,1.5f);
        unequal+=!mx::array_equal(mx::reshape(dout,{N,G,128}),inkling2_clip(mx::reshape(dref,{N,G,128}),1.5f,std::nullopt)).item<bool>();
        unequal+=!mx::array_equal(dnext,mx::slice(values,{N,0},{N+3,C})).item<bool>();
        // clip with a row scale
        auto q=noise({N,3,128},2),scale=noise({3,128},1),tau=mx::add(noise({N},.3f),mx::array(1.f));
        auto plain=inkling2_clip(q,1.f,scale);
        auto want=mx::astype(mx::multiply(mx::astype(plain,mx::float32),mx::reshape(tau,{N,1,1})),mx::bfloat16);
        unequal+=!mx::array_equal(inkling2_clip(q,1.f,scale,tau),want).item<bool>();
        // ordered combine: rows permuted, read through the order
        constexpr int TOPK=4;
        auto experts=mx::astype(noise({N,TOPK,C},2),mx::bfloat16);
        auto routes=noise({N,TOPK},1);
        std::vector<uint32_t> perm(N*TOPK);
        for (int i=0;i<N*TOPK;++i) perm[size_t(i)]=uint32_t((i*37+5)%(N*TOPK));
        auto order=mx::array(perm.data(),{N*TOPK},mx::uint32);
        auto inverse=mx::argsort(order,0);  // row r of the permuted holds pair inverse[r]
        auto permuted=mx::reshape(mx::take(mx::reshape(experts,{N*TOPK,C}),inverse,0),{N*TOPK,1,C});
        auto direct=inkling2_conv(std::nullopt,experts,routes,hist,taps,x,mx::bfloat16);
        auto read=inkling2_conv(std::nullopt,permuted,routes,hist,taps,x,mx::bfloat16,order);
        for (int i=0;i<3;++i) unequal+=!mx::array_equal(direct[size_t(i)],read[size_t(i)]).item<bool>();
        std::printf("KV_PREP / clip row scale / ordered combine unequal %d\n",unequal);
        failures+=unequal!=0;
    }
    {
        // Verify-block pair sort: a stable sort by expert, with each pair's token row and its sorted position, against
        // a host stable sort; repeated experts (ties) and every row count up to 8 rows of 8.
        int bad=0;
        for (int n : {1,2,3,8}) {
            constexpr int K=8;
            std::vector<uint32_t> ids(size_t(n*K));
            for (int p=0;p<n*K;++p) ids[size_t(p)]=uint32_t((p*37+p/3)%(n>1?20:128));  // n>1: many shared experts
            auto out=inkling2_pair_sort(mx::array(ids.data(),{n,K},mx::uint32));
            mx::eval(out);
            std::vector<int> order(size_t(n*K));
            for (int p=0;p<n*K;++p) order[size_t(p)]=p;
            std::stable_sort(order.begin(),order.end(),[&](int a,int b){return ids[size_t(a)]<ids[size_t(b)];});
            const auto* o=out.data<uint32_t>();
            const int P=n*K;
            for (int r=0;r<P;++r) {
                bad+=o[r]!=ids[size_t(order[size_t(r)])] || o[P+r]!=uint32_t(order[size_t(r)]/K);
                bad+=o[2*P+order[size_t(r)]]!=uint32_t(r);
            }
        }
        std::printf("PAIR_SORT stable sort / token rows / restore mismatches %d\n",bad);
        failures+=bad!=0;
    }
    {
        // Verify rollback of a convolution history: equal to the concatenate + slice it replaces, every keep 0..n.
        int unequal=0;
        constexpr int C=300,N=5;
        auto before=mx::random::normal({3,C}),input=mx::astype(mx::random::normal({N,C}),mx::bfloat16);
        for (int keep=0;keep<=N;++keep) {
            auto rows=mx::concatenate({before,mx::astype(input,mx::float32)},0);
            unequal+=!mx::array_equal(inkling2_history_rollback(before,input,keep),mx::slice(rows,{keep,0},{keep+3,C})).item<bool>();
        }
        std::printf("HISTORY_ROLLBACK unequal %d\n",unequal);
        failures+=unequal!=0;
    }
    std::puts(failures?"INKLING2_MLX_OPS_FAIL":"INKLING2_MLX_OPS_PASS");
    return failures?1:0;
}
