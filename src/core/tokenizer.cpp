#include "tqllm/tokenizer.hpp"
#include "gguf.hpp"
#include "hf_folder.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace tqllm {
namespace {
template<class T> T read(std::istream & in) { T x{}; in.read(reinterpret_cast<char *>(&x), sizeof(x)); if (!in) throw std::runtime_error("truncated tokenizer data"); return x; }
std::string read_string(std::istream & in, uint32_t n) { std::string s(n, '\0'); in.read(s.data(), n); if (!in) throw std::runtime_error("truncated tokenizer string"); return s; }
std::string read_u16_string(std::istream & in) { return read_string(in, read<uint16_t>(in)); }
void append_utf8(std::string & out, uint32_t cp) {
    if (cp <= 0x7f) out.push_back(char(cp));
    else if (cp <= 0x7ff) { out.push_back(char(0xc0 | (cp >> 6))); out.push_back(char(0x80 | (cp & 63))); }
    else if (cp <= 0xffff) { out.push_back(char(0xe0 | (cp >> 12))); out.push_back(char(0x80 | ((cp >> 6) & 63))); out.push_back(char(0x80 | (cp & 63))); }
    else { out.push_back(char(0xf0 | (cp >> 18))); out.push_back(char(0x80 | ((cp >> 12) & 63))); out.push_back(char(0x80 | ((cp >> 6) & 63))); out.push_back(char(0x80 | (cp & 63))); }
}
struct Rune { uint32_t cp; size_t begin, end; };
std::vector<Rune> runes(const std::string & s) {
    std::vector<Rune> out;
    for (size_t i = 0; i < s.size();) {
        const size_t b = i; const uint8_t c = uint8_t(s[i++]); uint32_t cp;
        if (c < 0x80) cp = c;
        else if ((c & 0xe0) == 0xc0 && i < s.size()) cp = ((c & 31) << 6) | (uint8_t(s[i++]) & 63);
        else if ((c & 0xf0) == 0xe0 && i + 1 < s.size()) { const auto c1=uint8_t(s[i++]),c2=uint8_t(s[i++]); cp=((c&15)<<12)|((c1&63)<<6)|(c2&63); }
        else if ((c & 0xf8) == 0xf0 && i + 2 < s.size()) { const auto c1=uint8_t(s[i++]),c2=uint8_t(s[i++]),c3=uint8_t(s[i++]); cp=((c&7)<<18)|((c1&63)<<12)|((c2&63)<<6)|(c3&63); }
        else cp = 0xfffd;
        out.push_back({cp, b, i});
    }
    return out;
}
uint32_t byte_to_code(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) return b;
    uint32_t n = 0;
    for (int x = 0; x < 256; ++x) if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174 && x <= 255)) continue; else { if (x == b) return 256 + n; ++n; }
    return b;
}
int code_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) return int(cp);
    if (cp < 256 || cp >= 512) return -1;
    int n = int(cp - 256);
    for (int x = 0; x < 256; ++x) if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174 && x <= 255)) continue; else { if (n-- == 0) return x; }
    return -1;
}
}

void Tokenizer::load(const std::string & path) {
    if (hf_folder(path)) { load_gguf(path); return; }  // gguf_open() presents the folder's tokenizer.json as GGUF keys
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open tokenizer: " + path);
    char magic[8]; in.read(magic, 8);
    if (in.gcount() >= 4 && std::string(magic, 4) == "GGUF") { in.close(); load_gguf(path); return; }
    if (!in || std::string(magic, 8) != "TQTOK001") throw std::runtime_error("invalid tokenizer file");
    const auto nv = read<uint32_t>(in), nm = read<uint32_t>(in), ns = read<uint32_t>(in);
    eos_id_ = read<uint32_t>(in);
    id_to_piece_.clear(); piece_to_id_.clear(); special_to_id_.clear(); merges_.clear(); merge_rank_.clear();
    id_to_piece_.resize(300000);
    for (uint32_t i = 0; i < nv; ++i) { auto id=read<uint32_t>(in); auto s=read_u16_string(in); if (id>=id_to_piece_.size()) id_to_piece_.resize(id+1); id_to_piece_[id]=s; piece_to_id_.emplace(std::move(s),id); }
    merges_.reserve(nm);
    for (uint32_t i = 0; i < nm; ++i) { auto a=read_u16_string(in), b=read_u16_string(in); const auto rank=static_cast<uint32_t>(i); merge_rank_.emplace(a+char(0)+b,rank); merges_.emplace_back(std::move(a),std::move(b)); }
    for (uint32_t i = 0; i < ns; ++i) { auto id=read<uint32_t>(in); auto s=read_u16_string(in); special_to_id_.emplace(std::move(s),id); }
    const auto nr=read<uint32_t>(in); unicode_ranges_.clear(); unicode_ranges_.reserve(nr);
    for (uint32_t i=0;i<nr;++i) { const auto a=read<uint32_t>(in); const auto b=read<uint32_t>(in); const auto m=read<uint8_t>(in); unicode_ranges_.emplace_back(a,b,m); }
    const auto regex_len=read<uint32_t>(in); marks_in_letters_=read_string(in,regex_len).find("\\p{M}")!=std::string::npos;
}

void Tokenizer::load_gguf(const std::string & path) {
    static const std::tuple<uint32_t, uint32_t, uint8_t> ranges[] = {
#include "unicode_ranges.inc"
    };
    const auto g = gguf_open(path);
    auto key = [&](const char * k) -> const GgufValue & {
        const auto it = g.kv.find(k);
        if (it == g.kv.end()) throw std::runtime_error(std::string("GGUF has no ") + k + ": " + path);
        return it->second;
    };
    const auto & pre = key("tokenizer.ggml.pre").str;
    if (key("tokenizer.ggml.model").str != "gpt2" || (pre != "qwen2" && pre != "qwen35" && pre != "minimax-m3" && pre != "glm4"))
        throw std::runtime_error("GGUF tokenizer differs from the implemented Qwen byte-level BPE (pre " + pre + "); pass --tokenizer");
    const auto & tokens = key("tokenizer.ggml.tokens").strs;
    const auto & types = key("tokenizer.ggml.token_type").arr;
    const auto & merges = key("tokenizer.ggml.merges").strs;
    if (tokens.empty() || types.size() != tokens.size()) throw std::runtime_error("GGUF tokenizer tokens/types mismatch: " + path);
    id_to_piece_.clear(); piece_to_id_.clear(); special_to_id_.clear(); merges_.clear(); merge_rank_.clear();
    id_to_piece_.resize(std::max<size_t>(300000, tokens.size()));
    // token_type: 1 normal, 3 control, 4 user-defined (both matched atomically, as HF added tokens), 5 unused padding
    for (size_t i = 0; i < tokens.size(); ++i) {
        const int t = int(types[i]);
        if (t == 1) { id_to_piece_[i] = tokens[i]; piece_to_id_.emplace(tokens[i], uint32_t(i)); }
        else if (t == 3 || t == 4) special_to_id_.emplace(tokens[i], uint32_t(i));
    }
    merges_.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); ++i) {
        const auto sp = merges[i].find(' ');
        if (sp == std::string::npos) throw std::runtime_error("bad GGUF merge entry: " + merges[i]);
        auto a = merges[i].substr(0, sp), b = merges[i].substr(sp + 1);
        merge_rank_.emplace(a + char(0) + b, uint32_t(i)); merges_.emplace_back(std::move(a), std::move(b));
    }
    eos_id_ = uint32_t(g.num("tokenizer.ggml.eos_token_id"));
    if (g.has("tokenizer.ggml.eot_token_id")) eot_id_ = uint32_t(g.num("tokenizer.ggml.eot_token_id"));
    if (g.has("tokenizer.ggml.eom_token_id")) eom_id_ = uint32_t(g.num("tokenizer.ggml.eom_token_id"));
    unicode_ranges_.assign(std::begin(ranges), std::end(ranges));
    marks_in_letters_ = pre == "qwen35";
    o200k_ = pre == "minimax-m3";
    digits3_ = pre == "glm4";
    static const std::tuple<uint32_t, uint32_t, uint8_t> cases[] = {
#include "unicode_case_ranges.inc"
    };
    case_ranges_.assign(std::begin(cases), std::end(cases));
}

uint8_t Tokenizer::letter_case(uint32_t cp) const {
    auto it=std::upper_bound(case_ranges_.begin(),case_ranges_.end(),cp,[](uint32_t x,const auto & r){return x<std::get<0>(r);});
    if (it==case_ranges_.begin()) return 0;
    --it; return cp<=std::get<1>(*it)?std::get<2>(*it):0;
}

// The MiniMax-M3 split (tokenizer.json, Split invert / ByteLevel without its own regex), alternatives in regex order:
//   A [^\r\n\p{L}\p{N}]?[Lu Lt Lm Lo M]*[Ll Lm Lo M]+('s|'t|'re|'ve|'m|'ll|'d)?   (contractions case-insensitive)
//   B [^\r\n\p{L}\p{N}]?[Lu Lt Lm Lo M]+[Ll Lm Lo M]*('s|...)?
//   C \p{N}{1,3}    D  ?[^\s\p{L}\p{N}]+[\r\n/]*    E \s*[\r\n]+    F \s+(?!\S)    G \s+
// with the regex engine's greedy-then-backtrack choices made explicitly.
std::vector<std::string> Tokenizer::pretokenize_o200k(const std::string & text) const {
    std::vector<std::string> out; const auto rs=runes(text); const size_t n=rs.size();
    auto cat=[&](size_t p){return p<n?category(rs[p].cp):uint8_t(0);};
    auto letter=[&](size_t p){return p<n&&(cat(p)&1);};
    auto mark=[&](size_t p){return p<n&&(cat(p)&2);};
    auto num=[&](size_t p){return p<n&&(cat(p)&4);};
    auto space=[&](size_t p){return p<n&&(cat(p)&8);};
    auto nl=[&](size_t p){return p<n&&(rs[p].cp=='\r'||rs[p].cp=='\n');};
    auto up=[&](size_t p){ if(p>=n) return false; if(mark(p)) return true; if(!letter(p)) return false; return (letter_case(rs[p].cp)&32)==0; };  // Lu Lt Lm Lo M
    auto lo=[&](size_t p){ if(p>=n) return false; if(mark(p)) return true; if(!letter(p)) return false; return (letter_case(rs[p].cp)&16)==0; };  // Ll Lm Lo M
    auto prefix_ok=[&](size_t p){return p<n&&!nl(p)&&!letter(p)&&!num(p);};
    auto contraction=[&](size_t p)->size_t{
        if(p>=n||rs[p].cp!='\'') return 0;
        static const char * suffix[]={"s","t","re","ve","m","ll","d"};
        for(const char * x:suffix){size_t m=0;while(x[m])++m; if(p+1+m>n) continue; bool ok=true;
            for(size_t k=0;k<m;++k){uint32_t c=rs[p+1+k].cp; if(c>127||char(std::tolower(int(c)))!=x[k]){ok=false;break;}} if(ok) return 1+m;}
        return 0;
    };
    // A / B from s (s already past any prefix); returns the end or 0 when the alternative does not match here
    auto alt_a=[&](size_t s)->size_t{
        size_t u=0; while(up(s+u))++u;
        for(size_t k=u+1;k-->0;){ size_t l=0; while(lo(s+k+l))++l; if(l){ size_t e=s+k+l; return e+contraction(e);} }
        return 0;
    };
    auto alt_b=[&](size_t s)->size_t{
        size_t u=0; while(up(s+u))++u;
        if(!u) return 0;
        size_t e=s+u; while(lo(e))++e;
        return e+contraction(e);
    };
    size_t i=0;
    while(i<n) {
        size_t j=0;
        for(int alt=0; alt<2 && !j; ++alt){  // A then B, each with the optional prefix tried first
            auto f=[&](size_t s){return alt==0?alt_a(s):alt_b(s);};
            if(prefix_ok(i)) j=f(i+1);
            if(!j) j=f(i);
        }
        if(!j && num(i)) { j=i; while(j<n&&j<i+3&&num(j))++j; }
        if(!j) {
            for(size_t s : {i+1, i}) {
                if(s==i+1 && !(rs[i].cp==' ')) continue;
                if(s<n && !space(s) && !letter(s) && !num(s)) { size_t e=s; while(e<n&&!space(e)&&!letter(e)&&!num(e))++e; while(e<n&&(nl(e)||rs[e].cp=='/'))++e; j=e; break; }
            }
        }
        if(!j && space(i)) {
            size_t w=i; while(space(w))++w;
            size_t last=n; for(size_t p=i;p<w;++p) if(nl(p)) last=p;
            if(last<n) j=last+1;                          // E
            else if(w==n) j=w;                             // F at the end
            else if(w-i>=2) j=w-1;                         // F: the last space goes with the next token
            else j=w;                                      // G
        }
        if(!j) j=i+1;
        out.push_back(text.substr(rs[i].begin, rs[j-1].end-rs[i].begin)); i=j;
    }
    return out;
}

uint8_t Tokenizer::category(uint32_t cp) const {
    auto it=std::upper_bound(unicode_ranges_.begin(),unicode_ranges_.end(),cp,[](uint32_t x,const auto & r){return x<std::get<0>(r);});
    if (it==unicode_ranges_.begin()) return 0;
    --it; return cp<=std::get<1>(*it)?std::get<2>(*it):0;
}
bool Tokenizer::is_letter(uint32_t cp) const { const auto c=category(cp); return (c&3)!=0; }

std::vector<std::string> Tokenizer::pretokenize(const std::string & text) const {
    if (o200k_) return pretokenize_o200k(text);
    std::vector<std::string> out; const auto rs=runes(text); size_t i=0;
    auto has=[&](size_t p,uint8_t mask){return p<rs.size()&&(category(rs[p].cp)&mask)!=0;};
    const uint8_t let=marks_in_letters_?(1|2):1;  // letter-run class
    auto word=[&](size_t p){return has(p,1|4);};      // excluded from the optional prefix before a letter run
    auto nopunct=[&](size_t p){return has(p,let|4);}; // excluded from punctuation runs
    auto slice=[&](size_t a,size_t b){return text.substr(rs[a].begin,rs[b-1].end-rs[a].begin);};
    while(i<rs.size()) {
        size_t j=i;
        if (rs[i].cp=='\'') {
            static const char * suffix[]={"s","t","re","ve","m","ll","d"};
            for (const char * x:suffix) { size_t n=0; while(x[n])++n; if(i+1+n>rs.size())continue; bool ok=true; for(size_t k=0;k<n;++k){uint32_t c=rs[i+1+k].cp; if(c>127||char(std::tolower(int(c)))!=x[k]){ok=false;break;}} if(ok){j=i+1+n;break;} }
        }
        if(j==i) {
            size_t p=i;
            if(!word(p)&&rs[p].cp!='\r'&&rs[p].cp!='\n'&&p+1<rs.size()&&has(p+1,let))++p;
            if(has(p,let)){j=p+1;while(has(j,let))++j;}
        }
        if(j==i && has(i,4)) { j=i+1; if(digits3_) while(j<i+3&&has(j,4))++j; }  // \p{N} (Qwen) or \p{N}{1,3} (GLM-4 / 5)
        if(j==i) {
            size_t p=i;
            if(rs[p].cp==' '&&p+1<rs.size()&&!has(p+1,8)&&!nopunct(p+1))++p;
            if(p<rs.size()&&!has(p,8)&&!nopunct(p)){j=p+1;while(j<rs.size()&&!has(j,8)&&!nopunct(j))++j;while(j<rs.size()&&(rs[j].cp=='\r'||rs[j].cp=='\n'))++j;}
        }
        if(j==i && has(i,8)) {
            size_t k=i;while(has(k,8))++k;
            size_t last_nl=k;for(size_t p=i;p<k;++p)if(rs[p].cp=='\r'||rs[p].cp=='\n')last_nl=p;
            if(last_nl<k)j=last_nl+1;                  // \s*[\r\n]+ : through the run's last newline
            else if(k==rs.size()||k-i==1)j=k;          // \s+(?!\S) at the end, or a lone \s via \s+
            else j=k-1;                                // \s+(?!\S) : the last space goes with the next token
        }
        if(j==i) j=i+1;
        out.push_back(slice(i,j)); i=j;
    }
    return out;
}

std::vector<uint32_t> Tokenizer::encode_piece(const std::string & piece) const {
    std::vector<std::string> symbols;
    for (unsigned char b:piece) { std::string s; append_utf8(s,byte_to_code(b)); symbols.push_back(std::move(s)); }
    while(symbols.size()>1) {
        uint32_t best=UINT32_MAX; size_t at=symbols.size();
        for(size_t i=0;i+1<symbols.size();++i){auto it=merge_rank_.find(symbols[i]+char(0)+symbols[i+1]); if(it!=merge_rank_.end()&&it->second<best){best=it->second;at=i;}}
        if(at==symbols.size())break;
        symbols[at]+=symbols[at+1]; symbols.erase(symbols.begin()+at+1);
    }
    std::vector<uint32_t> ids; ids.reserve(symbols.size());
    for(const auto & s:symbols){auto it=piece_to_id_.find(s);if(it==piece_to_id_.end())throw std::runtime_error("tokenizer has no byte-fallback piece");ids.push_back(it->second);}
    return ids;
}

std::vector<uint32_t> Tokenizer::encode(const std::string & text) const {
    std::vector<uint32_t> ids; size_t cursor=0;
    while(cursor<text.size()) {
        size_t pos=std::string::npos, end=0; std::string found;
        for(const auto & [sp,id]:special_to_id_){(void)id;auto p=text.find(sp,cursor);if(p<pos){pos=p;end=p+sp.size();found=sp;}}
        if(pos==std::string::npos){for(const auto & part:pretokenize(text.substr(cursor))){auto x=encode_piece(part);ids.insert(ids.end(),x.begin(),x.end());}break;}
        if(pos>cursor) for(const auto & part:pretokenize(text.substr(cursor,pos-cursor))){auto x=encode_piece(part);ids.insert(ids.end(),x.begin(),x.end());}
        ids.push_back(special_to_id_.at(found)); cursor=end;
    }
    return ids;
}
std::string Tokenizer::decode(const std::vector<uint32_t> & ids) const {
    std::string out;
    for(uint32_t id:ids){if(id>=id_to_piece_.size()||id_to_piece_[id].empty())continue; const auto & s=id_to_piece_[id]; bool special=false;for(const auto & p:special_to_id_)if(p.first==s){special=true;break;} if(special)continue;
        for(const auto & r:runes(s)){int b=code_to_byte(r.cp);if(b>=0)out.push_back(char(b));}
    }
    return out;
}
}
