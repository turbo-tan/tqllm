#include "tokenizer_path.hpp"
#include "serve/http.hpp"
#include "serve/monitor_server.hpp"
#include "serve/scheduler.hpp"
#include "serve/peer.hpp"
#include "serve/net.hpp"
#include "serve/session.hpp"
#include "serve/api_adapters.hpp"
#include "serve/jinja_template.hpp"
#include "serve/embeddings.hpp"
#include "gguf.hpp"
#include "flags_gen.hpp"
#if TQLLM_WITH_CUDA
#include "kv_format.hpp"
#include "models/qwen/qwen35/pplx_embed.hpp"
#include "plugin_format.hpp"
#endif
#include "kv_store.hpp"
#include "load_guard.hpp"
#include "device_free.hpp"

#if TQLLM_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <filesystem>
#include <cerrno>
#include <cstdio>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iostream>
#include <atomic>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#include <chrono>
#include <memory>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <stdexcept>
#include <string>
#include <vector>

#include <cstdlib>

namespace {
constexpr uint64_t MAX_HEADER = 64 * 1024;
constexpr uint64_t MAX_BODY = 64 * 1024 * 1024;  // base64 images
volatile std::sig_atomic_t running = 1;
int listener = -1;

// first INT / TERM: stop accepting, let the running request finish; a second one exits at once (a 64K prompt on a
// 16 GB card can run for an hour and the server could not be stopped without SIGKILL)
void stop_server(int) {
    if (!running) {
        static const char msg[] = "tqserve: second stop signal, exiting now\n";
        std::fwrite(msg, 1, sizeof msg - 1, stderr);
        std::_Exit(130);
    }
    running = 0;
    if (listener >= 0) tqnet::close(listener);
}

bool receive_request(int fd, tqserve::Request & request, std::string & error, int & status) {
    tqserve::RequestReader reader(MAX_BODY, MAX_HEADER);
    uint8_t chunk[8192];
    while (!reader.complete() && !reader.failed()) {
        const ssize_t n = tqnet::recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) {
            error = n == 0 ? "incomplete request" : "receive failed (error " + std::to_string(tqnet::last_error()) + ")";
            status = 400;
            return false;
        }
        reader.append(chunk, static_cast<size_t>(n));
    }
    request = reader.request();
    if (!request.ok()) {
        error = request.error;
        status = request.error == "header block too large" ? 431 :
                 request.error == "content-length above body limit" || request.error == "request too large" ? 413 : 400;
        return false;
    }
    return true;
}

bool send_all(int fd, const std::string & data) {
    size_t sent = 0;
    while (sent < data.size()) {
        const ssize_t n = tqnet::send(fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

std::string response(int status, const std::string & body) {
    const char * reason = status == 200 ? "OK" : status == 400 ? "Bad Request" : status == 401 ? "Unauthorized" :
        status == 404 ? "Not Found" : status == 405 ? "Method Not Allowed" : status == 413 ? "Payload Too Large" :
        status == 431 ? "Request Header Fields Too Large" : "Internal Server Error";
    return "HTTP/1.1 " + std::to_string(status) + " " + reason +
           "\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: " + std::to_string(body.size()) +
           "\r\nConnection: close\r\n\r\n" + body;
}

int bind_socket(const std::string & host, uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    addrinfo * result = nullptr;
    tqnet::startup();
    const int gai = ::getaddrinfo(host.c_str(), nullptr, &hints, &result);
    if (gai != 0) throw std::runtime_error(std::string("invalid bind host: ") + gai_strerror(gai));
    int fd = -1;
    for (auto * it = result; it; it = it->ai_next) {
        fd = tqnet::socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if (fd < 0) continue;
        tqnet::set_reuse_address(fd);
        auto address = *reinterpret_cast<sockaddr_in *>(it->ai_addr);
        address.sin_port = htons(port);
        if (::bind(tqnet::raw(fd), reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0 && ::listen(tqnet::raw(fd), 16) == 0) break;
        tqnet::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(result);
    return fd;
}

void usage() {
    std::cerr << "usage: tqserve (--model-dir DIR | --model FILE [--tokenizer FILE]) [--host IPv4] [--port N] [--context N] [--max-tokens N] [--temperature|--temp T --top-p P --top-k K --presence-penalty X --repeat-penalty R --frequency-penalty F (defaults for requests that set none; default greedy, no penalty)] [--reasoning on|off] [--reasoning-effort low|medium|xhigh] [--reasoning-budget N (-1 unlimited, 0 end now; alias --thinking-budget)] [--reasoning-budget-message TEXT] [--reasoning-answer-reserve N (default 8192; 0 = llama.cpp: budget counts inside max_tokens)] [--jinja (default) | --no-jinja] [--chat-template-file FILE] [--chat-template-kwargs JSON] [--mmproj FILE [--image-min-tokens N] [--image-max-tokens N]] "
                 "[--system TEXT] [--name NAME] [--api-key KEY | --api-key-file FILE] [--np N] [--draft-max N (default: DFlash2 7, DSpark 3)] "
                 "[--monitor-port N (optional separate dashboard port; default: API port)] [--repeat-limit N (default 256, 0 off)] [--peer http://HOST:PORT ...] [--list-kv] [--list-flags]\n"
              << tqllm::engine_options_usage()
              << "  GET /admin/slots: allocated / active limit / busy / queued; POST /admin/slots {\"slots\": k}: live parallelism (np > 1)\n"
                 "  --np N: serve N requests concurrently (batched verify; drafts with -md, else one token per step); --context is split evenly across them\n"
                 "  --peer URL (repeatable, needs --np > 1): other tqserve boxes with the same model; each chat request goes to the box expected to\n"
                 "    answer first (free slots, decoding load, the K/V a slot already holds for the prompt); POST /admin/route reports a box's status\n";
}
// A generation that throws (e.g. a kernel launch that fails on a nearly full device) fails its own request only: the next
// request starts at position 0 and resets the sequence state.
const char * const kFailed = "generation failed: ";
template <class F> tqserve::DecodeResult guarded(F && f) {
    try {
        return f();
    } catch (const std::exception & e) {
        std::cerr << "tqserve: " << kFailed << e.what() << "\n";
        tqserve::DecodeResult r;
        r.error = std::string(kFailed) + e.what();
        return r;
    }
}

#if TQLLM_WITH_CUDA
// A contextual embedding GGUF (pplx-embed-v2-context): POST /v1/embeddings only. Connections run on their own threads;
// tokenizing runs in parallel, the forward passes one request at a time on the one GPU.
int serve_embeddings(const std::string & model_path, const tqllm::Tokenizer & tokenizer, const std::string & host, int port,
                     const std::string & name, const std::string & api_key) {
    constexpr size_t kEmbedBatchTokens = 65536;
    auto model = tqllm::PplxEmbedder::load(model_path, 0, tqllm::PplxEmbedder::Mode::fast);
    {
        const auto t0 = std::chrono::steady_clock::now();
        const size_t n = tokenizer.warm_cache();
        std::cerr << "tqserve: tokenizer piece cache warmed (" << n << " pieces, "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s)\n";
    }
    {  // one short document and one query at load: lazy CUDA module loading, cuBLAS setup and the activation buffers
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<tqllm::EmbedInput> warm;
        warm.push_back(tqllm::embed_document_input(tokenizer, model->prompt(), {"Warm-up document.", "Second chunk."}));
        model->embed_many(warm);
        warm.clear();
        warm.push_back(tqllm::embed_query_input(tokenizer, model->prompt(), "warm-up query"));
        model->embed_many(warm);
        std::cerr << "tqserve: embedder warmed (" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s)\n";
    }
    listener = bind_socket(host, static_cast<uint16_t>(port));
    if (listener < 0) throw std::runtime_error("socket bind/listen failed (error " + std::to_string(tqnet::last_error()) + ")");
    std::signal(SIGINT, stop_server);
    std::signal(SIGTERM, stop_server);
#ifdef SIGPIPE
    std::signal(SIGPIPE, SIG_IGN);
#endif
    std::cerr << "tqserve: engine turbo-tan/tqllm " TQLLM_COMMIT "\n";
    std::cerr << "tqserve: embeddings on " << host << ':' << port << " (model=" << name << ", dim " << model->dim()
              << ", POST /v1/embeddings)\n";
    std::string models = "{\"object\":\"list\",\"data\":[{\"id\":";
    tqserve::jstr(models, name);
    models += ",\"object\":\"model\",\"owned_by\":\"tqllm\"}]}";
    std::mutex gpu;
    std::atomic<int> inflight{0};
    // one thread per connection (at most 64): a slow client, or a long request, never holds up /health or another accept
    auto handle = [&](int client) {
            tqnet::set_recv_timeout(client, 30);
            tqserve::Request request;
            std::string error;
            int status = 400;
            std::string body;
            if (!receive_request(client, request, error, status)) {
                body = tqserve::json_error(error);
            } else if (!api_key.empty() && request.authorization != "Bearer " + api_key) {
                status = 401; body = tqserve::json_error("invalid or missing API key");
            } else if (request.method == "GET" && request.path == "/health") {  // never waits for the GPU
                status = 200; body = "{\"status\":\"ok\"}";
            } else if (request.method == "GET" && request.path == "/v1/models") {
                status = 200; body = models;
            } else if (request.path != "/v1/embeddings") {
                status = 404; body = tqserve::json_error("not found: this server serves an embedding model (POST /v1/embeddings)");
            } else if (request.method != "POST") {
                status = 405; body = tqserve::json_error("POST /v1/embeddings");
            } else {
                const auto t_start = std::chrono::steady_clock::now();
                auto ms_since = [](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); };
                const auto r = tqserve::parse_embed_request(std::string(request.body.begin(), request.body.end()));
                const std::string dim_error = r.error.empty() ? tqserve::check_dimensions(r, model->dim()) : std::string();
                if (!r.error.empty() || !dim_error.empty()) {
                    status = 400; body = tqserve::json_error(r.error.empty() ? dim_error : r.error);
                } else {
                    try {
                        std::vector<std::vector<float>> vectors;
                        // tokenize the entries on worker threads (the tokenizer is read-only and its piece cache is shared)
                        std::vector<tqllm::EmbedInput> ins(r.docs.size());
                        std::vector<std::exception_ptr> errs(r.docs.size());
                        {
                            const size_t nt = std::min<size_t>(r.docs.size(), std::max(1u, std::thread::hardware_concurrency()));
                            std::vector<std::thread> pool;
                            for (size_t t = 0; t < nt; ++t)
                                pool.emplace_back([&, t] {
                                    for (size_t i = t; i < r.docs.size(); i += nt) {
                                        try {
                                            ins[i] = r.query ? tqllm::embed_query_input(tokenizer, model->prompt(), r.docs[i][0])
                                                             : tqllm::embed_document_input(tokenizer, model->prompt(), r.docs[i]);
                                        } catch (...) { errs[i] = std::current_exception(); }
                                    }
                                });
                            for (auto & th : pool) th.join();
                            for (auto & e : errs) if (e) std::rethrow_exception(e);
                        }
                        size_t request_tokens = 0;
                        for (const auto & in : ins) request_tokens += in.ids.size();
                        if (request_tokens > tqserve::kEmbedMaxTokens)
                            throw std::invalid_argument("the request has " + std::to_string(request_tokens) + " tokens; the limit per request is " +
                                                        std::to_string(tqserve::kEmbedMaxTokens));
                        const double t_tok = ms_since(t_start);
                        std::lock_guard<std::mutex> lock(gpu);  // one forward at a time on the one GPU; tokenizing runs outside it
                        const auto t_gpu = std::chrono::steady_clock::now();
                        std::vector<tqllm::EmbedInput> batch;
                        size_t tokens = 0, batch_tokens = 0;
                        auto flush = [&] {
                            for (auto & v : model->embed_many(batch)) vectors.push_back(std::move(v));
                            batch.clear(); batch_tokens = 0;
                        };
                        for (auto & in : ins) {  // entries run as segments of one pass, up to kEmbedBatchTokens (64K; the engine splits passes)
                            if (in.ids.size() > model->max_tokens())
                                throw std::invalid_argument("input of " + std::to_string(in.ids.size()) + " tokens; the model takes " +
                                                            std::to_string(model->max_tokens()));
                            if (!batch.empty() && batch_tokens + in.ids.size() > kEmbedBatchTokens) flush();
                            tokens += in.ids.size();
                            batch_tokens += in.ids.size();
                            batch.push_back(std::move(in));
                        }
                        if (!batch.empty()) flush();
                        const double t_fwd = ms_since(t_gpu);
                        const auto t_out = std::chrono::steady_clock::now();
                        status = 200; body = tqserve::embed_response(r, vectors, model->dim(), name, tokens);
                        std::fprintf(stderr, "tqserve: embeddings %zu inputs, %zu tokens: parse + tokenize %.1f ms, forward %.1f ms, response %.1f ms\n",
                                     r.docs.size(), tokens, t_tok, t_fwd, ms_since(t_out));
                    } catch (const std::invalid_argument & e) {
                        status = 400; body = tqserve::json_error(e.what());
                    } catch (const std::exception & e) {
                        std::cerr << "tqserve: embeddings failed: " << e.what() << "\n";
                        status = 500; body = tqserve::json_error(std::string("embeddings failed: ") + e.what());
                    }
                }
            }
        tqnet::set_send_timeout(client, 30);  // a client that stops reading cannot hold its thread forever
        send_all(client, response(status, body));
        tqnet::close(client);
    };
    while (running) {
        const int client = tqnet::accept(listener);
        if (client < 0) {
            const int e = tqnet::last_error();
            if (!running || tqnet::interrupted(e)) break;
            continue;
        }
        if (inflight >= 64) {
            tqnet::set_send_timeout(client, 5);
            send_all(client, response(503, tqserve::json_error("too many connections")));
            tqnet::close(client);
            continue;
        }
        ++inflight;
        try {
            std::thread([&, client] {
                try { handle(client); }
                catch (const std::exception & e) {
                    std::fprintf(stderr, "tqserve: embeddings request failed: %s\n", e.what());
                    send_all(client, response(500, tqserve::json_error(std::string("request failed: ") + e.what())));
                    tqnet::close(client);
                }
                --inflight;
            }).detach();
        } catch (...) { --inflight; tqnet::close(client); }
    }
    for (; inflight > 0;) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return 0;
}
#endif

} // namespace

void observe_result(const std::shared_ptr<tqserve::Monitor::Request> & observation, const tqserve::DecodeResult & result) {
    if (!observation) return;
    if (!result.error.empty()) { observation->finish("failed"); return; }
    observation->timings(result.prompt_tokens, result.completion_tokens, result.prefill_ms, result.decode_ms,
                         result.draft_n, result.draft_accepted);
}

int main(int argc, char ** argv) {
    tqllm::check_env_flags("tqserve");
#if TQLLM_WITH_CUDA
    tqserve::free_mb_probe() = [] {  // CUDA only: Metal has no free-memory query, and a fake "0 MB" misled (issue #24)
        size_t free = 0, total = 0;
        cudaMemGetInfo(&free, &total);
        return free >> 20;
    };
#endif
    std::string model_dir, model_path, tokenizer_path, host = "127.0.0.1", name = "tqllm";
    std::string system;  // none by default, like the Qwen3.5/3.8 chat template (--system sets one)
    std::string api_key;
    std::vector<tqserve::Peer> peers;  // --peer: other boxes this front door routes requests to
    uint32_t monitor_port = 0;
    bool reasoning = false;
    std::string effort = "xhigh";  // default reasoning_effort when a request does not set one
    int32_t thinking_budget = -1;  // default reasoning budget (llama.cpp --reasoning-budget): -1 unlimited, 0 end immediately, N tokens
    std::string budget_message;    // text forced before the end-of-thinking tag when the budget runs out (default none)
    uint32_t answer_reserve = 8192;  // with a thinking budget, max_tokens is raised to at least budget + this, so the answer is not cut off
    uint32_t port = 8086, context = 2048, max_tokens = 16384, np = 1;
    std::string mmproj;
    bool use_jinja = true;  // llama.cpp default: render the prompt with the model's Jinja chat template; --no-jinja = native Qwen layout
    std::string template_file, template_kwargs_text;
    int image_min_tokens = 8, image_max_tokens = 4096;
    tqllm::EngineOptions engine;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[i];
            };
            if (arg == "--list-flags") {  // every TQLLM_* flag from config/flags.yaml (kind, default, what it does)
                static const char * kinds[] = {"env", "debug", "retired"};
                for (const auto & f : tqllm::flags::ALL)
                    std::printf("%-28s %-12s %-28s %s%s%s\n", f.name, kinds[int(f.kind)], f.def, f.doc,
                                *f.replaced_by ? "  -> use " : "", f.replaced_by);
                return 0;
            }
            if (arg == "--list-kv") {  // registered KV-cache formats with size, scratch need and provenance (Qwen3.8 27B: 16 attention layers, 4 kv heads, head_dim 256)
#if TQLLM_WITH_CUDA
                std::fputs(tqllm::kv_format_listing(16, 4, 256).c_str(), stdout);
#endif
                std::printf("\nKV stores (--kv-store):\n");
                for (const auto & n : tqllm::kv_store_names())
                    std::printf("  %-8s %s\n", n.c_str(), tqllm::kv_store_policy(n)->summary());
                return 0;
            }
            else if (arg == "--model-dir") model_dir = value();
            else if (arg == "--model") model_path = value();
            else if (arg == "--tokenizer") tokenizer_path = value();
            else if (arg == "--host") host = value();
            else if (arg == "--port") port = static_cast<uint32_t>(std::stoul(value()));
            else if (arg == "--context") context = static_cast<uint32_t>(std::stoul(value()));
            else if (arg == "--max-tokens") max_tokens = static_cast<uint32_t>(std::stoul(value()));
            else if (arg == "--temperature" || arg == "--temp") tqserve::sampling_defaults().temperature = std::stof(value());
            else if (arg == "--top-p") tqserve::sampling_defaults().top_p = std::stof(value());
            else if (arg == "--top-k") tqserve::sampling_defaults().top_k = std::stoi(value());
            else if (arg == "--reasoning") {
                const std::string setting = value();
                if (setting != "on" && setting != "off") throw std::invalid_argument("--reasoning must be on or off");
                reasoning = setting == "on";
            }
            else if (arg == "--system") system = value();
            else if (arg == "--name") name = value();
            else if (arg == "--reasoning-effort") effort = value();
            else if (arg == "--reasoning-budget" || arg == "--thinking-budget") {
                thinking_budget = std::stoi(value());
                if (thinking_budget < -1) throw std::invalid_argument("reasoning budget must be >= -1");
            }
            else if (arg == "--reasoning-budget-message") budget_message = value();
            else if (arg == "--reasoning-answer-reserve") answer_reserve = static_cast<uint32_t>(std::stoul(value()));
            else if (arg == "--jinja") use_jinja = true;
            else if (arg == "--no-jinja") use_jinja = false;
            else if (arg == "--chat-template-file") template_file = value();
            else if (arg == "--chat-template-kwargs") template_kwargs_text = value();
            else if (tqllm::parse_engine_option(argc, argv, i, engine)) continue;
            else if (arg == "--draft-max" || arg == "--draft") tqserve::serve_options().draft_max = std::clamp(std::stoi(value()), 1, 15);
            else if (arg == "--repeat-limit") tqserve::serve_options().repeat_limit = static_cast<uint32_t>(std::stoul(value()));
            else if (arg == "--presence-penalty") tqserve::sampling_defaults().presence_penalty = std::stof(value());
            else if (arg == "--repeat-penalty") tqserve::sampling_defaults().repeat_penalty = std::stof(value());
            else if (arg == "--frequency-penalty") tqserve::sampling_defaults().frequency_penalty = std::stof(value());
            else if (arg == "--np") np = static_cast<uint32_t>(std::stoul(value()));
            else if (arg == "--mmproj") mmproj = value();
            else if (arg == "--image-min-tokens") image_min_tokens = std::stoi(value());
            else if (arg == "--image-max-tokens") image_max_tokens = std::stoi(value());
            else if (arg == "--monitor-port") {
                const auto input = value(); size_t consumed = 0;
                const auto v = std::stoul(input, &consumed);
                if (!v || v > 65535 || consumed != input.size()) throw std::invalid_argument("monitor port must be 1..65535");
                monitor_port = static_cast<uint32_t>(v);
            }
            else if (arg == "--peer") {
                tqserve::Peer p;
                if (!tqserve::parse_peer(value(), p)) throw std::invalid_argument("--peer takes http://HOST:PORT");
                peers.push_back(p);
            }
            else if (arg == "--api-key") api_key = value();
            else if (arg == "--api-key-file") {
                std::ifstream key_file(value());
                if (!key_file) throw std::runtime_error("cannot read API key file");
                std::getline(key_file, api_key);
                if (api_key.empty()) throw std::runtime_error("API key file is empty");
            } else { usage(); return 2; }
        }
        if (!model_dir.empty()) {
            if (model_path.empty()) model_path = model_dir + "/weights.tqbin";
            if (tokenizer_path.empty()) tokenizer_path = model_dir + "/tokenizer.tqtok";
        }
        tokenizer_path = tqllm::tokenizer_for(model_path, tokenizer_path);
        if (model_path.empty() || tokenizer_path.empty() || port == 0 || port > 65535 || context == 0 || max_tokens == 0 || name.empty() || np == 0 || np > 64) {
            usage();
            return 2;
        }
        if (!peers.empty() && np < 2) throw std::invalid_argument("--peer needs --np N (N > 1): the router reads the slot scheduler");

        std::cerr << "tqserve: loading model from " << model_path << "\n";
        tqllm::Tokenizer tokenizer;
        tokenizer.load(tokenizer_path);
#if TQLLM_WITH_CUDA
        if (tqllm::PplxEmbedder::is_embedding_gguf(model_path)) return serve_embeddings(model_path, tokenizer, host, port, name, api_key);
#endif
        if (tokenizer.encode("</mm:think>").size() == 1) {  // MiniMax-M3: its own thinking tags; the answer follows the end tag
            auto & so = tqserve::serve_options();
            so.think_open = "<mm:think>"; so.think_close = "</mm:think>"; so.think_off = "</mm:think>";
        }
        std::string plugins_json;  // /health "plugins" (only when --plugin was given)
#if !TQLLM_WITH_CUDA
        if (!engine.plugins.empty()) throw std::runtime_error("--plugin (weight-format plugins) needs a CUDA build");
#else
        for (const auto & path : engine.plugins) {  // before the model: its matrices load with the plugin formats
            const auto info = tqllm::load_plugin(path);
            auto list = [](const std::vector<std::string> & v) { std::string s; for (const auto & x : v) s += (s.empty() ? "\"" : ",\"") + x + "\""; return "[" + s + "]"; };
            plugins_json += std::string(plugins_json.empty() ? "" : ",") + "{\"path\":\"" + path + "\",\"formats\":" + list(info.formats) +
                            ",\"patches\":" + list(info.patches) + ",\"skipped\":" + list(info.skipped) + "}";
            for (const auto & p : info.patches) if (p.find("(not exact)") != std::string::npos) engine.inexact_patches.push_back(p.substr(0, p.find(" (not exact)")));
            std::cerr << "tqserve: plugin " << path << ": " << info.formats.size() << " format(s), " << info.patches.size() << " patch(es)"
                      << (info.skipped.empty() ? "" : ", " + std::to_string(info.skipped.size()) + " skipped (another GPU)") << "\n";
        }
#endif
        tqserve::Vision vis;
#if !TQLLM_WITH_CUDA
        if (!mmproj.empty()) throw std::runtime_error("--mmproj (image input) needs a CUDA build");
#else
        if (!mmproj.empty()) {  // before the model, so the context fit sees the encoder's memory
            const auto pad = tokenizer.encode("<|image_pad|>");
            if (pad.size() != 1) throw std::runtime_error("tokenizer has no <|image_pad|> token");
            vis.pad_id = pad[0];
            if (cudaStreamCreateWithFlags(&vis.stream, cudaStreamNonBlocking) != cudaSuccess)
                throw std::runtime_error("cannot create the vision stream");
            vis.enc = tqllm::VisionEncoder::load(mmproj, vis.stream);
            vis.enc->set_token_limits(image_min_tokens, image_max_tokens);
            tqserve::vision() = &vis;
            std::cerr << "tqserve: vision encoder " << mmproj << " (" << image_min_tokens << ".." << image_max_tokens
                      << " tokens per image)\n";
        }
#endif
#if defined(__GLIBC__)
        // loading allocates and frees ~17 GB of large host buffers per load (tensor reads, repacks): by default each comes
        // back from the kernel as fresh pages (page faults + zeroing: 10 of a 60 s 27B load on a 3090, 11-02); keep freed
        // blocks for reuse while loading, then hand the heap back
        mallopt(M_MMAP_THRESHOLD, 1 << 30);
        mallopt(M_TRIM_THRESHOLD, 1 << 30);
#endif
#if TQLLM_WITH_CUDA
        {  // GPU memory shared with the CPU (GB10): refuse a load that cannot fit beside another model process (.44, 11-08)
            int dev = 0, integrated = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev);
            std::ifstream mi("/proc/meminfo");
            const std::string meminfo((std::istreambuf_iterator<char>(mi)), std::istreambuf_iterator<char>());
            if (integrated && !meminfo.empty()) {
                const uint64_t need = tqllm::model_files_bytes(model_path) + tqllm::model_files_bytes(engine.draft_model);
                const std::string why = tqllm::unified_memory_refusal(tqllm::meminfo_available(meminfo), need);
                if (!why.empty()) throw std::runtime_error(why);
            }
        }
#endif
        const auto model_ptr = std::make_unique<tqllm::Model>(model_path, tokenizer_path, context / np, int(np), engine);
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
        const uint32_t slot_context = model_ptr->context();  // the memory fit may have halved it
        tqllm::Model & model = *model_ptr;
        if (engine.split_layers && engine.tp_rank > 0) {  // --split layer: this rank follows rank 0's server
            std::fprintf(stderr, "tqserve: --split layer rank %d loaded; following rank 0\n", engine.tp_rank);
            model.follow();
            return 0;
        }
        std::unique_ptr<tqserve::Monitor> monitor;
        std::unique_ptr<tqserve::MonitorServer> monitor_server;
        {
            char hostname[256]{}, pci[32]{}; ::gethostname(hostname, sizeof(hostname) - 1);
            int device = 0;
#if TQLLM_WITH_CUDA
            if (cudaGetDevice(&device) == cudaSuccess) cudaDeviceGetPCIBusId(pci, sizeof(pci), device);
#else
            (void)device;
#endif
            monitor = std::make_unique<tqserve::Monitor>(tqserve::Monitor::Info{name, TQLLM_COMMIT, hostname, pci, slot_context, np});
            monitor_server = std::make_unique<tqserve::MonitorServer>(*monitor, host, uint16_t(monitor_port), api_key, pci, uint16_t(port), monitor_port && monitor_port != port);
            std::cerr << "tqserve: monitor http://" << host << ':' << (monitor_port ? monitor_port : port) << "/monitor (read-only)\n";
        }
        std::unique_ptr<tqserve::Scheduler> scheduler;
        if (np > 1) scheduler = std::make_unique<tqserve::Scheduler>(tokenizer, model, slot_context);
        std::atomic<int> inflight{0};
        std::mutex generation_mutex;
        // one slot: after a chat reply the engine may prefill the reply as the client will return it (Engine::prefill_ahead).
        // It runs here under the generation lock, so a request that arrives first is served first (and cancels it); a
        // request that arrives during it waits for the tokens it would prefill itself when the expectation holds.
        struct AheadWorker {
            tqllm::Model & model;
            std::mutex & generation;
            std::mutex m;
            std::condition_variable cv;
            // the expected next prompt, rendered and tokenized on this thread (off the reply's path)
            std::function<std::vector<uint32_t>()> next;
            uint64_t posted = 0;  // counts posts: a post or cancel during the rendering makes that rendering stale
            bool stop = false;
            std::thread t;
            AheadWorker(tqllm::Model & mo, std::mutex & g) : model(mo), generation(g), t([this] { run(); }) {}
            ~AheadWorker() {
                { std::lock_guard<std::mutex> l(m); stop = true; }
                cv.notify_all();
                t.join();
            }
            void post(std::function<std::vector<uint32_t>()> f) {
                { std::lock_guard<std::mutex> l(m); next = std::move(f); ++posted; }
                cv.notify_all();
            }
            void cancel() { std::lock_guard<std::mutex> l(m); next = nullptr; ++posted; }
            void run() {
                for (;;) {
                    {
                        std::unique_lock<std::mutex> l(m);
                        cv.wait(l, [&] { return stop || next; });
                        if (stop) return;
                    }
                    std::vector<uint32_t> p;
                    uint64_t mine = 0;
                    try {
                        std::function<std::vector<uint32_t>()> f;
                        { std::lock_guard<std::mutex> l(m); f = next; mine = posted; }
                        if (f) p = f();
                    } catch (const std::exception & e) {
                        std::cerr << "tqserve: next prompt for the idle prefill: " << e.what() << "\n";
                    }
                    std::lock_guard<std::mutex> g(generation);
                    {
                        std::lock_guard<std::mutex> l(m);
                        if (!next || posted != mine) continue;  // a request came first, or a newer reply was posted
                        next = nullptr;
                    }
                    if (p.empty()) continue;
                    try {
                        model.select_slot(0);
                        model.prefill_ahead(0, p);
                    } catch (const std::exception & e) {
                        std::cerr << "tqserve: idle prefill failed: " << e.what() << "\n";
                    }
                }
            }
        } ahead_worker{model, generation_mutex};
        listener = bind_socket(host, static_cast<uint16_t>(port));
        if (listener < 0) throw std::runtime_error("socket bind/listen failed (error " + std::to_string(tqnet::last_error()) + ")");
        std::signal(SIGINT, stop_server);
        std::signal(SIGTERM, stop_server);
#ifdef SIGPIPE
        std::signal(SIGPIPE, SIG_IGN);
#endif
    tqserve::PromptRenderer renderer;
    tqllm::Json default_kwargs;
    if (use_jinja || !template_file.empty()) {
        std::string src;
        if (!template_file.empty()) {
            std::ifstream tf(template_file, std::ios::binary);
            if (!tf) { std::cerr << "tqserve: cannot read " << template_file << "\n"; return 2; }
            src.assign(std::istreambuf_iterator<char>(tf), std::istreambuf_iterator<char>());
        } else if (!model_path.empty()) {
            // the template is GGUF / Hugging Face metadata; a model file without any (a .tqfn manifest, weights.tqbin, a
            // plugin engine's file) or without tokenizer metadata (MiniMax-M3 from tools/minimax_convert.cpp) takes the
            // --tokenizer GGUF's
            auto metadata = [](const std::string & p) {
                return std::filesystem::is_directory(p) || (p.size() > 5 && p.compare(p.size() - 5, 5, ".gguf") == 0);
            };
            const bool own = metadata(model_path);
            if (own || metadata(tokenizer_path)) {
                const auto g = tqllm::gguf_open(own ? model_path : tokenizer_path);
                if (const auto it = g.kv.find("tokenizer.chat_template"); it != g.kv.end()) src = it->second.str;
            }
            if (src.empty() && own && tokenizer_path != model_path && tokenizer_path.size() > 5 &&
                tokenizer_path.compare(tokenizer_path.size() - 5, 5, ".gguf") == 0) {
                const auto tg = tqllm::gguf_open_shard(tokenizer_path);
                if (const auto it = tg.kv.find("tokenizer.chat_template"); it != tg.kv.end()) src = it->second.str;
            }
        } else if (!model_dir.empty()) {
            std::ifstream tf(model_dir + "/chat_template.jinja", std::ios::binary);
            if (tf) src.assign(std::istreambuf_iterator<char>(tf), std::istreambuf_iterator<char>());
        }
        if (src.empty()) {
            std::cerr << "tqserve: no chat template in the model (GGUF tokenizer.chat_template or <model-dir>/chat_template.jinja); using the native Qwen layout\n";
        } else {
            try {
                auto tpl = std::make_shared<tqserve::JinjaTemplate>(src);
                renderer = [tpl](const tqllm::Json & messages, const tqllm::Json * tools, const tqllm::Json & kwargs) {
                    return tpl->render(messages, tools, true, &kwargs);
                };
            } catch (const std::exception & e) { std::cerr << "tqserve: chat template: " << e.what() << "\n"; return 2; }
            if (!template_kwargs_text.empty()) {
                try { default_kwargs = tqllm::parse_json(template_kwargs_text); } catch (const std::exception & e) { std::cerr << "tqserve: --chat-template-kwargs: " << e.what() << "\n"; return 2; }
            }
            std::cerr << "tqserve: chat template: Jinja (" << src.size() << " bytes), llama.cpp's engine\n";
        }
    }

        std::cerr << "tqserve: engine turbo-tan/tqllm " TQLLM_COMMIT "\n";
        std::cerr << "tqserve: listening on " << host << ':' << port << " (model=" << name
                  << ", context=" << slot_context << (np > 1 ? " per slot, slots=" + std::to_string(np) : std::string())
                  << ")\n";

        while (running) {
            if (!scheduler && inflight == 0 && model.kv_idle_pending()) {  // idle: compact one old block, unless a client is waiting
                auto pf = tqnet::poll_entry(listener, POLLIN);
                if (tqnet::poll(&pf, 1, 20) == 0) {
                    model.kv_idle_step();
                    continue;
                }
            }
            const int client = tqnet::accept(listener);
            if (client < 0) {
                const int e = tqnet::last_error();
                if (!running || tqnet::interrupted(e)) break;
                std::cerr << "tqserve: accept failed (error " << e << ")\n";
                continue;
            }
            auto handle = [&, client] {
            tqnet::set_recv_timeout(client, 30);
            tqserve::Request request;
            std::string error;
            int status = 400;
            if (!receive_request(client, request, error, status)) {
                send_all(client, response(status, tqserve::json_error(error)));
                tqnet::close(client);
                return;
            }
            if (request.path == "/" || request.path == "/monitor" || request.path == "/monitor/" ||
                request.path == "/api/monitor/v1/snapshot" || request.path == "/api/monitor/v1/chat/completions") {
                if (!monitor_server->serve(client, request)) tqnet::close(client);
                return;
            }
            if (!api_key.empty() && request.authorization != "Bearer " + api_key) {
                send_all(client, response(401, tqserve::json_error("invalid or missing API key")));
                tqnet::close(client);
                return;
            }

            // --np 1 (no scheduler): one generation at a time, but /health and /v1/models answer while it runs; every
            // connection has its own thread, so a long generation no longer blocks the accept loop
            static std::mutex single_slot;
            const bool status_path = request.method == "GET" && (request.path == "/health" || request.path == "/v1/models" || request.path == "/v1/models/" + name);
            std::unique_lock<std::mutex> hold(single_slot, std::defer_lock);
            if (!scheduler && !status_path) hold.lock();
            std::string body;
            status = 200;
            if (request.method == "GET" && request.path == "/health") {
                std::string fit;  // what the memory fit chose (KV, context, buffers); quotes cannot occur in it
                for (char ch : model.fit_note()) if (ch != '"' && ch != '\\') fit += ch;
                body = "{\"status\":\"ok\",\"engine\":\"" TQLLM_COMMIT "\",\"context\":" + std::to_string(model.context()) +
                       ",\"fit\":\"" + fit + "\"" +
                       (plugins_json.empty() ? std::string() : ",\"plugins\":[" + plugins_json + "]") + "}";
            } else if (request.path == "/admin/slots" && (request.method == "GET" || request.method == "POST")) {
                // live parallelism: POST {"slots": k} admits at most k requests at once (1 .. --np), no reload
                if (!scheduler) {
                    status = 400;
                    body = tqserve::json_error("start tqserve with --np N (N > 1) to change slots live");
                } else {
                    if (request.method == "POST") {
                        const auto root = tqllm::parse_json(std::string(request.body.begin(), request.body.end()));
                        const auto * k = root.kind == tqllm::Json::Object ? root.get("slots") : nullptr;
                        if (!k || k->kind != tqllm::Json::Number || k->num < 1) {
                            status = 400;
                            body = tqserve::json_error("body must be {\"slots\": N}, N >= 1");
                        } else {
                            scheduler->set_limit(int(k->num));
                        }
                    }
                    if (status == 200) body = scheduler->status_json();
                }
            } else if (request.method == "POST" && request.path == "/admin/route") {
                // a front door's status query (tqserve --peer): the body is the rendered prompt
                if (!scheduler) {
                    status = 400;
                    body = tqserve::json_error("routing needs --np N (N > 1)");
                } else {
                    body = tqserve::status_json(scheduler->route_status(std::string(request.body.begin(), request.body.end())));
                }
            } else if (request.method == "GET" && (request.path == "/v1/models" || request.path == "/v1/models/" + name)) {
                body = "{\"object\":\"list\",\"data\":[{\"id\":";
                tqserve::jstr(body, name);
                body += ",\"object\":\"model\",\"owned_by\":\"tqllm\"}]}";
            } else if (request.method == "POST" && request.path == "/v1/messages/count_tokens") {
                body = "{\"input_tokens\":" + std::to_string(request.body.size() / 4) + "}";  // estimate (Claude Code asks)
            } else if (request.method == "POST" && tqserve::api_for_path(request.path) != tqserve::Api::None) {
                // /v1/responses and /v1/messages are rewritten into a chat-completions request and their answers back
                const auto api = tqserve::api_for_path(request.path);
                const bool chat = api != tqserve::Api::Completions;
                std::string in(request.body.begin(), request.body.end()), adapt_error;
                if (api == tqserve::Api::Responses) in = tqserve::responses_to_chat(in, adapt_error);
                else if (api == tqserve::Api::Messages) in = tqserve::messages_to_chat(in, adapt_error);
                std::unique_lock<std::mutex> generation_lock(generation_mutex, std::defer_lock);
                if (!scheduler) generation_lock.lock();
                if (!scheduler) ahead_worker.cancel();
                auto parsed = adapt_error.empty()
                    ? tqserve::parse_request(in, max_tokens, system, chat, reasoning, effort, renderer ? &renderer : nullptr, &default_kwargs)
                    : tqserve::ChatRequest{};
                std::atomic<int> * local_routed = nullptr;  // the router's count for this box, released when the request ends
                struct RoutedDone { std::atomic<int> *& n; ~RoutedDone() { if (n) --*n; } } routed_done{local_routed};
                if (parsed.ok && adapt_error.empty() && scheduler && !peers.empty() && !request.routed) {
                    // multi-box: query every box at once, pick the one expected to answer first (serve/router.hpp)
                    std::vector<tqserve::BoxStatus> boxes(peers.size() + 1);
                    std::vector<std::thread> asks;
                    for (size_t i = 0; i < peers.size(); ++i)
                        asks.emplace_back([&, i] { boxes[i + 1] = tqserve::peer_status(peers[i], parsed.prompt_text, request.authorization, 100); });
                    boxes[0] = scheduler->route_status(parsed.prompt_text);
                    boxes[0].helped = engine.split_experts && engine.tp_rank == 0;
                    for (auto & t : asks) t.join();
                    const double prompt = double(parsed.prompt_text.size()) / scheduler->chars_per_token();
                    const double gen = std::min<double>(parsed.max_tokens, 512);
                    static std::vector<std::atomic<int>> inflight(peers.size() + 1);  // routed here, not finished
                    static std::vector<tqserve::PeerMemory> memory(peers.size() + 1);  // last answer per peer (router.hpp)
                    static std::mutex route_mutex;
                    int pick = 0;
                    {
                        std::lock_guard<std::mutex> lk(route_mutex);
                        for (size_t i = 0; i < boxes.size(); ++i) {
                            boxes[i].inflight = inflight[i].load();
                            if (i > 0) boxes[i] = tqserve::remembered_status(boxes[i], memory[i]);
                        }
                        pick = tqserve::pick_box(boxes, prompt, gen);
                        ++inflight[size_t(pick)];
                        if (tqllm::debug_on("route")) {
                            std::string line = "tqserve: route -> box " + std::to_string(pick) + " |";
                            for (size_t i = 0; i < boxes.size(); ++i) {
                                const auto & b = boxes[i];
                                char buf[160];
                                std::snprintf(buf, sizeof buf, " b%zu ok %d free %d busy %d queued %d inflight %d cost %.2f", i, int(b.ok),
                                              b.free, b.busy, b.queued, b.inflight, b.ok ? tqserve::route_cost(b, prompt, gen) : -1.0);
                                line += buf;
                            }
                            std::fprintf(stderr, "%s\n", line.c_str());
                        }
                    }
                    if (pick > 0) {
                        const auto & peer = peers[size_t(pick - 1)];
                        const bool sent = tqserve::peer_proxy(peer, tqserve::forward_bytes(request, peer), client);
                        --inflight[size_t(pick)];
                        if (sent) { tqnet::close(client); return; }
                        std::fprintf(stderr, "tqserve: peer %s:%u did not take the request; serving it here\n", peer.host.c_str(), unsigned(peer.port));
                        ++inflight[0];
                    }
                    local_routed = &inflight[0];
                }
                if (monitor && parsed.ok) parsed.observation = monitor->begin();
                if (!adapt_error.empty()) { parsed.ok = false; parsed.error = adapt_error; }
                if (parsed.ok && parsed.thinking) {
                    if (!parsed.budget_set) parsed.thinking_budget = thinking_budget;
                    if (!parsed.message_set) parsed.budget_message = budget_message;
                    if (parsed.thinking_budget > 0 && answer_reserve > 0) {  // keep room for the answer after the thinking budget
                        const uint64_t want = static_cast<uint64_t>(parsed.thinking_budget) + answer_reserve;
                        if (parsed.max_tokens < want) parsed.max_tokens = static_cast<uint32_t>(std::min<uint64_t>(want, 0xFFFFFFFFull));
                    }
                    if (parsed.thinking_budget == 0) {  // budget 0: end thinking immediately (llama.cpp): force message + end tag
                        parsed.prompt_text += parsed.budget_message + tqserve::serve_options().think_close;
                        parsed.thinking = false;
                    }
                }
                // --prompt-cache on, after a chat reply: the engine may prefill the reply as the client will return it while the server is idle
                // (Engine::prefill_ahead). Not for tool calls (the next request follows at once) or image prompts.
                auto prefill_ahead = [&](const tqserve::DecodeResult & reply) {
                    if (!engine.prompt_cache || scheduler || !chat || !reply.error.empty() || !reply.tool_calls.empty() || !parsed.images.empty()) return;
                    // rendered on the worker: a non-streamed reply is sent without waiting for it
                    ahead_worker.post([&tokenizer, &renderer, &default_kwargs, body = in, reply, max_tokens, system, reasoning, effort] {
                        const auto prefix = tqserve::next_prompt_prefix(body, reply, tqserve::returns_reasoning(body), max_tokens, system, reasoning,
                                                                        effort, renderer ? &renderer : nullptr, &default_kwargs);
                        return prefix.empty() ? std::vector<uint32_t>{} : tokenizer.encode(prefix);
                    });
                };
                if (!parsed.ok) {
                    status = 400;
                    body = tqserve::json_error(parsed.error);
                } else if (parsed.stream) {
                    send_all(client, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                                     "Connection: close\r\n\r\n");
                    // SSE comments every 10 s until the first token: a long prefill otherwise looks like a dead connection to
                    // clients with a read timeout (agents on 200K contexts)
                    std::mutex ka_mx; std::condition_variable ka_cv; bool ka_stop = false;
                    auto raw = [&](const std::string & d) { std::lock_guard<std::mutex> lk(ka_mx); return send_all(client, d); };
                    std::thread keepalive([&] {
                        std::unique_lock<std::mutex> lk(ka_mx);
                        while (!ka_cv.wait_for(lk, std::chrono::seconds(10), [&] { return ka_stop; }))
                            if (!send_all(client, ": keepalive\n\n")) break;
                    });
                    auto stop_keepalive = [&] { { std::lock_guard<std::mutex> lk(ka_mx); ka_stop = true; } ka_cv.notify_all(); };
                    struct KeepaliveJoin { std::function<void()> stop; std::thread & t; ~KeepaliveJoin() { stop(); if (t.joinable()) t.join(); } } ka_join{stop_keepalive, keepalive};
                    std::unique_ptr<tqserve::ResponsesSse> rsse;
                    std::unique_ptr<tqserve::MessagesSse> msse;
                    if (api == tqserve::Api::Responses) { rsse = std::make_unique<tqserve::ResponsesSse>(name, raw); rsse->begin(); }
                    if (api == tqserve::Api::Messages) { msse = std::make_unique<tqserve::MessagesSse>(name, raw); msse->begin(); }
                    auto send = [&](const std::string & d) { return rsse ? rsse->feed(d) : msse ? msse->feed(d) : raw(d); };
                    auto run_stream = [&](const tqserve::ChatRequest & r, tqserve::StreamWriter & w) {
                        auto on_tokens = [&](const std::vector<uint32_t> & ids) { stop_keepalive(); return w.update(ids); };
                        auto result = guarded([&] {
                            return scheduler ? scheduler->submit_stream(r, on_tokens)
                                             : tqserve::decode_stream(tokenizer, model, r, slot_context, on_tokens);
                        });
                        observe_result(r.observation, result);
                        return result;
                    };
                    if (parsed.thinking && parsed.thinking_budget > 0 && static_cast<uint32_t>(parsed.thinking_budget) < parsed.max_tokens) {
                        auto phase1 = parsed;  // thinking budget, streamed: phase 1 deltas, then phase 2 continues the answer
                        phase1.max_tokens = parsed.thinking_budget;
                        phase1.continuation = 1;
                        tqserve::StreamWriter w1(tokenizer, phase1, name, send);
                        auto first = run_stream(phase1, w1);
                        if (!first.error.empty()) w1.error(first.error);
                        else if (first.finish_reason != "length") w1.close(first);
                        else {
                            const bool closed = !first.text.empty();
                            if (!closed) first.reasoning = tqserve::trim_incomplete_utf8(first.reasoning) + parsed.budget_message;
                            w1.flush(first);
                            auto phase2 = parsed;
                            phase2.thinking = false;
                            phase2.continuation = 2;
                            phase2.max_tokens = parsed.max_tokens - std::min(parsed.max_tokens, first.completion_tokens);
                            phase2.prompt_text += closed ? first.reasoning + "\n" + tqserve::serve_options().think_close + "\n\n" + first.text
                                                         : first.reasoning + tqserve::serve_options().think_close;  // llama.cpp: message + end tag forced
                            tqserve::StreamWriter w2(tokenizer, phase2, name, send);
                            w2.mark_role_sent();
                            auto more = phase2.max_tokens > 0 ? run_stream(phase2, w2) : tqserve::DecodeResult{};
                            if (!more.error.empty()) w2.error(more.error);
                            else {
                                more.completion_tokens += first.completion_tokens;
                                more.prefill_ms += first.prefill_ms;
                                more.decode_ms += first.decode_ms;
                                more.draft_n += first.draft_n;
                                more.draft_accepted += first.draft_accepted;
                                more.prompt_tokens = first.prompt_tokens;
                                w2.close(more);
                            }
                        }
                    } else {
                        tqserve::StreamWriter out(tokenizer, parsed, name, send);
                        const auto generated = run_stream(parsed, out);
                        if (!generated.error.empty()) out.error(generated.error);
                        else out.close(generated);
                        prefill_ahead(generated);
                    }
                    if (!scheduler) model.set_continuation(0);  // a thinking budget's held phase 1 state, if phase 2 did not run
                    if (parsed.observation) parsed.observation->finish();
                    tqnet::close(client);
                    return;
                } else {
                    auto run = [&](const tqserve::ChatRequest & r) {
                        auto result = guarded([&] { return scheduler ? scheduler->submit(r) : tqserve::decode(tokenizer, model, r, slot_context); });
                        observe_result(r.observation, result);
                        return result;
                    };
                    tqserve::DecodeResult generated;
                    if (parsed.thinking && parsed.thinking_budget > 0 && static_cast<uint32_t>(parsed.thinking_budget) < parsed.max_tokens) {
                        // thinking budget (Qwen's method): think for at most N tokens; if the block is still open, close it with
                        // the documented sentence and continue the answer from prompt + thinking (one extra prefill)
                        auto phase1 = parsed;
                        phase1.max_tokens = parsed.thinking_budget;
                        phase1.continuation = 1;
                        generated = run(phase1);
                        const bool closed = !generated.text.empty() || generated.finish_reason == "stop";
                        if (generated.error.empty() && generated.finish_reason == "length") {
                            auto phase2 = parsed;
                            phase2.thinking = false;
                            phase2.continuation = 2;
                            phase2.max_tokens = parsed.max_tokens - std::min(parsed.max_tokens, generated.completion_tokens);
                            if (!closed) generated.reasoning = tqserve::trim_incomplete_utf8(generated.reasoning) + parsed.budget_message;
                            phase2.prompt_text += generated.reasoning;
                            phase2.prompt_text += closed ? "\n" + tqserve::serve_options().think_close + "\n\n" + generated.text : tqserve::serve_options().think_close;
                            auto more = phase2.max_tokens > 0 ? run(phase2) : tqserve::DecodeResult{};
                            if (!more.error.empty()) generated.error = more.error;
                            else {
                                generated.text = (closed ? generated.text : std::string()) + more.text;
                                generated.tool_calls = more.tool_calls;
                                generated.completion_tokens += more.completion_tokens;
                                generated.prefill_ms += more.prefill_ms;
                                generated.decode_ms += more.decode_ms;
                                generated.draft_n += more.draft_n;
                                generated.draft_accepted += more.draft_accepted;
                                generated.finish_reason = more.finish_reason;
                            }
                        }
                    } else {
                        generated = run(parsed);
                        prefill_ahead(generated);
                    }
                    if (!scheduler) model.set_continuation(0);
                    if (!generated.error.empty()) {
                        status = generated.error.rfind(kFailed, 0) == 0 ? 500 : 400;
                        body = tqserve::json_error(generated.error, status == 500 ? "server_error" : "invalid_request_error");
                    } else {
                        body = tqserve::completion_json(name, generated, chat);
                        if (api == tqserve::Api::Responses) body = tqserve::chat_to_responses(body, name);
                        else if (api == tqserve::Api::Messages) body = tqserve::chat_to_messages(body, name);
                    }
                }
                if (parsed.observation) parsed.observation->finish(status == 200 ? "complete" : "failed");
                if (status != 200 && api == tqserve::Api::Messages) body = tqserve::messages_error(body);
            } else {
                const bool known_path = request.path == "/health" || request.path == "/v1/models" || request.path == "/admin/slots" || request.path == "/admin/route" ||
                    request.path == "/v1/models/" + name || tqserve::api_for_path(request.path) != tqserve::Api::None;
                status = known_path ? 405 : 404;
                body = tqserve::json_error(known_path ? "method not allowed" : "not found");
            }
            send_all(client, response(status, body));
            tqnet::close(client);
            };
            if (scheduler || monitor) {
                if (inflight >= 64) {
                    send_all(client,response(503,tqserve::json_error("too many connections"))); tqnet::close(client); continue;
                }
                ++inflight;
                try { std::thread([handle, &inflight, client] {
                    try { handle(); }
                    catch (const std::exception & e) {  // never a silent empty reply: log it and answer 500
                        std::fprintf(stderr, "tqserve: request failed: %s\n", e.what());
                        send_all(client, response(500, tqserve::json_error(std::string("request failed: ") + e.what()))); tqnet::close(client);
                    }
                    catch (...) { std::fputs("tqserve: request failed (unknown exception)\n", stderr); send_all(client, response(500, tqserve::json_error("request failed"))); tqnet::close(client); }
                    --inflight;
                }).detach(); }
                catch (...) { --inflight; tqnet::close(client); }
            } else {
                handle();
            }
        }
        scheduler.reset();  // fails queued and running requests so their connection threads can answer
        for (; inflight > 0;) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (listener >= 0) tqnet::close(listener);
        listener = -1;
        return 0;
    } catch (const std::exception & e) {
        if (listener >= 0) tqnet::close(listener);
        std::cerr << "tqserve: " << e.what() << "\n";
        return 1;
    }
}
