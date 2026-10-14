#include "../src/tokenizer_path.hpp"
#include "serve/http.hpp"
#include "serve/scheduler.hpp"
#include "serve/session.hpp"
#include "serve/api_adapters.hpp"
#include "serve/jinja_template.hpp"
#include "src/gguf.hpp"
#include "src/flags_gen.hpp"
#include "src/formats/kv_format.hpp"
#include "src/kv_store.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iostream>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace {
constexpr uint64_t MAX_HEADER = 64 * 1024;
constexpr uint64_t MAX_BODY = 64 * 1024 * 1024;  // base64 images
volatile std::sig_atomic_t running = 1;
int listener = -1;

void stop_server(int) {
    running = 0;
    if (listener >= 0) ::close(listener);
}

bool receive_request(int fd, tqserve::Request & request, std::string & error, int & status) {
    tqserve::RequestReader reader(MAX_BODY, MAX_HEADER);
    uint8_t chunk[8192];
    while (!reader.complete() && !reader.failed()) {
        const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) {
            error = n == 0 ? "incomplete request" : std::strerror(errno);
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
        const ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
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
    const int gai = ::getaddrinfo(host.c_str(), nullptr, &hints, &result);
    if (gai != 0) throw std::runtime_error(std::string("invalid bind host: ") + gai_strerror(gai));
    int fd = -1;
    for (auto * it = result; it; it = it->ai_next) {
        fd = ::socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if (fd < 0) continue;
        int one = 1;
        ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        auto address = *reinterpret_cast<sockaddr_in *>(it->ai_addr);
        address.sin_port = htons(port);
        if (::bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0 && ::listen(fd, 16) == 0) break;
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(result);
    return fd;
}

void usage() {
    std::cerr << "usage: tqserve (--model-dir DIR | --model FILE [--tokenizer FILE]) [--host IPv4] [--port N] [--context N] [--max-tokens N] [--reasoning on|off] [--reasoning-effort low|medium|xhigh] [--reasoning-budget N (-1 unlimited, 0 end now; alias --thinking-budget)] [--reasoning-budget-message TEXT] [--reasoning-answer-reserve N (default 8192; 0 = llama.cpp: budget counts inside max_tokens)] [--jinja (default) | --no-jinja] [--chat-template-file FILE] [--chat-template-kwargs JSON] [--mmproj FILE [--image-min-tokens N] [--image-max-tokens N]] "
                 "[--system TEXT] [--name NAME] [--api-key KEY | --api-key-file FILE] [--np N] [--draft-max N (default: DFlash2 7, DSpark 3)] "
                 "[--repeat-limit N (default 256, 0 off)] [--list-kv] [--list-flags]\n"
              << tqllm::engine_options_usage()
              << "  GET /admin/slots: allocated / active limit / busy / queued; POST /admin/slots {\"slots\": k}: live parallelism (np > 1)\n"
                 "  --np N: serve N requests concurrently (batched verify; drafts with -md, else one token per step); --context is split evenly across them\n";
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

} // namespace

int main(int argc, char ** argv) {
    tqllm::check_env_flags("tqserve");
    tqserve::free_mb_probe() = [] {
        size_t free = 0, total = 0;
        cudaMemGetInfo(&free, &total);
        return free >> 20;
    };
    std::string model_dir, model_path, tokenizer_path, host = "127.0.0.1", name = "tqllm";
    std::string system;  // none by default, like the Qwen3.5/3.8 chat template (--system sets one)
    std::string api_key;
    bool reasoning = false;
    std::string effort = "xhigh";  // default reasoning_effort when a request does not set one
    int32_t thinking_budget = -1;  // default reasoning budget (llama.cpp --reasoning-budget): -1 unlimited, 0 end immediately, N tokens
    std::string budget_message;    // text forced before the end-of-thinking tag when the budget runs out (default none)
    uint32_t answer_reserve = 8192;  // with a thinking budget, max_tokens is raised to at least budget + this, so the answer is not cut off
    uint32_t port = 8086, context = 2048, max_tokens = 512, np = 1;
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
                std::fputs(tqllm::kv_format_listing(16, 4, 256).c_str(), stdout);
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
            else if (arg == "--np") np = static_cast<uint32_t>(std::stoul(value()));
            else if (arg == "--mmproj") mmproj = value();
            else if (arg == "--image-min-tokens") image_min_tokens = std::stoi(value());
            else if (arg == "--image-max-tokens") image_max_tokens = std::stoi(value());
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

        std::cerr << "tqserve: loading model from " << model_path << "\n";
        tqllm::Tokenizer tokenizer;
        tokenizer.load(tokenizer_path);
        tqserve::Vision vis;
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
        const auto model_ptr = std::make_unique<tqllm::Model>(model_path, tokenizer_path, context / np, int(np), engine);
        const uint32_t slot_context = model_ptr->context();  // the memory fit may have halved it
        tqllm::Model & model = *model_ptr;
        std::unique_ptr<tqserve::Scheduler> scheduler;
        if (np > 1) scheduler = std::make_unique<tqserve::Scheduler>(tokenizer, model, slot_context);
        std::atomic<int> inflight{0};
        listener = bind_socket(host, static_cast<uint16_t>(port));
        if (listener < 0) throw std::runtime_error("socket bind/listen failed: " + std::string(std::strerror(errno)));
        std::signal(SIGINT, stop_server);
        std::signal(SIGTERM, stop_server);
        std::signal(SIGPIPE, SIG_IGN);
    tqserve::PromptRenderer renderer;
    tqllm::Json default_kwargs;
    if (use_jinja || !template_file.empty()) {
        std::string src;
        if (!template_file.empty()) {
            std::ifstream tf(template_file, std::ios::binary);
            if (!tf) { std::cerr << "tqserve: cannot read " << template_file << "\n"; return 2; }
            src.assign(std::istreambuf_iterator<char>(tf), std::istreambuf_iterator<char>());
        } else if (!model_path.empty()) {
            // a .tqfn manifest has no metadata of its own: the chat template comes from the tokenizer GGUF
            const bool manifest = model_path.size() > 5 && model_path.compare(model_path.size() - 5, 5, ".tqfn") == 0;
            const auto g = tqllm::gguf_open(manifest ? tokenizer_path : model_path);
            if (const auto it = g.kv.find("tokenizer.chat_template"); it != g.kv.end()) src = it->second.str;
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
            if (!scheduler && model.kv_idle_pending()) {  // idle: compact one old block, unless a client is waiting
                pollfd pf{listener, POLLIN, 0};
                if (::poll(&pf, 1, 20) == 0) {
                    model.kv_idle_step();
                    continue;
                }
            }
            const int client = ::accept(listener, nullptr, nullptr);
            if (client < 0) {
                if (!running || errno == EINTR || errno == EBADF) break;
                std::cerr << "tqserve: accept failed: " << std::strerror(errno) << "\n";
                continue;
            }
            auto handle = [&, client] {
            timeval timeout{30, 0};
            ::setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            tqserve::Request request;
            std::string error;
            int status = 400;
            if (!receive_request(client, request, error, status)) {
                send_all(client, response(status, tqserve::json_error(error)));
                ::close(client);
                return;
            }
            if (!api_key.empty() && request.authorization != "Bearer " + api_key) {
                send_all(client, response(401, tqserve::json_error("invalid or missing API key")));
                ::close(client);
                return;
            }

            std::string body;
            status = 200;
            if (request.method == "GET" && request.path == "/health") {
                std::string fit;  // what the memory fit chose (KV, context, buffers); quotes cannot occur in it
                for (char ch : model.fit_note()) if (ch != '"' && ch != '\\') fit += ch;
                body = "{\"status\":\"ok\",\"engine\":\"" TQLLM_COMMIT "\",\"fit\":\"" + fit + "\"}";
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
                auto parsed = adapt_error.empty()
                    ? tqserve::parse_request(in, max_tokens, system, chat, reasoning, effort, renderer ? &renderer : nullptr, &default_kwargs)
                    : tqserve::ChatRequest{};
                if (!adapt_error.empty()) { parsed.ok = false; parsed.error = adapt_error; }
                if (parsed.ok && parsed.thinking) {
                    if (!parsed.budget_set) parsed.thinking_budget = thinking_budget;
                    if (!parsed.message_set) parsed.budget_message = budget_message;
                    if (parsed.thinking_budget > 0 && answer_reserve > 0) {  // keep room for the answer after the thinking budget
                        const uint64_t want = static_cast<uint64_t>(parsed.thinking_budget) + answer_reserve;
                        if (parsed.max_tokens < want) parsed.max_tokens = static_cast<uint32_t>(std::min<uint64_t>(want, 0xFFFFFFFFull));
                    }
                    if (parsed.thinking_budget == 0) {  // budget 0: end thinking immediately (llama.cpp): force message + end tag
                        parsed.prompt_text += parsed.budget_message + "</think>";
                        parsed.thinking = false;
                    }
                }
                if (!parsed.ok) {
                    status = 400;
                    body = tqserve::json_error(parsed.error);
                } else if (parsed.stream) {
                    send_all(client, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                                     "Connection: close\r\n\r\n");
                    auto raw = [&](const std::string & d) { return send_all(client, d); };
                    std::unique_ptr<tqserve::ResponsesSse> rsse;
                    std::unique_ptr<tqserve::MessagesSse> msse;
                    if (api == tqserve::Api::Responses) { rsse = std::make_unique<tqserve::ResponsesSse>(name, raw); rsse->begin(); }
                    if (api == tqserve::Api::Messages) { msse = std::make_unique<tqserve::MessagesSse>(name, raw); msse->begin(); }
                    auto send = [&](const std::string & d) { return rsse ? rsse->feed(d) : msse ? msse->feed(d) : raw(d); };
                    auto run_stream = [&](const tqserve::ChatRequest & r, tqserve::StreamWriter & w) {
                        auto on_tokens = [&](const std::vector<uint32_t> & ids) { return w.update(ids); };
                        return guarded([&] {
                            return scheduler ? scheduler->submit_stream(r, on_tokens)
                                             : tqserve::decode_stream(tokenizer, model, r, slot_context, on_tokens);
                        });
                    };
                    if (parsed.thinking && parsed.thinking_budget > 0 && static_cast<uint32_t>(parsed.thinking_budget) < parsed.max_tokens) {
                        auto phase1 = parsed;  // thinking budget, streamed: phase 1 deltas, then phase 2 continues the answer
                        phase1.max_tokens = parsed.thinking_budget;
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
                            phase2.max_tokens = parsed.max_tokens - std::min(parsed.max_tokens, first.completion_tokens);
                            phase2.prompt_text += closed ? first.reasoning + "\n</think>\n\n" + first.text
                                                         : first.reasoning + "</think>";  // llama.cpp: message + end tag forced
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
                    }
                    ::close(client);
                    return;
                } else {
                    auto run = [&](const tqserve::ChatRequest & r) {
                        return guarded([&] { return scheduler ? scheduler->submit(r) : tqserve::decode(tokenizer, model, r, slot_context); });
                    };
                    tqserve::DecodeResult generated;
                    if (parsed.thinking && parsed.thinking_budget > 0 && static_cast<uint32_t>(parsed.thinking_budget) < parsed.max_tokens) {
                        // thinking budget (Qwen's method): think for at most N tokens; if the block is still open, close it with
                        // the documented sentence and continue the answer from prompt + thinking (one extra prefill)
                        auto phase1 = parsed;
                        phase1.max_tokens = parsed.thinking_budget;
                        generated = run(phase1);
                        const bool closed = !generated.text.empty() || generated.finish_reason == "stop";
                        if (generated.error.empty() && generated.finish_reason == "length") {
                            auto phase2 = parsed;
                            phase2.thinking = false;
                            phase2.max_tokens = parsed.max_tokens - std::min(parsed.max_tokens, generated.completion_tokens);
                            if (!closed) generated.reasoning = tqserve::trim_incomplete_utf8(generated.reasoning) + parsed.budget_message;
                            phase2.prompt_text += generated.reasoning;
                            phase2.prompt_text += closed ? "\n</think>\n\n" + generated.text : std::string("</think>");
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
                    }
                    if (!generated.error.empty()) {
                        status = generated.error.rfind(kFailed, 0) == 0 ? 500 : 400;
                        body = tqserve::json_error(generated.error, status == 500 ? "server_error" : "invalid_request_error");
                    } else {
                        body = tqserve::completion_json(name, generated, chat);
                        if (api == tqserve::Api::Responses) body = tqserve::chat_to_responses(body, name);
                        else if (api == tqserve::Api::Messages) body = tqserve::chat_to_messages(body, name);
                    }
                }
                if (status != 200 && api == tqserve::Api::Messages) body = tqserve::messages_error(body);
            } else {
                const bool known_path = request.path == "/health" || request.path == "/v1/models" || request.path == "/admin/slots" ||
                    request.path == "/v1/models/" + name || tqserve::api_for_path(request.path) != tqserve::Api::None;
                status = known_path ? 405 : 404;
                body = tqserve::json_error(known_path ? "method not allowed" : "not found");
            }
            send_all(client, response(status, body));
            ::close(client);
            };
            if (scheduler) {
                ++inflight;
                std::thread([handle, &inflight] {
                    handle();
                    --inflight;
                }).detach();
            } else {
                handle();
            }
        }
        scheduler.reset();  // fails queued and running requests so their connection threads can answer
        for (int i = 0; i < 500 && inflight > 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (listener >= 0) ::close(listener);
        listener = -1;
        return 0;
    } catch (const std::exception & e) {
        if (listener >= 0) ::close(listener);
        std::cerr << "tqserve: " << e.what() << "\n";
        return 1;
    }
}
#else
int main() {
    std::cerr << "tqserve requires Linux/POSIX sockets and a CUDA-enabled build\n";
    return 1;
}
#endif
