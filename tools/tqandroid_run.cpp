#include "android_model.hpp"
#include "tqllm/tokenizer.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <csignal>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void interrupt(int) { interrupted = 1; }
int number(const std::string & text, int minimum, int maximum) {
    int n = 0;
    auto r = std::from_chars(text.data(), text.data() + text.size(), n);
    if (r.ec != std::errc{} || r.ptr != text.data() + text.size() || n < minimum || n > maximum)
        throw std::invalid_argument("integer out of range: " + text);
    return n;
}
void usage() {
    std::cout << "tqandroid-run --model FILE --prompt TEXT [--raw] [--thinking]\n"
                 "  [--context 4096] [--threads 4] [--max-tokens 128] [--exact]\n"
                 "  [--token-ids-file FILE] [--output-ids FILE] [--logits FILE] [--metrics FILE]\n"
                 "tqandroid-run --model FILE --chat [--context 4096] [--threads 4] [--max-tokens 512] [--exact]\n"
                 "Native tqLLM CPU backend: dense qwen35 GGUF; no CUDA, MLX or llama runtime.\n";
}
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
std::string unescape(const std::string & line) {
    std::string out;
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] != '\\' || i + 1 == line.size()) { out += line[i]; continue; }
        const char c = line[++i];
        out += c == 'n' ? '\n' : c;
    }
    return out;
}
// Line protocol on stdin, one conversation kept in the model state between turns:
//   SYSTEM <text>  set the system prompt and reset;  USER <text>  answer;  RESET
// Text escapes newline as \n and backslash as \\. Each answer streams to stdout and ends with
// 0x1E plus a metrics JSON line. Recurrent state only appends, so overflow restarts from the system prompt.
int chat(tqllm::android_backend::Model & model, tqllm::Tokenizer & tokenizer, int maximum) {
    std::string system, line;
    const std::vector<float> * logits = nullptr;
    auto feed = [&](const std::string & text) {
        for (auto token : tokenizer.encode(text)) logits = &model.forward(token);
    };
    auto restart = [&] {
        model.reset(); logits = nullptr;
        if (!system.empty()) feed("<|im_start|>system\n" + system + "<|im_end|>\n");
    };
    while (!interrupted && std::getline(std::cin, line)) {
        if (line == "RESET") { restart(); std::cout << "\x1e{}\n" << std::flush; continue; }
        if (line.rfind("SYSTEM ", 0) == 0) { system = unescape(line.substr(7)); restart(); std::cout << "\x1e{}\n" << std::flush; continue; }
        if (line.rfind("USER ", 0)) { std::cout << "\x1e{\"error\":\"unknown command\"}\n" << std::flush; continue; }
        const auto prompt = tokenizer.encode("<|im_start|>user\n" + unescape(line.substr(5)) +
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        bool restarted = false;
        if (model.position() + int(prompt.size()) + maximum + 2 > model.context()) { restart(); restarted = true; }
        if (model.position() + int(prompt.size()) + maximum + 2 > model.context()) {
            std::cout << "\x1e{\"error\":\"message exceeds context\"}\n" << std::flush; continue;
        }
        const auto start = Clock::now();
        for (auto token : prompt) logits = &model.forward(token);
        const auto prefilled = Clock::now();
        int generated = 0;
        const char * finish = "length";
        for (; generated < maximum; ++generated) {
            if (interrupted) { finish = "cancelled"; break; }
            const auto token = uint32_t(std::max_element(logits->begin(), logits->end()) - logits->begin());
            if (token == model.eos()) { finish = "eos"; break; }
            std::cout << tokenizer.decode({token}) << std::flush;
            logits = &model.forward(token);
        }
        feed("<|im_end|>\n");
        const auto end = Clock::now();
        const double p = ms(start, prefilled), d = ms(prefilled, end);
        std::cout << "\x1e{\"prompt_tokens\":" << prompt.size() << ",\"generated_tokens\":" << generated
                  << ",\"prefill_tps\":" << (p > 0 ? 1000. * prompt.size() / p : 0.)
                  << ",\"decode_tps\":" << (d > 0 ? 1000. * generated / d : 0.)
                  << ",\"position\":" << model.position() << ",\"restarted\":" << (restarted ? "true" : "false")
                  << ",\"finish_reason\":\"" << finish << "\"}\n" << std::flush;
    }
    return interrupted ? 130 : 0;
}
}
int main(int argc, char ** argv) {
    try {
        std::string path, prompt, ids_path, output_path, logits_path, metrics_path;
        tqllm::android_backend::Options options;
        int maximum = 128;
        bool raw = false, thinking = false, interactive = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (++i >= argc) throw std::invalid_argument("missing value: " + arg);
                return argv[i];
            };
            if (arg == "--help") { usage(); return 0; }
            else if (arg == "--model") path = value();
            else if (arg == "--prompt") prompt = value();
            else if (arg == "--token-ids-file") ids_path = value();
            else if (arg == "--output-ids") output_path = value();
            else if (arg == "--logits") logits_path = value();
            else if (arg == "--metrics") metrics_path = value();
            else if (arg == "--context") options.context = number(value(), 1, 32768);
            else if (arg == "--threads") options.threads = number(value(), 1, 8);
            else if (arg == "--max-tokens") maximum = number(value(), 0, 32768);
            else if (arg == "--raw") raw = true;
            else if (arg == "--thinking") thinking = true;
            else if (arg == "--exact") options.exact = true;
            else if (arg == "--chat") interactive = true;
            else throw std::invalid_argument("unknown option: " + arg);
        }
        if (interactive) {
            if (path.empty() || !prompt.empty() || !ids_path.empty()) throw std::invalid_argument("--chat takes a model and no prompt");
            std::signal(SIGINT, interrupt);
            tqllm::Tokenizer tokenizer; tokenizer.load(path);
            tqllm::android_backend::Model model(path, options);
            return chat(model, tokenizer, maximum);
        }
        if (path.empty() || (prompt.empty() && ids_path.empty()) || (!prompt.empty() && !ids_path.empty()))
            throw std::invalid_argument("provide a model and exactly one of --prompt or --token-ids-file");
        if (ids_path.empty() && !raw) {
            prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n";
            if (!thinking) prompt += "\n</think>\n\n";
        }
        std::vector<uint32_t> tokens;
        tqllm::Tokenizer tokenizer;
        if (ids_path.empty()) { tokenizer.load(path); tokens = tokenizer.encode(prompt); }
        else {
            std::ifstream input(ids_path);
            if (!input) throw std::runtime_error("cannot open input token IDs");
            std::string word;
            while (input >> word) {
                if (tokens.size() >= size_t(options.context)) throw std::invalid_argument("input exceeds context");
                tokens.push_back(uint32_t(number(word, 0, 299999)));
            }
            if (maximum > 0) tokenizer.load(path);
        }
        if (tokens.empty() || tokens.size() + size_t(maximum) > size_t(options.context))
            throw std::invalid_argument("prompt plus output budget exceeds context, or empty prompt");
        auto alias = [](const std::string & a, const std::string & b) {
            if (a.empty() || b.empty()) return false;
            return std::filesystem::weakly_canonical(a) == std::filesystem::weakly_canonical(b) ||
                (std::filesystem::exists(a) && std::filesystem::exists(b) && std::filesystem::equivalent(a, b));
        };
        const std::vector<std::string> outputs{output_path, logits_path, metrics_path};
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (alias(outputs[i], path) || alias(outputs[i], ids_path))
                throw std::invalid_argument("output would overwrite an input file");
            for (size_t j = 0; j < i; ++j) if (alias(outputs[i], outputs[j]))
                throw std::invalid_argument("output files must have distinct paths");
        }
        std::ofstream ids, dump, metrics;
        if (!output_path.empty()) { ids.open(output_path); if (!ids) throw std::runtime_error("cannot write output IDs"); }
        if (!logits_path.empty()) { dump.open(logits_path, std::ios::binary); if (!dump) throw std::runtime_error("cannot write logits"); }
        if (!metrics_path.empty()) { metrics.open(metrics_path); if (!metrics) throw std::runtime_error("cannot write metrics"); }
        std::signal(SIGINT, interrupt);
        const auto load_start = Clock::now();
        tqllm::android_backend::Model model(path, options);
        for (auto token : tokens) if (token >= uint32_t(model.vocabulary())) throw std::invalid_argument("input token outside vocabulary");
        const auto start = Clock::now();
        const std::vector<float> * logits = nullptr;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (interrupted) return 130;
            logits = &model.forward(tokens[i], i + 1 == tokens.size());
        }
        const auto prefilled = Clock::now();
        if (dump.is_open()) { dump.write(reinterpret_cast<const char *>(logits->data()), std::streamsize(logits->size() * sizeof(float))); if (!dump) throw std::runtime_error("logits write failed"); }
        int generated = 0, decode_steps = 0;
        double decode_ms = 0., ttft = 0.;
        const char * finish = "length";
        for (int i = 0; i < maximum; ++i) {
            if (interrupted) { finish = "cancelled"; break; }
            auto token = uint32_t(std::max_element(logits->begin(), logits->end()) - logits->begin());
            if (token == model.eos()) { finish = "eos"; break; }
            if (generated == 0) ttft = ms(start, Clock::now());
            if (ids.is_open()) ids << token << '\n';
            std::cout << tokenizer.decode({token}) << std::flush;
            ++generated;
            if (i + 1 < maximum) {
                const auto before = Clock::now();
                logits = &model.forward(token);
                decode_ms += ms(before, Clock::now()); ++decode_steps;
            }
        }
        const auto end = Clock::now();
        std::cout << '\n';
        const double prefill_ms = ms(start, prefilled);
        const std::string report = "{\"backend\":\"native-cpu\",\"prompt_tokens\":" + std::to_string(tokens.size()) +
            ",\"generated_tokens\":" + std::to_string(generated) + ",\"decode_steps\":" + std::to_string(decode_steps) +
            ",\"load_ms\":" + std::to_string(ms(load_start, start)) + ",\"prefill_ms\":" + std::to_string(prefill_ms) +
            ",\"prefill_tps\":" + std::to_string(prefill_ms > 0. ? 1000. * tokens.size() / prefill_ms : 0.) +
            ",\"decode_ms\":" + std::to_string(decode_ms) +
            ",\"decode_tps\":" + std::to_string(decode_ms > 0. ? 1000. * decode_steps / decode_ms : 0.) +
            ",\"ttft_ms\":" + std::to_string(ttft) + ",\"wall_ms\":" + std::to_string(ms(start, end)) +
            ",\"finish_reason\":\"" + finish + "\"}";
        std::cerr << report << '\n';
        if (metrics.is_open()) { metrics << report << '\n'; if (!metrics) throw std::runtime_error("metrics write failed"); }
        if (ids.is_open()) { ids.flush(); if (!ids) throw std::runtime_error("token ID write failed"); }
        return interrupted ? 130 : 0;
    } catch (const std::exception & e) { std::cerr << "tqandroid-run: " << e.what() << '\n'; return 1; }
}
