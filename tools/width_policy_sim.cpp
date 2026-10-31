// Offline replay of the verify-width policy on per-step traces (tqmlx-dflash TRACE lines, j1031): for each prompt a
// width-15 run gives the acceptance a at each text position (a step starting o tokens into a width-15 step's run of a
// accepts min(a - o, width)) and its step ms,
// a width-7 run gives the narrow step ms. The replay walks the text by tokens produced, so a policy's narrow steps see
// the acceptance of the position they reach. Prints decode tok/s per prompt for always-15, always-7 and the policy.
// usage: width-policy-sim traces.txt   (lines: TRACE {"model","prompt","width","a":[..],"ms":[..],"out"})
#include "width_policy.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using tqllm::mlx_backend::WidthPolicy;
namespace {
std::vector<double> numbers(const std::string& line, const std::string& key) {
    std::vector<double> v;
    const auto at = line.find("\"" + key + "\":[");
    if (at == std::string::npos) return v;
    std::stringstream in(line.substr(at + key.size() + 4, line.find(']', at) - at - key.size() - 4));
    std::string x;
    while (std::getline(in, x, ',')) if (!x.empty()) v.push_back(std::stod(x));
    return v;
}
std::string text(const std::string& line, const std::string& key) {
    const auto at = line.find("\"" + key + "\":\"");
    return line.substr(at + key.size() + 4, line.find('"', at + key.size() + 4) - at - key.size() - 4);
}
int integer(const std::string& line, const std::string& key) {
    const auto at = line.find("\"" + key + "\":");
    return std::stoi(line.substr(at + key.size() + 3));
}
struct Run {std::vector<double> a, ms; int out = 0, context = 0;};
double mean_tail(const std::vector<double>& v) {  // step ms without the first (graph build) step
    double s = 0; for (size_t i = 1; i < v.size(); ++i) s += v[i];
    return v.size() > 1 ? s / double(v.size() - 1) : 0;
}
// replays the width-15 steps in order; a 7-token step at step i accepts min(a_i, 7) of the same draft (the measured
// width-7 runs accept ~10 % more than that: slightly pessimistic for narrow). mode 15 / 7 fixed or 0 = policy
template <class Policy> double replay(const Run& wide, double ms7, double ms15, int mode, Policy& policy, int* narrow) {
    policy.begin(wide.context);
    int produced = 0; double ms = 0; *narrow = 0;
    for (size_t i = 0; i < wide.a.size(); ++i) {
        const int width = mode ? mode : policy.choose(0);
        const int a = width == WidthPolicy::kWide ? int(wide.a[i]) : std::min(int(wide.a[i]), WidthPolicy::kNarrow);
        const double t = width == WidthPolicy::kWide ? ms15 : ms7;
        policy.record(width, a, 0, i == 0 ? 3 * t : t);
        if (i > 0) {ms += t; produced += a + 1;}
        *narrow += width == WidthPolicy::kNarrow;
    }
    return ms > 0 ? produced / (ms / 1000.0) : 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {std::fprintf(stderr, "usage: width-policy-sim traces.txt\n"); return 2;}
    std::map<std::string, std::map<int, Run>> runs;  // model|prompt -> width -> run
    std::vector<std::string> order;
    std::ifstream in(argv[1]);
    for (std::string line; std::getline(in, line);) {
        if (line.rfind("TRACE ", 0) != 0) continue;
        const std::string key = text(line, "model") + "|" + text(line, "prompt");
        if (!runs.count(key)) order.push_back(key);
        Run r; r.a = numbers(line, "a"); r.ms = numbers(line, "ms"); r.out = integer(line, "out"); r.context = integer(line, "prompt_tokens");
        runs[key][integer(line, "width")] = r;
    }
    std::map<std::string, std::vector<double>> logs;  // model -> log tok/s per mode (15, 7, policy)
    std::string model;
    WidthPolicy policy;  // one process per model, as the server
    for (const auto& key : order) {
        const auto& m = runs[key];
        if (!m.count(15) || !m.count(7)) continue;
        const std::string mk = key.substr(0, key.find('|'));
        if (mk != model) {policy = WidthPolicy(); model = mk;}
        const double ms15 = mean_tail(m.at(15).ms), ms7 = mean_tail(m.at(7).ms);
        int n15, n7, np; WidthPolicy fixed;
        const double d15 = replay(m.at(15), ms7, ms15, 15, fixed, &n15), d7 = replay(m.at(15), ms7, ms15, 7, fixed, &n7);
        const double dp = replay(m.at(15), ms7, ms15, 0, policy, &np);
        auto measured = [](const Run& r) {double ms = 0; int t = 1; for (size_t i = 1; i < r.ms.size(); ++i) {ms += r.ms[i];}
                                          for (size_t i = 1; i < r.a.size(); ++i) t += int(r.a[i]) + 1; return ms > 0 ? t / (ms / 1000.0) : 0;};
        std::printf("SIM %-26s ms15 %6.1f ms7 %6.1f | w15 %6.1f w7 %6.1f policy %6.1f (narrow %3d) policy/best %.3f | measured w7/w15 %.3f sim %.3f\n",
                    key.c_str(), ms15, ms7, d15, d7, dp, np, dp / std::max(d15, d7), measured(m.at(7)) / measured(m.at(15)), d7 / d15);
        auto& l = logs[mk]; if (l.empty()) l.assign(4, 0.0);
        l[0] += std::log(d15); l[1] += std::log(d7); l[2] += std::log(dp); l[3] += 1;
    }
    for (const auto& [mk, l] : logs)
        std::printf("SIM_GEOMEAN %s w15 %.1f w7 %.1f policy %.1f (n %d)\n", mk.c_str(), std::exp(l[0] / l[3]), std::exp(l[1] / l[3]),
                    std::exp(l[2] / l[3]), int(l[3]));
    return 0;
}
