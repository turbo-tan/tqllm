// Gate for the MoE routing primitive (src/moe.cpp). Host-only: no CUDA, no GPU, no model.
//
// The reference implementation below is deliberately written as a DIFFERENT algorithm from the
// one under test - repeated max-scan selection instead of sort-then-slice - so agreement is
// evidence, not a tautology. It is transcribed from the description of llama.cpp's
// build_moe_ffn, not from src/moe.cpp.
//
// The tests target the specific ways this can fail silently:
//   1. selection bias leaking into the weights (the classic MoE bug),
//   2. ties resolving non-deterministically,
//   3. every prob underflowing to zero -> 0/0 NaN poisoning the residual stream,
//   4. weight_scale of 0 or 1 accidentally multiplying,
//   5. the SwiGLU clamp branch (GLM clamps gate, generic clamps silu(gate)).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

#include "moe.hpp"

using namespace tqllm;

static int failures = 0;
static void check(bool ok, const char * what) {
    printf("   %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

// ---------------------------------------------------------------- reference (independent) ----
static float ref_sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

// Selects by repeated max-scan with an "already taken" mask - a different algorithm from
// sort+slice. Ties resolve to the lowest index, matching the stable order under test.
static MoeSelection ref_route(const MoeConfig & cfg, const float * logits, const float * bias) {
    const int n = cfg.n_expert;
    std::vector<float> probs(n);
    switch (cfg.gating) {
        case MoeGating::Sigmoid:
            for (int i = 0; i < n; ++i) probs[i] = ref_sigmoid(logits[i]);
            break;
        case MoeGating::Softmax: {
            float mx = logits[0], sum = 0.f;
            for (int i = 1; i < n; ++i) if (logits[i] > mx) mx = logits[i];
            for (int i = 0; i < n; ++i) { probs[i] = expf(logits[i] - mx); sum += probs[i]; }
            for (int i = 0; i < n; ++i) probs[i] /= sum;
            break;
        }
        case MoeGating::SoftmaxWeight:
            for (int i = 0; i < n; ++i) probs[i] = logits[i];
            break;
        case MoeGating::SqrtSoftplus:
            for (int i = 0; i < n; ++i) probs[i] = sqrtf(log1pf(expf(logits[i])));
            break;
        default: break;
    }

    std::vector<char> taken(n, 0);
    MoeSelection s;
    for (int k = 0; k < cfg.n_expert_used; ++k) {
        int best = -1;
        float bv = -INFINITY;
        for (int i = 0; i < n; ++i) {
            if (taken[i]) continue;
            const float v = probs[i] + (bias ? bias[i] : 0.f);
            if (v > bv) { bv = v; best = i; }   // strict > keeps the lowest index on a tie
        }
        taken[best] = 1;
        s.ids.push_back(best);
        s.weights.push_back(probs[best]);       // UNBIASED weight
    }

    if (cfg.gating == MoeGating::SoftmaxWeight) {
        float mx = s.weights[0], sum = 0.f;
        for (float w : s.weights) if (w > mx) mx = w;
        for (float & w : s.weights) { w = expf(w - mx); sum += w; }
        for (float & w : s.weights) w /= sum;
    }
    if (cfg.norm_weights) {
        float sum = 0.f;
        for (float w : s.weights) sum += w;
        if (sum < MoeConfig::WEIGHT_SUM_EPS) sum = MoeConfig::WEIGHT_SUM_EPS;
        for (float & w : s.weights) w /= sum;
    }
    if (cfg.weight_scale != 0.f && cfg.weight_scale != 1.f)
        for (float & w : s.weights) w *= cfg.weight_scale;
    return s;
}

static bool same_sel(const MoeSelection & a, const MoeSelection & b, float tol = 1e-6f) {
    if (a.ids.size() != b.ids.size()) return false;
    for (size_t i = 0; i < a.ids.size(); ++i) {
        if (a.ids[i] != b.ids[i]) return false;
        if (std::fabs(a.weights[i] - b.weights[i]) > tol) return false;
    }
    return true;
}

// --------------------------------------------------------------------------- activations ----
static void test_activations() {
    printf("-- activations --\n");
    check(std::fabs(moe_sigmoid(0.f) - 0.5f) < 1e-7f, "sigmoid(0) == 0.5");
    check(moe_sigmoid(-1e30f) == 0.f, "sigmoid(-huge) == 0 (no overflow)");
    check(std::fabs(moe_sigmoid(1e30f) - 1.f) < 1e-7f, "sigmoid(+huge) == 1");
    check(std::fabs(moe_softplus(0.f) - std::log(2.f)) < 1e-6f, "softplus(0) == ln2");
    check(std::fabs(moe_softplus(100.f) - 100.f) < 1e-4f, "softplus(100) == 100 (stable)");
    check(std::isfinite(moe_softplus(-100.f)), "softplus(-100) finite");
}

// ------------------------------------------------------------------------------ routing ----
static void test_random_routing() {
    printf("-- random routing vs independent reference (sigmoid / softmax / softmaxweight / sqrtsoftplus) --\n");
    uint64_t s = 0x9E3779B97F4A7C15ull;
    auto rnd = [&]() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return double(s >> 11) / double(1ull << 53);
    };

    const MoeGating gatings[] = {MoeGating::Sigmoid, MoeGating::Softmax,
                                 MoeGating::SoftmaxWeight, MoeGating::SqrtSoftplus};
    int checked = 0, mismatched = 0;
    for (auto g : gatings) {
        for (const bool norm : {false, true}) {
            for (const bool biased : {false, true}) {
                for (int trial = 0; trial < 200; ++trial) {
                    const int n_expert = 2 + int(rnd() * 60);        // 2..61, incl. GLM's 288 shape class
                    const int n_used = 1 + int(rnd() * n_expert);
                    MoeConfig cfg;
                    cfg.n_expert = n_expert;
                    cfg.n_expert_used = n_used;
                    cfg.gating = g;
                    cfg.norm_weights = norm;
                    cfg.weight_scale = float(rnd() * 4.0);
                    if (cfg.weight_scale < 0.01f) cfg.weight_scale = 2.5f;  // GLM's value

                    std::vector<float> logits(n_expert), bias(n_expert);
                    for (int i = 0; i < n_expert; ++i) logits[i] = float(rnd() * 16.0 - 8.0);
                    for (int i = 0; i < n_expert; ++i) bias[i] = float(rnd() * 2.0 - 1.0);

                    const MoeSelection a = moe_route(cfg, logits.data(), biased ? bias.data() : nullptr);
                    const MoeSelection b = ref_route(cfg, logits.data(), biased ? bias.data() : nullptr);
                    ++checked;
                    if (!same_sel(a, b)) ++mismatched;
                }
            }
        }
    }
    printf("   %d routing cases checked, %d mismatched\n", checked, mismatched);
    check(mismatched == 0, "routing matches the independent reference on every case");
}

// The bug this whole file exists to catch: the selection bias must change WHICH experts are
// picked, but the returned weights must come from the unbiased probs.
static void test_bias_does_not_leak_into_weights() {
    printf("-- selection bias changes selection, not weights --\n");
    MoeConfig cfg;
    cfg.n_expert = 4;
    cfg.n_expert_used = 2;
    cfg.gating = MoeGating::Sigmoid;
    cfg.norm_weights = false;
    cfg.weight_scale = 1.f;   // no scaling, so weights are raw sigmoid values

    const float logits[4] = {2.f, 1.f, 0.f, -1.f};
    // Without bias, expert 0 (sigmoid 0.881) then 1 (0.731) win.
    const MoeSelection plain = moe_route(cfg, logits, nullptr);
    check(plain.ids[0] == 0 && plain.ids[1] == 1, "no bias: experts 0,1 selected");

    // A large positive bias on expert 3 must reorder the SELECTION...
    const float bias[4] = {0.f, 0.f, 0.f, 100.f};
    const MoeSelection biased = moe_route(cfg, logits, bias);
    check(biased.ids[0] == 3, "bias: expert 3 promoted to first");

    // ...but expert 3's WEIGHT must still be its unbiased sigmoid(-1) = 0.2689, NOT ~1.0.
    const float expect_w3 = 1.f / (1.f + std::exp(1.f));
    check(std::fabs(biased.weights[0] - expect_w3) < 1e-6f,
          "biased selection still carries the UNBIASED weight (the classic MoE bug)");
    if (std::fabs(biased.weights[0] - expect_w3) >= 1e-6f)
        printf("        got %.6f, expected %.6f\n", biased.weights[0], expect_w3);
    check(biased.weights[0] < 0.5f, "weight is not the sigmoid of (logit+bias)");
}

static void test_ties_are_deterministic() {
    printf("-- ties --\n");
    MoeConfig cfg;
    cfg.n_expert = 8;
    cfg.n_expert_used = 3;
    cfg.gating = MoeGating::Sigmoid;
    std::vector<float> logits(8, 0.5f);   // every expert identical
    const MoeSelection a = moe_route(cfg, logits.data(), nullptr);
    const MoeSelection b = moe_route(cfg, logits.data(), nullptr);
    check(same_sel(a, b), "all-equal logits: selection is reproducible run to run");
    check(a.ids[0] == 0 && a.ids[1] == 1 && a.ids[2] == 2, "all-equal logits: lowest indices win");
}

static void test_underflow_cannot_produce_nan() {
    printf("-- underflow guard --\n");
    MoeConfig cfg;
    cfg.n_expert = 4;
    cfg.n_expert_used = 4;
    cfg.gating = MoeGating::Sigmoid;
    cfg.norm_weights = true;
    const float logits[4] = {-200.f, -300.f, -400.f, -500.f};  // every sigmoid -> exactly 0
    const MoeSelection s = moe_route(cfg, logits, nullptr);
    bool any_nan = false;
    for (float w : s.weights) if (!std::isfinite(w)) any_nan = true;
    check(!any_nan, "all-zero probs with norm_weights: no NaN/Inf in the weights");
    printf("        weights = %.6g %.6g %.6g %.6g\n", s.weights[0], s.weights[1],
           s.weights[2], s.weights[3]);
}

static void test_weight_scale_identity_values() {
    printf("-- weight_scale identity values --\n");
    MoeConfig cfg;
    cfg.n_expert = 4;
    cfg.n_expert_used = 2;
    cfg.gating = MoeGating::Sigmoid;
    cfg.norm_weights = true;
    const float logits[4] = {3.f, 1.f, -1.f, -3.f};

    cfg.weight_scale = 0.f;
    const MoeSelection z = moe_route(cfg, logits, nullptr);
    cfg.weight_scale = 1.f;
    const MoeSelection o = moe_route(cfg, logits, nullptr);
    cfg.weight_scale = 2.5f;
    const MoeSelection s = moe_route(cfg, logits, nullptr);

    check(same_sel(z, o), "weight_scale 0 and 1 both mean no scaling");
    check(std::fabs(s.weights[0] - 2.5f * o.weights[0]) < 1e-6f, "weight_scale 2.5 multiplies");
    float sum = 0.f;
    for (float w : o.weights) sum += w;
    check(std::fabs(sum - 1.f) < 1e-6f, "norm_weights: selected weights sum to 1");
    float sums = 0.f;
    for (float w : s.weights) sums += w;
    check(std::fabs(sums - 2.5f) < 1e-6f, "GLM config: normalised then scaled by 2.5");
}

static void test_glm_shape() {
    printf("-- GLM-5.3-Flash shape (288 experts, top 8, sigmoid, norm, scale 2.5) --\n");
    MoeConfig cfg;
    cfg.n_expert = 288;
    cfg.n_expert_used = 8;
    cfg.gating = MoeGating::Sigmoid;
    cfg.norm_weights = true;
    cfg.weight_scale = 2.5f;
    std::vector<float> logits(288), bias(288);
    for (int i = 0; i < 288; ++i) logits[i] = float(std::sin(i * 0.7) * 3.0);
    for (int i = 0; i < 288; ++i) bias[i] = 0.001f * float(i % 7);

    const MoeSelection a = moe_route(cfg, logits.data(), bias.data());
    const MoeSelection b = ref_route(cfg, logits.data(), bias.data());
    check(same_sel(a, b), "288x8 routing matches the independent reference");
    check(a.ids.size() == 8, "exactly 8 experts selected");

    // The correct invariant: SELECTION follows the BIASED probs in descending order, while the
    // WEIGHTS are the UNBIASED probs at those ids. The two orders legitimately differ - an
    // expert with a big bias but a small sigmoid can be picked early yet carry a small weight.
    // Asserting that weights descend would be wrong, and would hide the bias/weight split.
    bool sel_desc = true;
    for (int k = 1; k < 8; ++k) {
        const float prev = moe_sigmoid(logits[a.ids[k - 1]]) + bias[a.ids[k - 1]];
        const float cur = moe_sigmoid(logits[a.ids[k]]) + bias[a.ids[k]];
        if (prev < cur) sel_desc = false;
    }
    check(sel_desc, "selection order is descending by BIASED probability");

    bool weights_unbiased = true;
    for (int k = 0; k < 8; ++k) {
        // weights are normalised (sum 1) then scaled by 2.5, so undo both before comparing
        const float raw = moe_sigmoid(logits[a.ids[k]]);
        float raw_sum = 0.f;
        for (int id : a.ids) raw_sum += moe_sigmoid(logits[id]);
        const float expect = raw / std::max(raw_sum, MoeConfig::WEIGHT_SUM_EPS) * 2.5f;
        if (std::fabs(a.weights[k] - expect) > 1e-5f) weights_unbiased = false;
    }
    check(weights_unbiased, "weights are the UNBIASED sigmoid probs, normalised then scaled");

    // Confirm the two orders really can differ here, so the checks above have discriminating
    // power rather than passing because bias happened to be order-preserving.
    bool orders_differ = false;
    std::vector<int> by_weight(a.ids);
    std::stable_sort(by_weight.begin(), by_weight.end(), [&](int p, int q) {
        return moe_sigmoid(logits[p]) > moe_sigmoid(logits[q]);
    });
    for (int k = 0; k < 8; ++k) if (by_weight[k] != a.ids[k]) orders_differ = true;
    check(orders_differ, "biased selection order differs from unbiased weight order (test is live)");
}

// ----------------------------------------------------------------------------- expert FFN ----
static void test_expert_ffn() {
    printf("-- expert FFN (down * swiglu(gate, up)) --\n");
    const int hidden = 8, ff = 6;
    std::vector<float> x(hidden), wg(ff * hidden), wu(ff * hidden), wd(hidden * ff), out(hidden);
    uint64_t s = 12345;
    auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return double(s >> 11) / double(1ull << 53) - 0.5; };
    for (auto & v : x) v = float(rnd());
    for (auto & v : wg) v = float(rnd());
    for (auto & v : wu) v = float(rnd());
    for (auto & v : wd) v = float(rnd());

    // Independent naive reference, written straight from out = W_down * (silu(W_gate x) * (W_up x)).
    std::vector<float> ref(hidden, 0.f);
    for (int o = 0; o < ff; ++o) {
        float sg = 0.f, su = 0.f;
        for (int i = 0; i < hidden; ++i) {
            sg += wg[size_t(o) * hidden + i] * x[i];
            su += wu[size_t(o) * hidden + i] * x[i];
        }
        const float silu = sg / (1.f + expf(-sg));
        const float a = silu * su;
        for (int i = 0; i < hidden; ++i) ref[i] += wd[size_t(i) * ff + o] * a;
    }
    moe_expert_ffn(x.data(), hidden, wg.data(), wu.data(), wd.data(), ff, 0.f, false, out.data());
    double worst = 0;
    for (int i = 0; i < hidden; ++i) worst = std::max(worst, double(std::fabs(out[i] - ref[i])));
    printf("        max abs diff vs naive reference: %.3e\n", worst);
    check(worst < 1e-5, "unclamped SwiGLU FFN matches the naive reference");

    // The clamp must actually change the result (otherwise the branch is dead code).
    std::vector<float> xc = x;
    for (auto & v : xc) v *= 40.f;   // drive gate/up far past the clamp limit
    std::vector<float> o1(hidden), o2(hidden), o3(hidden);
    moe_expert_ffn(xc.data(), hidden, wg.data(), wu.data(), wd.data(), ff, 0.f, false, o1.data());
    moe_expert_ffn(xc.data(), hidden, wg.data(), wu.data(), wd.data(), ff, 7.f, true, o2.data());
    moe_expert_ffn(xc.data(), hidden, wg.data(), wu.data(), wd.data(), ff, 7.f, false, o3.data());
    bool clamped_differs = false, branches_differ = false;
    for (int i = 0; i < hidden; ++i) {
        if (std::fabs(o1[i] - o2[i]) > 1e-4f) clamped_differs = true;
        if (std::fabs(o2[i] - o3[i]) > 1e-4f) branches_differ = true;
    }
    check(clamped_differs, "clamp limit 7 changes the output (branch is live)");
    check(branches_differ, "GLM gate-clamp and generic silu-clamp branches differ");

    // Clamping must bound the activation: with a huge limit-free input the unclamped result can
    // be enormous, the clamped one stays bounded.
    bool bounded = true;
    for (int i = 0; i < hidden; ++i) if (!std::isfinite(o2[i])) bounded = false;
    check(bounded, "clamped output stays finite");
}

static void test_combine() {
    printf("-- combine (weighted sum) --\n");
    MoeSelection sel;
    sel.ids = {2, 5};
    sel.weights = {0.75f, 0.25f};
    const int hidden = 4;
    std::vector<float> outs(2 * hidden), out(hidden, 0.f);
    for (int i = 0; i < hidden; ++i) { outs[i] = float(i + 1); outs[hidden + i] = float(i + 10); }
    moe_combine(sel, outs.data(), hidden, out.data());
    bool ok = true;
    for (int i = 0; i < hidden; ++i) {
        const float expect = 0.75f * float(i + 1) + 0.25f * float(i + 10);
        if (std::fabs(out[i] - expect) > 1e-6f) ok = false;
    }
    check(ok, "combine = sum of weight * expert output");
}

static void test_end_to_end_linearity() {
    printf("-- end to end: route + ffn + combine --\n");
    // With norm_weights on and a single expert, the combined output must equal that expert's
    // output scaled by weight_scale (weight normalises to 1 then multiplies by the scale).
    const int hidden = 6, ff = 4, n_expert = 5;
    MoeConfig cfg;
    cfg.n_expert = n_expert;
    cfg.n_expert_used = 1;
    cfg.gating = MoeGating::Sigmoid;
    cfg.norm_weights = true;
    cfg.weight_scale = 2.5f;
    std::vector<float> logits(n_expert, 0.f);
    logits[3] = 5.f;   // expert 3 wins outright

    std::vector<float> x(hidden), wg(ff * hidden), wu(ff * hidden), wd(hidden * ff);
    uint64_t s = 999;
    auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return double(s >> 11) / double(1ull << 53) - 0.5; };
    for (auto & v : x) v = float(rnd());
    for (auto & v : wg) v = float(rnd());
    for (auto & v : wu) v = float(rnd());
    for (auto & v : wd) v = float(rnd());

    const MoeSelection sel = moe_route(cfg, logits.data(), nullptr);
    check(sel.ids[0] == 3, "single expert: the clear winner is selected");
    check(std::fabs(sel.weights[0] - 2.5f) < 1e-6f, "single expert: normalised to 1 then scaled to 2.5");

    std::vector<float> direct(hidden), eout(hidden), combined(hidden, 0.f);
    moe_expert_ffn(x.data(), hidden, wg.data(), wu.data(), wd.data(), ff, 0.f, false, direct.data());
    moe_expert_ffn(x.data(), hidden, wg.data(), wu.data(), wd.data(), ff, 0.f, false, eout.data());
    moe_combine(sel, eout.data(), hidden, combined.data());
    bool ok = true;
    for (int i = 0; i < hidden; ++i)
        if (std::fabs(combined[i] - 2.5f * direct[i]) > 1e-4f) ok = false;
    check(ok, "combined output == 2.5 x the single expert's FFN output");
}

int main() {
    test_activations();
    test_random_routing();
    test_bias_does_not_leak_into_weights();
    test_ties_are_deterministic();
    test_underflow_cannot_produce_nan();
    test_weight_scale_identity_values();
    test_glm_shape();
    test_expert_ffn();
    test_combine();
    test_end_to_end_linearity();
    if (failures) {
        printf("\nFAIL: %d check(s)\n", failures);
        return 1;
    }
    printf("\nPASS: MoE routing primitive verified against an independent reference\n");
    return 0;
}
