#pragma once
// Weight-format contract: a format's GPU decode (dequant_h) equals the CPU reference (dequantize_host_rows, in the
// format's own basis: TQ3 rotated, F32 stored as BF16), and its matvec agrees with a double dot over the reference
// weights. Two precision classes, by how the GPU stores the format: exact (within 1 FP16 ulp, or FP16 precision of the
// 32-value block's largest value) and repacked (FP16 scale + min per block: Q4_K, Q5_K and the IQ2 / IQ1 / IQ3_XXS grids
// on the u4h kernel; the 8-bit codebook of TQ3_4S): within 2^-8 of the 256-value super-block's largest value.
// Pass 0 keeps every byte below 0x3C (scale fields finite and small), pass 1 uses full-range bytes (blocks whose values
// leave the FP16 range are skipped; the matvec is checked on pass 0 only). Run by tests/format_contract.cu for every
// registered format and by the plugin loader for every plugin format.
#include "weight_format.hpp"
#include <random>
#include <string>

namespace tqllm {

struct ContractReport {
    bool ok = false;
    bool gap = false;            // failed, but a known gap (contract_known_gap)
    std::string error;           // no CPU reference, or a CUDA error
    size_t compared = 0, bad = 0;
    int worst_ulp = 0;
    float worst_rel = 0.f;       // max |diff| / group max
    double matvec_rel = 0.0;     // rel-rms of the matvec against the double dot
    int rows_used = 0;
};

// Known gaps (the list may only shrink): the IQ2 / IQ1 / IQ3_XXS grids on the u4h kernel differ from the CPU reference by
// 2-40 % of the group's largest value on a few hundred of 65,536 random values (10-31): one side decodes some encodings wrong.
bool contract_known_gap(uint32_t type);
bool contract_repacked(uint32_t type);

// One pass over a 64 x 1024 matrix of random blocks of `type`: loaded by `loader`, run by `runtime` when not null (a
// patch under test) else by the format the loader returns.
ContractReport contract_pass(uint32_t type, const WeightFormat & loader, const WeightFormat * runtime, int pass, std::mt19937 & rng);

// Byte equality of two run-time formats on the same loaded matrices (dequant_h, and matvec when `matvec`): the
// exactness check of a patch that claims to be exact. Returns the number of differing output values (0: identical).
size_t contract_identical(uint32_t type, const WeightFormat & loader, const WeightFormat & a, const WeightFormat & b, bool matvec,
                          std::mt19937 & rng);

}  // namespace tqllm
