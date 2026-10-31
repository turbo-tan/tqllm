# Plugins

[Documentation](../README.md) / [Project home](../../README.md)

A plugin is a shared library that tqserve loads with `--plugin PATH` (repeatable). ABI 2 (`include/tqllm/plugin.h`)
covers weight formats on the CUDA backend:

- **a new format**: a GGUF tensor type the built-in formats do not know (type ids 2000-2999), with its own kernels;
- **a kernel patch**: a faster dequant or matvec for a built-in run-time format, kept only on the GPUs it declares.

Engines as plugins (a new architecture out of tree) come next; until then a new architecture is added in-tree
([Adding a model](adding-a-model.md)).

## Writing one

Export one C function and register a `tq_format_v2` per format:

```c
#include "tqllm/plugin.h"

static const tq_format_v2 p8 = {
    sizeof(tq_format_v2), TQ_ABI_VERSION_2, "P8", 2001,
    32, 34,          /* values and bytes per GGUF block */
    0, 0,            /* sm_min, sm_max (x10, e.g. 80..120); 0, 0: any GPU */
    0,               /* exact: patches only */
    p8_cpu_dequant, p8_gpu_dequant, p8_gpu_matvec};

int32_t tq_plugin_init_v2(const tq_host_v2 * host, tq_error * err) {
    return host->register_format(host->context, &p8, err);
}
```

- `cpu_dequant` is the reference: every format is checked against it.
- `gpu_dequant_f16` writes the matrix as FP16 (prefill GEMMs); `gpu_matvec` computes `y = W x` for up to 8 columns
  (decode). Device pointers and the stream are `void *` (`cudaStream_t`), so the header needs no CUDA include.
- A new format receives the tensor's GGUF blocks unchanged in device memory.
- A patch names a patchable run-time format and receives that format's device layout. Today: `TQ_PATCH_I8H` (1000),
  the run-time format of Q8_0 and the int8 heads: int8 weights `[rows][cols]` row-major, one FP16 scale per 16
  consecutive values in `aux`. A callback left null keeps the built-in kernel; everything the plugin does not replace
  (expert GEMMs, int8 prefill, ...) stays built-in.

`tests/plugins/example_plugin.cu` has a complete new format and a complete patch. Build with
`nvcc -shared -Xcompiler -fPIC -Iinclude ...`.

## What the server checks at load

| Check | Fails the load when |
|---|---|
| ABI | no `tq_plugin_init_v2`, a descriptor of another ABI version or size, a type outside 2000-2999 that is not patchable |
| Format contract (`src/formats/weight/format_contract.hpp`, the same test every built-in format passes) | GPU dequant differs from `cpu_dequant` by more than 1 FP16 ulp / 2^-10 of the 32-value block's largest value, or the matvec is off a double dot by more than 2 % rel-rms |
| Exactness (patches with `exact = 1`) | dequant or matvec output differs from the built-in in any bit, on the contract's random matrices |
| GPU range | a new format's `sm_min..sm_max` excludes this GPU (a patch for another GPU is skipped and listed instead) |

A patch that is not byte-identical can still be loaded (`exact = 0`). With `--exact on` (the default) the server
prints one warning naming it and uses it (the asked option wins), and `/health` notes it. Compare a patch by
starting the server with and without `--plugin`; there is no switch to turn a loaded plugin off.

`GET /health` lists what each plugin added:
`"plugins":[{"path":"...","formats":["P8 (type 2001)"],"patches":[],"skipped":[]}]`.
