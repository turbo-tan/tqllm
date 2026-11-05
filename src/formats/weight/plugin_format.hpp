#pragma once
// Weight-format plugins (include/tqllm/plugin.h ABI 2, docs/guides/plugins.md): a shared library adds a new GGUF type
// or patches a run-time format's kernels. Every plugin format passes the format contract at load, and a patch that
// claims exact is checked byte for byte against the built-in; a plugin that fails either is refused (the load throws).
// Load plugins before any model opens: matrices loaded earlier keep the formats they were loaded with.
#include <string>
#include <vector>

namespace tqllm {

struct PluginInfo {
    std::string path;
    std::vector<std::string> formats;  // new types: "NAME (type T)"
    std::vector<std::string> patches;  // applied patches: "NAME -> type T"
    std::vector<std::string> skipped;  // patches for another GPU: "NAME (sm 80-90, this GPU sm 86)"
    bool inexact = false;              // an applied patch is not byte-identical to the built-in (--exact on warns)
};

PluginInfo load_plugin(const std::string & path);
const std::vector<PluginInfo> & loaded_plugins();

}  // namespace tqllm
