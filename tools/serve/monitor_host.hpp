#pragma once
#include "monitor.hpp"
#include <fstream>
#include <sstream>
#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace tqserve {
// NVML is optional and selected by PCI bus ID, never by an assumed device index.
class MonitorHost {
public:
    explicit MonitorHost(const std::string & pci) {
#if defined(__linux__)
        if (pci.empty()) return;
        library_ = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!library_) return;
        auto init = symbol<int(*)()>("nvmlInit_v2");
        auto handle = symbol<int(*)(const char *, void **)>("nvmlDeviceGetHandleByPciBusId_v2");
        initialized_ = init && init() == 0;
        if (initialized_ && handle && !pci.empty() && handle(pci.c_str(), &device_) != 0) device_ = nullptr;
#else
        (void)pci;
#endif
    }
    ~MonitorHost() {
#if defined(__linux__)
        if (initialized_) if (auto fn = symbol<int(*)()>("nvmlShutdown")) fn();
        if (library_) dlclose(library_);
#endif
    }
    MonitorHost(const MonitorHost &) = delete;
    MonitorHost & operator=(const MonitorHost &) = delete;
    Monitor::Hardware sample() {
        Monitor::Hardware h;
#if defined(__linux__)
        std::ifstream stat("/proc/stat");
        std::string label; uint64_t value, total = 0, idle = 0;
        std::string line;
        if (std::getline(stat, line)) {
            std::istringstream row(line); row >> label;
            for (int i = 0; i < 8 && row >> value; ++i) { total += value; if (i == 3 || i == 4) idle += value; }
            if (previous_total_ && total > previous_total_ && idle >= previous_idle_)
                h.cpu_percent = 100.0 * (1.0 - double(idle - previous_idle_) / double(total - previous_total_));
            previous_total_ = total; previous_idle_ = idle;
        }
        std::ifstream mem("/proc/meminfo");
        uint64_t available = 0, ram = 0; bool has_available = false;
        while (std::getline(mem, line)) {
            std::istringstream row(line); row >> label >> value;
            if (label == "MemTotal:") ram = value;
            if (label == "MemAvailable:") { available = value; has_available = true; }
        }
        if (ram && has_available && available <= ram) { h.ram_total_bytes = double(ram) * 1024; h.ram_used_bytes = double(ram - available) * 1024; }
        if (device_) {
            char name[96]{};
            if (auto fn = symbol<int(*)(void *, char *, unsigned)>("nvmlDeviceGetName"))
                if (fn(device_, name, sizeof(name)) == 0) h.device_name = name;
            struct Memory { unsigned long long total, free, used; } m{};
            if (auto fn = symbol<int(*)(void *, Memory *)>("nvmlDeviceGetMemoryInfo"))
                if (fn(device_, &m) == 0) { h.gpu_used_bytes = double(m.used); h.gpu_total_bytes = double(m.total); }
            struct Utilization { unsigned gpu, memory; } u{};
            if (auto fn = symbol<int(*)(void *, Utilization *)>("nvmlDeviceGetUtilizationRates"))
                if (fn(device_, &u) == 0) h.gpu_percent = u.gpu;
            unsigned v;
            if (auto fn = symbol<int(*)(void *, unsigned, unsigned *)>("nvmlDeviceGetTemperature"))
                if (fn(device_, 0, &v) == 0) h.gpu_celsius = v;
            if (auto fn = symbol<int(*)(void *, unsigned *)>("nvmlDeviceGetPowerUsage"))
                if (fn(device_, &v) == 0) h.gpu_watts = double(v) / 1000;
        }
#endif
        return h;
    }
private:
#if defined(__linux__)
    template<class T> T symbol(const char * name) { return reinterpret_cast<T>(dlsym(library_, name)); }
    void * library_ = nullptr, * device_ = nullptr;
    bool initialized_ = false;
    uint64_t previous_total_ = 0, previous_idle_ = 0;
#endif
};
} // namespace tqserve
