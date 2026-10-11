#include "tqllm/runtime.hpp"
#include <iostream>
int main(int argc, char ** argv) {
    try {
        tqllm::Runtime empty;
        try { (void) empty.resolve("missing", 4); return 1; } catch (const std::runtime_error &) {}
        if (argc == 2) {
            auto plan = [&] {
                tqllm::Runtime r;
                r.load(argv[1]);
                try { r.load(argv[1]); return tqllm::Plan{}; } catch (const std::runtime_error &) {}
                try { (void) r.resolve("example.host.add.f32.v1", 4097); return tqllm::Plan{}; } catch (const std::runtime_error &) {}
                return r.resolve("example.host.add.f32.v1", 4);
            }();
            if (!plan.invoke) return 1;
            float a[]{1,-2,3,0}, b[]{2,2,-3,4}, out[4]{};
            plan.run(a,b,out);
            for (int i=0;i<4;++i) if (out[i] != a[i]+b[i]) return 1;
        }
        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
