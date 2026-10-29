#include "android_model.hpp"
#include <iostream>
#include <stdexcept>
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    try {
        tqllm::android_backend::Model model(argv[1], {8, 2});
        model.forward(1, false);
        auto first = model.forward(2);
        model.forward(3);
        model.reset();
        if (model.position() != 0) throw std::runtime_error("reset position");
        model.forward(1, false);
        if (model.forward(2) != first) throw std::runtime_error("reset changed logits");
        bool rejected = false;
        try { model.forward(uint32_t(model.vocabulary())); } catch (const std::out_of_range &) { rejected = true; }
        if (!rejected || model.position() != 2) throw std::runtime_error("token bounds changed state");
        for (int i = model.position(); i < model.context(); ++i) model.forward(1, false);
        rejected = false;
        try { model.forward(1); } catch (const std::out_of_range &) { rejected = true; }
        if (!rejected) throw std::runtime_error("context overflow accepted");
        std::cout << "android_state: PASS (GDN+KV reset, token/context bounds)\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
