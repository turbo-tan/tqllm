#include "tqllm/tokenizer.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char ** argv) {
    const bool file = argc == 5 && std::string(argv[3]) == "--text-file";
    if (argc != 5 || std::string(argv[1]) != "--tokenizer" || (!file && std::string(argv[3]) != "--text")) {
        std::cerr << "Usage: tqtokenize --tokenizer FILE (--text TEXT | --text-file FILE)\n";
        return 2;
    }
    try {
        tqllm::Tokenizer tokenizer;
        tokenizer.load(argv[2]);
        std::string text = argv[4];
        if (file) {
            std::ifstream in(argv[4], std::ios::binary);
            if (!in) throw std::runtime_error(std::string("cannot read ") + argv[4]);
            std::ostringstream ss;
            ss << in.rdbuf();
            text = ss.str();
        }
        const auto ids = tokenizer.encode(text);
        for (size_t i = 0; i < ids.size(); ++i) std::cout << (i ? " " : "") << ids[i];
        std::cout << '\n';
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
