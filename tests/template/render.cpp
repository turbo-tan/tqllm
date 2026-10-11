#include "tools/serve/session.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
int main(int argc, char ** argv) {
    std::ifstream f(argv[1]); std::stringstream ss; ss << f.rdbuf();
    const auto r = tqserve::parse_request(ss.str(), 32768, "", true, true, argc > 2 ? argv[2] : "xhigh");
    if (!r.ok) { std::cerr << "error: " << r.error << "\n"; return 1; }
    std::cout << r.prompt_text;
}
