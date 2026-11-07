// extract_tool_calls(): parallel calls in one turn (Qwen early-access model / Qwen4 emit several <function=...> blocks),
// JSON-typed and string parameters, text before the first call kept, no calls -> text unchanged.
#include "session.hpp"

#include <cstdio>
#include <string>

int main() {
    int fails = 0;
    auto expect = [&](bool ok, const char * what) { if (!ok) { std::printf("FAIL %s\n", what); ++fails; } };
    {
        tqserve::DecodeResult r;
        r.text = "Checking both cities.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>\n"
                 "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        tqserve::extract_tool_calls(r);
        expect(r.tool_calls.size() == 2, "two parallel calls");
        expect(r.tool_calls.size() == 2 && r.tool_calls[0].name == "get_weather" && r.tool_calls[0].arguments == "{\"city\": \"Tokyo\"}",
               "first call: string parameter");
        expect(r.tool_calls.size() == 2 && r.tool_calls[1].arguments == "{\"city\": \"Paris\", \"days\": 3}", "second call: JSON number kept as JSON");
        expect(r.text == "Checking both cities." && r.finish_reason == "tool_calls", "text before the first call kept, finish tool_calls");
    }
    {
        tqserve::DecodeResult r;
        r.text = "<function=a>\n<parameter=x>\n[1, 2]\n</parameter>\n</function>\n<function=b>\n<parameter=y>\n{\"k\": true}\n</parameter>\n</function>\n"
                 "<function=c>\n</function>";
        tqserve::extract_tool_calls(r);
        expect(r.tool_calls.size() == 3, "three calls without <tool_call> markers (the detokenizer drops them)");
        expect(r.tool_calls.size() == 3 && r.tool_calls[0].arguments == "{\"x\": [1, 2]}" && r.tool_calls[1].arguments == "{\"y\": {\"k\": true}}" &&
                   r.tool_calls[2].name == "c" && r.tool_calls[2].arguments == "{}", "arrays, objects, no parameters");
    }
    {
        tqserve::DecodeResult r;
        r.text = "No tools needed: 42.";
        tqserve::extract_tool_calls(r);
        expect(r.tool_calls.empty() && r.text == "No tools needed: 42." && r.finish_reason == "length", "no calls: unchanged");
    }
    std::printf("tool calls: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
