# Renders requests through the froggeric v22 chat template (jinja2) and through tqserve::parse_request
# (tests/template/render.cpp, built with: g++ -std=c++20 -I. -Iinclude tests/template/render.cpp -o render) and diffs them.
# usage: python compare.py DIR_WITH_render /path/to/chat_template.jinja   (needs jinja2)
import json, subprocess, sys, difflib, jinja2
S = sys.argv[1]; T = open(sys.argv[2]).read()
tool = {"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
cases = {
 "A_think_default": {"messages": [{"role": "user", "content": "What is 2+2?"}]},
 "B_think_off": {"messages": [{"role": "user", "content": "What is 2+2?"}], "chat_template_kwargs": {"enable_thinking": False}},
 "C_system_low": {"messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}], "reasoning_effort": "low"},
 "D_tools": {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [tool]},
 "E_tool_multiturn": {"messages": [{"role": "user", "content": "Weather in Paris?"},
     {"role": "assistant", "content": "", "reasoning_content": "Need the weather tool.", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
     {"role": "tool", "tool_call_id": "c1", "content": "18C sunny"}], "tools": [tool]},
 "E2_tool_dict_args_2calls": {"messages": [{"role": "user", "content": "Weather in Paris and Rome?"},
     {"role": "assistant", "content": "", "reasoning_content": "Two cities.", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
        {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]},
     {"role": "tool", "tool_call_id": "c1", "content": "Error: city service unavailable"},
     {"role": "tool", "tool_call_id": "c2", "content": "Traceback (most recent call last): boom"}], "tools": [tool]},
 "G_preserve_off": {"messages": [{"role": "user", "content": "Pick a number."}, {"role": "assistant", "content": "7", "reasoning_content": "Say 7."}, {"role": "user", "content": "Double it."}],
     "chat_template_kwargs": {"preserve_thinking": False}},
 "H_inline_think": {"messages": [{"role": "user", "content": "Pick a number."}, {"role": "assistant", "content": "<think>\nSay 7.\n</think>\n\n7"}, {"role": "user", "content": "Double it."}]},
 "F_chat_multiturn": {"messages": [{"role": "user", "content": "Pick a number."}, {"role": "assistant", "content": "7", "reasoning_content": "Any number works; say 7."}, {"role": "user", "content": "Double it."}]},
}
import json as _j
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
def rex(m): raise Exception(m)
env.globals['raise_exception'] = rex
env.policies['json.dumps_kwargs'] = {'sort_keys': False, 'ensure_ascii': False}
tpl = env.from_string(T)
for name, body in cases.items():
    kw = dict(body.get("chat_template_kwargs", {}))
    if "reasoning_effort" in body: kw["reasoning_effort"] = body["reasoning_effort"]
    want = tpl.render(messages=body["messages"], tools=body.get("tools"), add_generation_prompt=True, **kw)
    p = f"{S}/{name}.json"; json.dump(body, open(p, "w"))
    got = subprocess.run([f"{S}/render", p], capture_output=True, text=True).stdout
    same = want == got
    print(f"== {name}: {'IDENTICAL' if same else 'DIFFERENT'}")
    if not same:
        for l in difflib.unified_diff(want.splitlines(), got.splitlines(), "v22 template", "tqserve", lineterm="", n=1):
            print("   " + l[:230])
