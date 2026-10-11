# Checks the vendored C++ Jinja engine (test-jinja) against Python jinja2 on the same template and cases: the engine, not the template, is under test.
# usage: python compare_jinja.py TEMPLATE.jinja CASES.json DUMP_DIR   (DUMP_DIR written by: test-jinja TEMPLATE CASES DUMP_DIR)
import json, sys, os, difflib, jinja2
T = open(sys.argv[1]).read(); cases = json.load(open(sys.argv[2])); D = sys.argv[3]
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
def rex(m): raise Exception(m)
env.globals['raise_exception'] = rex
env.policies['json.dumps_kwargs'] = {'sort_keys': False, 'ensure_ascii': False}
tpl = env.from_string(T)
bad = 0
for name, body in cases.items():
    msgs = json.loads(json.dumps(body["messages"]))
    for m in msgs:  # arguments as objects, like llama.cpp and HF
        for tc in m.get("tool_calls", []) or []:
            a = tc["function"].get("arguments")
            if isinstance(a, str):
                try: tc["function"]["arguments"] = json.loads(a)
                except Exception: pass
    kw = dict(body.get("chat_template_kwargs", {}))
    if "reasoning_effort" in body: kw["reasoning_effort"] = body["reasoning_effort"]
    ref = tpl.render(messages=msgs, tools=body.get("tools"), add_generation_prompt=True, bos_token="", eos_token="", **kw)
    got = open(os.path.join(D, name + ".txt"), encoding="utf-8").read()
    if got == ref: print(name, "identical")
    else:
        bad += 1; print(name, "DIFFERENT")
        for l in list(difflib.unified_diff(ref.splitlines(), got.splitlines(), "jinja2", "cpp-engine", lineterm="", n=1))[:12]: print("  ", l)
print("PASS" if not bad else "FAIL", len(cases), "cases,", bad, "differ")
sys.exit(1 if bad else 0)
