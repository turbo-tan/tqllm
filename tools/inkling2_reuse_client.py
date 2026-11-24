#!/usr/bin/env python3
"""Inkling conversation reuse through tqserve: a two-turn chat over HTTP (temperature 0, streaming).

usage: inkling2_reuse_client.py URL TEXT_FILE TOKENS_APPROX turn1|turn2 STATE_JSON
  turn1: sends the long user turn (about TOKENS_APPROX tokens of TEXT_FILE), keeps the reply in STATE_JSON
  turn2: sends [user 1, assistant reply 1, user 2] and prints TTFT and the reply; run it once on the server that served
         turn 1 (reuse) and once on a freshly started server (the 1.7.0 path) and compare the replies.
"""
import json
import sys
import time
import urllib.request


def chat(url, messages, max_tokens=32):
    body = json.dumps({"model": "inkling", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    text = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            for ch in ev.get("choices", []):
                d = ch.get("delta") or {}
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")  # a thinking reply streams reasoning first
                if piece and ttft is None:
                    ttft = time.time() - t0
                text.append(piece)
    return ttft, "".join(text), time.time() - t0


def main():
    url, text_file, approx, turn, state = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    words = open(text_file, encoding="utf-8", errors="replace").read().split()
    n_words = int(approx * 0.54)  # this text: ~0.54 words a token (measured: 1,106 words -> 2,048 tokens)
    long_text = " ".join((words * (1 + n_words // max(1, len(words))))[:n_words])
    user1 = {"role": "user", "content": long_text + "\n\nSummarise the text above in one sentence."}
    if turn == "turn1":
        ttft, reply, total = chat(url, [user1])
        json.dump({"reply": reply}, open(state, "w"))
        print(f"turn1 ttft {ttft if ttft is not None else -1:.3f} s total {total:.3f} s reply {json.dumps(reply)}")
    else:
        reply1 = json.load(open(state))["reply"]
        msgs = [user1, {"role": "assistant", "content": reply1}, {"role": "user", "content": "Now name the three most important points as a list."}]
        ttft, reply, total = chat(url, msgs)
        print(f"turn2 ttft {ttft if ttft is not None else -1:.3f} s total {total:.3f} s reply {json.dumps(reply)}")


if __name__ == "__main__":
    main()
