#!/usr/bin/env python3
"""Deterministic 10-task coding, reasoning & tool-call quality + OpenCode coding speed eval."""
import json, re, statistics, sys, time, urllib.request


def chat(base_url, model, prompt, max_tokens=512, temperature=0.0, effort=None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
        "seed": 42,
    }
    if effort is not None:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    t_first = None
    content = []
    reasoning = []
    usage = {}
    timings = {}
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            if ev.get("timings"):
                timings = ev["timings"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                c = d.get("content") or ""
                rc = d.get("reasoning_content") or d.get("reasoning") or ""
                if c or rc:
                    if t_first is None:
                        t_first = time.monotonic()
                if c:
                    content.append(c)
                if rc:
                    reasoning.append(rc)
    t_end = time.monotonic()
    ct = usage.get("completion_tokens") or 0
    tps = (ct - 1) / (t_end - t_first) if ct > 1 and t_first and t_end > t_first else 0.0
    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "tokens": ct,
        "tps": round(tps, 2),
        "srv_tps": round(timings.get("predicted_per_second", 0.0), 2),
        "wall": round(t_end - t0, 2),
    }


def extract_code(text):
    m = re.findall(r"```(?:python|py|json)?\s*\n(.*?)```", text, re.S)
    return m[-1].strip() if m else text.strip()


def run_py(code, tests):
    ns = {}
    try:
        exec(code, ns)
        exec(tests, ns)
        return True, "ok"
    except Exception as e:
        return False, repr(e)


TASKS = [
    (
        "merge_intervals",
        "Write ONLY a Python function `merge_intervals(intervals)` (inside a ```python block) that merges overlapping or touching closed intervals [[s,e], ...] and returns them sorted.",
        "assert merge_intervals([]) == []\n"
        "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
        "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n"
        "assert merge_intervals([[5,7],[1,10],[2,3]]) == [[1,10]]\n",
        "py",
    ),
    (
        "eval_rpn",
        "Write ONLY a Python function `eval_rpn(tokens)` (inside a ```python block) evaluating Reverse Polish Notation (+, -, *, /) with integer division truncating toward zero.",
        "assert eval_rpn(['2','1','+','3','*']) == 9\n"
        "assert eval_rpn(['4','13','5','/','+']) == 6\n"
        "assert eval_rpn(['10','6','9','3','+','-11','*','/','*','17','+','5','+']) == 22\n"
        "assert eval_rpn(['-7','2','/']) == -3\n",
        "py",
    ),
    (
        "binary_search_first_ge",
        "Fix this buggy Python function and return ONLY the corrected `first_ge(arr, target)` inside a ```python block:\n```python\ndef first_ge(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] > target: hi = mid - 1\n        else: lo = mid\n    return lo\n```\nIt must return the smallest index i with arr[i] >= target, or len(arr) if none exists.",
        "assert first_ge([], 5) == 0\n"
        "assert first_ge([1,2,2,2,5], 2) == 1\n"
        "assert first_ge([1,3,5], 6) == 3\n"
        "assert first_ge([1,3,5], 0) == 0\n"
        "assert first_ge([1,3,5], 3) == 1\n",
        "py",
    ),
    (
        "topo_sort",
        "Write ONLY a Python function `topo_sort(n, edges)` (inside a ```python block) for directed edges (u,v) on nodes 0..n-1. Return a valid topological ordering list, or None if there is a cycle.",
        "r = topo_sort(4, [(0,1),(0,2),(1,3),(2,3)])\n"
        "assert r is not None and len(r) == 4 and set(r) == {0,1,2,3}\n"
        "pos = {v:i for i,v in enumerate(r)}\n"
        "assert all(pos[u] < pos[v] for u,v in [(0,1),(0,2),(1,3),(2,3)])\n"
        "assert topo_sort(3, [(0,1),(1,2),(2,0)]) is None\n",
        "py",
    ),
    (
        "lru_cache",
        "Write ONLY a Python class `LRUCache` with `__init__(self, capacity)`, `get(self, key)` (-1 if absent), and `put(self, key, value)` in O(1) time inside a ```python block.",
        "c = LRUCache(2)\n"
        "c.put(1, 1); c.put(2, 2); assert c.get(1) == 1\n"
        "c.put(3, 3); assert c.get(2) == -1\n"
        "c.put(4, 4); assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4\n",
        "py",
    ),
    (
        "parse_semver",
        "Write ONLY a Python function `parse_semver(s)` (inside a ```python block) returning `(major, minor, patch, prerelease)` where the first three are ints and `prerelease` is a string or None. Raise ValueError on invalid input.",
        "assert parse_semver('1.2.3') == (1, 2, 3, None)\n"
        "assert parse_semver('0.10.4-alpha.1') == (0, 10, 4, 'alpha.1')\n"
        "try:\n    parse_semver('1.2')\n    assert False\nexcept ValueError:\n    pass\n",
        "py",
    ),
    (
        "json_tool_call",
        'Output ONLY a valid JSON object (no markdown prose outside JSON) matching this tool schema: {"name": "edit_file", "arguments": {"path": string, "line": int, "replacement": string}} for replacing line 42 of src/router.cpp with "return top_k;".',
        None,
        "json_tool",
    ),
    (
        "bandwidth_math",
        "A chip has 307.2 GB/s memory bandwidth and reads 2.56 GB of weights per token. Assuming 100% bandwidth utilization, how many tokens per second is the ceiling? Reply with the exact integer only.",
        "120",
        "exact_num",
    ),
    (
        "cpp_bug",
        "What bug is in `int* f(){ int x[4]={1,2,3,4}; return x; }`? Answer in one short sentence naming the exact issue.",
        ("dangling", "stack", "local", "lifetime", "undefined"),
        "keywords",
    ),
    (
        "opencode_refactor_speed",
        "Refactor the following Python module to add full type annotations, a `batch_get(keys)` method, and keep all existing logic and docstrings intact. Return the complete updated file.\n\n```python\n"
        "class RingBuffer:\n"
        "    \"\"\"Fixed-size circular buffer storing recent telemetry samples.\"\"\"\n"
        "    def __init__(self, capacity):\n"
        "        if capacity <= 0:\n"
        "            raise ValueError('capacity must be positive')\n"
        "        self._cap = capacity\n"
        "        self._buf = [None] * capacity\n"
        "        self._head = 0\n"
        "        self._size = 0\n\n"
        "    def push(self, key, value):\n"
        "        \"\"\"Insert (key, value), overwriting the oldest entry when full.\"\"\"\n"
        "        idx = (self._head + self._size) % self._cap\n"
        "        if self._size == self._cap:\n"
        "            self._head = (self._head + 1) % self._cap\n"
        "        else:\n"
        "            self._size += 1\n"
        "        self._buf[idx] = (key, value)\n\n"
        "    def get(self, key):\n"
        "        \"\"\"Return the most recent value for key, or None if missing.\"\"\"\n"
        "        for off in range(self._size - 1, -1, -1):\n"
        "            item = self._buf[(self._head + off) % self._cap]\n"
        "            if item is not None and item[0] == key:\n"
        "                return item[1]\n"
        "        return None\n\n"
        "    def items(self):\n"
        "        \"\"\"Return stored (key, value) pairs from oldest to newest.\"\"\"\n"
        "        return [self._buf[(self._head + i) % self._cap] for i in range(self._size)]\n"
        "```\n",
        "rb = RingBuffer(3)\n"
        "rb.push('a', 1); rb.push('b', 2); rb.push('c', 3); rb.push('d', 4)\n"
        "assert rb.get('a') is None and rb.get('d') == 4\n"
        "assert rb.batch_get(['b', 'z', 'd']) == [2, None, 4] or rb.batch_get(['b', 'z', 'd']) == {'b': 2, 'z': None, 'd': 4}\n"
        "assert len(rb.items()) == 3\n",
        "py",
    ),
]


def main():
    args = sys.argv[1:]
    effort = None
    if "--effort" in args:
        idx = args.index("--effort")
        effort = args[idx + 1]
        args = args[:idx] + args[idx + 2 :]
    base = args[0] if len(args) > 0 else "http://127.0.0.1:8089"
    model = args[1] if len(args) > 1 else "inkling-2-mini"
    passed = 0
    tps_all = []
    refactor_tps = 0.0
    results = []
    for name, prompt, check, kind in TASKS:
        r = chat(base, model, prompt, max_tokens=2048, effort=effort)
        text = r["content"] or r["reasoning"]
        ok = False
        detail = ""
        if kind == "py":
            code = extract_code(text)
            ok, detail = run_py(code, check)
        elif kind == "json_tool":
            try:
                obj = json.loads(extract_code(text))
                ok = (
                    obj.get("name") == "edit_file"
                    and obj.get("arguments", {}).get("path") == "src/router.cpp"
                    and obj.get("arguments", {}).get("line") == 42
                    and "return top_k;" in str(obj.get("arguments", {}).get("replacement", ""))
                )
                detail = "ok" if ok else f"bad_json={obj}"
            except Exception as e:
                detail = repr(e)
        elif kind == "exact_num":
            ok = check in re.findall(r"\b\d+\b", text)
            detail = "ok" if ok else f"got={text[:80]!r}"
        elif kind == "keywords":
            low = text.lower()
            ok = any(k in low for k in check)
            detail = "ok" if ok else f"got={text[:80]!r}"
        passed += int(ok)
        if r["tps"] > 0:
            tps_all.append(r["tps"])
        if name == "opencode_refactor_speed":
            refactor_tps = r["tps"]
        results.append({"task": name, "ok": ok, "tps": r["tps"], "srv_tps": r["srv_tps"], "tokens": r["tokens"], "detail": detail})
        print(f"  [{ 'PASS' if ok else 'FAIL' }] {name:24s} tps={r['tps']:6.2f} (srv={r['srv_tps']:6.2f}) toks={r['tokens']:4d} {detail}", flush=True)
    med = round(statistics.median(tps_all), 2) if tps_all else 0.0
    summary = {"passed": passed, "total": len(TASKS), "effort": effort, "coding_med_tps": med, "refactor_tps": refactor_tps, "results": results}
    print(f"SUMMARY (effort={effort}): {passed}/{len(TASKS)} passed | coding_med_tps={med} tok/s | opencode_refactor_tps={refactor_tps} tok/s", flush=True)
    if len(args) > 2:
        with open(args[2], "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
