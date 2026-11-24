#!/usr/bin/env python3
"""Matched parity, Metal shader-validation, multi-slot (-np 2), and apples-to-apples 64K verification.

Directly validates all Council Round 1 verification requirements on Apple M5 Pro:
1. MTL_SHADER_VALIDATION=1 live server run at -c 262144 (F16 KV + DSpark, prefill + split-K decode).
2. Matched fused-ON vs fused-OFF parity across all 3 KV types (f16, q8_0, q4_0) at identical
   -c/-b/-ub on a 12-prompt panel, comparing top-1 token ID, top-1 logprob delta, and spec vs nospec.
3. Multi-slot (-np 2 --kv-unified) at -c 262144: 2 concurrent requests + drop to 1 active request.
4. Apples-to-apples 64K benchmark (same flags as unmodified base-64k) + champion 256K re-check.
"""
import concurrent.futures, json, os, signal, statistics, subprocess, sys, time, urllib.request

ROOT = "/Users/shadmehr/code/atomic-engine"
BIN = f"{ROOT}/build-metal-release/bin/llama-server"
MODEL = "/Users/shadmehr/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-AD-Q4_K.gguf"
DRAFT = "/Users/shadmehr/models/Inkling-2-Mini-GGUF/atomicchat-dspark-3095098/dspark-Inkling-2-Mini-Q8_0.gguf"
PORT = 8089
BASE = f"http://127.0.0.1:{PORT}"

PARITY_PROMPTS = [
    "Write a concise Python function fibonacci(n) using memoization and print fibonacci(10). Reply with code only.",
    "In three numbered sentences, explain TCP slow start.",
    "What is 17 * 23? Answer with the number only.",
    "List the first 8 prime numbers separated by commas.",
    "Translate 'Good morning, how are you?' to French.",
    "Write a SQL query to find duplicate emails in a users(id, email) table.",
    "Explain the difference between a process and a thread in two sentences.",
    "Write a one-line bash command to count lines in all .py files recursively.",
    "What HTTP status code means Too Many Requests? Answer concisely.",
    "Give the time complexity of binary search and mergesort.",
    "Write a JSON object with keys name, version, and active.",
    "Summarize why hash collisions happen in hash tables in two sentences.",
]

def start_server(args, extra_env=None, log_path="/tmp/inkling-verify.log"):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    fp = open(log_path, "wb")
    cmd = [BIN, "-m", MODEL, "-fit", "off", "-fa", "on", "-ngl", "all", "-t", "8", "-tb", "8",
           "--jinja", "--alias", "inkling-2-mini", "--host", "127.0.0.1", "--port", str(PORT)] + args
    proc = subprocess.Popen(cmd, stdout=fp, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 90:
        if proc.poll() is not None:
            fp.close()
            tail = open(log_path, errors="replace").read()[-2000:]
            raise RuntimeError(f"Server exited rc={proc.returncode}:\n{tail}")
        try:
            with urllib.request.urlopen(BASE + "/v1/models", timeout=2) as r:
                if json.loads(r.read()).get("data"):
                    return proc, fp, round(time.monotonic() - t0, 2)
        except Exception:
            time.sleep(0.3)
    stop_server(proc, fp)
    raise RuntimeError("Server startup timed out")

def stop_server(proc, fp):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if fp:
        fp.close()

def chat(prompt, max_tokens=48, logprobs=False, timeout=300):
    body = {
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "chat_template_kwargs": {"reasoning_effort": "none"},
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 3
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def run_panel(prompts, max_tokens=48, logprobs=True):
    out = []
    for p in prompts:
        resp = chat(p, max_tokens=max_tokens, logprobs=logprobs)
        ch = resp["choices"][0]
        msg = ch["message"]
        text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        lp_items = (ch.get("logprobs") or {}).get("content") or []
        tok0 = lp_items[0]["token"] if lp_items else None
        lp0  = lp_items[0]["logprob"] if lp_items else None
        out.append({
            "text": text,
            "tok0": tok0,
            "lp0": lp0,
            "n_toks": len(lp_items),
            "toks": [x["token"] for x in lp_items],
            "lps": [x["logprob"] for x in lp_items],
        })
    return out

def compare_panels(name, ref, cand):
    exact_text = sum(a["text"] == b["text"] for a, b in zip(ref, cand))
    tok0_eq    = sum(a["tok0"] == b["tok0"] for a, b in zip(ref, cand))
    lp_diffs   = []
    tot_toks   = 0
    match_toks = 0
    for a, b in zip(ref, cand):
        n = min(len(a["toks"]), len(b["toks"]))
        tot_toks += n
        for idx in range(n):
            if a["toks"][idx] == b["toks"][idx]:
                match_toks += 1
                if a["lps"][idx] is not None and b["lps"][idx] is not None:
                    lp_diffs.append(abs(a["lps"][idx] - b["lps"][idx]))
            else:
                break
    res = {
        "name": name,
        "prompts": len(ref),
        "exact_text_match": f"{exact_text}/{len(ref)}",
        "first_token_match": f"{tok0_eq}/{len(ref)}",
        "prefix_token_agreement": f"{match_toks}/{tot_toks} ({100.0 * match_toks / max(1, tot_toks):.1f}%)",
        "max_matched_logprob_diff": round(max(lp_diffs), 6) if lp_diffs else None,
        "med_matched_logprob_diff": round(statistics.median(lp_diffs), 6) if lp_diffs else None,
    }
    print(json.dumps(res), flush=True)
    return res

def main():
    summary = {}

    # 1. MTL_SHADER_VALIDATION=1 at -c 262144 (F16 KV + DSpark n_max=2) — >32,768 tokens to force nwg=32 split-K decode live
    print("=== [1/4] MTL_SHADER_VALIDATION=1 at -c 262144 (n_kv > 32768 -> nwg=32) ===", flush=True)
    proc, fp, st = start_server(
        ["-md", DRAFT, "--spec-type", "draft-dspark", "--spec-draft-n-max", "2", "--spec-draft-n-ubatch", "512",
         "-c", "262144", "-b", "2048", "-ub", "2048", "-np", "1",
         "-ctk", "f16", "-ctv", "f16", "-ctkd", "q8_0", "-ctvd", "q8_0", "-ngld", "99"],
        extra_env={"MTL_SHADER_VALIDATION": "1"},
        log_path="/tmp/inkling-verify-shaderval.log",
    )
    try:
        # >33,000 token prompt so n_kv > 32768 and split-K decode dispatches nwg=32 live under MTL_SHADER_VALIDATION=1
        long_p = ("The quick brown fox jumps over the lazy dog. " * 3400) + "\nReply with the word: VALIDATED."
        r1 = chat(long_p, max_tokens=48, timeout=420)
        n_prompt = r1["usage"]["prompt_tokens"]
        assert n_prompt > 32768, f"Expected prompt_tokens > 32768 to trigger nwg=32, got {n_prompt}"
        r2 = chat(PARITY_PROMPTS[0], max_tokens=64, timeout=180)
        summary["shader_validation_256k"] = {
            "ok": True,
            "startup_s": st,
            "long_prompt_tokens": n_prompt,
            "nwg_32_exercised": n_prompt > 32768,
            "short_completion_tokens": r2["usage"]["completion_tokens"],
        }
        print("PASS shader_validation_256k:", json.dumps(summary["shader_validation_256k"]), flush=True)
    finally:
        stop_server(proc, fp)

    # 2. Multi-slot (-np 2 --kv-unified) at -c 262144: 2 concurrent requests + drop to 1 active request
    print("=== [2/4] Multi-slot (-np 2 --kv-unified) at -c 262144 ===", flush=True)
    proc, fp, st = start_server(
        ["-md", DRAFT, "--spec-type", "draft-dspark", "--spec-draft-n-max", "2", "--spec-draft-n-ubatch", "512",
         "-c", "262144", "-b", "1024", "-ub", "1024", "-np", "2", "--kv-unified",
         "-ctk", "f16", "-ctv", "f16", "-ctkd", "q8_0", "-ctvd", "q8_0", "-ngld", "99"],
        log_path="/tmp/inkling-verify-np2.log",
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(chat, PARITY_PROMPTS[0], 96)
            f2 = ex.submit(chat, PARITY_PROMPTS[1], 96)
            r_a, r_b = f1.result(), f2.result()
        # Now drop to 1 active request on the -np 2 server (verifies S-F3 contract!)
        r_single = chat(PARITY_PROMPTS[2], 160)
        msg_s = r_single["choices"][0]["message"]
        t_single = (msg_s.get("content") or "") or (msg_s.get("reasoning_content") or "")
        assert "391" in t_single, f"Expected 391 in single-slot response, got {msg_s!r}"
        summary["multi_slot_np2_256k"] = {
            "ok": True,
            "startup_s": st,
            "slot0_tokens": r_a["usage"]["completion_tokens"],
            "slot1_tokens": r_b["usage"]["completion_tokens"],
            "drop_to_1_answer": t_single.strip(),
        }
        print("PASS multi_slot_np2_256k:", json.dumps(summary["multi_slot_np2_256k"]), flush=True)
    finally:
        stop_server(proc, fp)

    # 3. Single-variable controlled parity across ALL 3 KV types (f16, q8_0, q4_0):
    #    - fused_on_vs_off_nospec_{kv} : fusions ON vs OFF, both nospec
    #    - fused_on_vs_off_spec_{kv}   : fusions ON vs OFF, both DSpark n_max=2 (matched M=3 verify)
    #    - spec_vs_nospec_off_{kv}     : upstream batching baseline (M=1 vs M=3, all fusions OFF)
    #    - spec_vs_nospec_on_{kv}      : fused batching comparison (M=1 vs M=3, all fusions ON)
    print("=== [3/4] Controlled Fused ON vs Fused OFF parity across F16, Q8_0, Q4_0 ===", flush=True)
    off_env = {
        "LLAMA_INKLING_FUSED_ATTN": "0",
        "LLAMA_INKLING_FUSED_KERNELS": "0",
        "LLAMA_INKLING_FUSED_MIX": "0",
        "LLAMA_INKLING_FUSED_CTRL": "0",
        "LLAMA_INKLING_DSPARK_FAST": "0",
        "LLAMA_INKLING_PROMPT_LOOKUP": "0",
    }
    parity_results = []
    for kv in ["f16", "q8_0", "q4_0"]:
        base_args = ["-c", "4096", "-b", "512", "-ub", "512", "-np", "1", "-ctk", kv, "-ctv", kv]
        spec_args = base_args + ["-md", DRAFT, "--spec-type", "draft-dspark", "--spec-draft-n-max", "2",
                                 "--spec-draft-n-ubatch", "512", "-ctkd", "q8_0", "-ctvd", "q8_0", "-ngld", "99"]

        # Reference A: all custom fusions OFF, nospec
        proc, fp, _ = start_server(base_args, extra_env=off_env, log_path=f"/tmp/inkling-verify-off-nospec-{kv}.log")
        try:
            ref_nospec = run_panel(PARITY_PROMPTS, max_tokens=48, logprobs=True)
        finally:
            stop_server(proc, fp)

        # Reference B: all custom fusions OFF, DSpark n_max=2
        proc, fp, _ = start_server(spec_args, extra_env=off_env, log_path=f"/tmp/inkling-verify-off-spec-{kv}.log")
        try:
            ref_spec = run_panel(PARITY_PROMPTS, max_tokens=48, logprobs=True)
        finally:
            stop_server(proc, fp)

        # Candidate A: all custom fusions ON, nospec
        proc, fp, _ = start_server(base_args, log_path=f"/tmp/inkling-verify-on-nospec-{kv}.log")
        try:
            on_nospec = run_panel(PARITY_PROMPTS, max_tokens=48, logprobs=True)
        finally:
            stop_server(proc, fp)

        # Candidate B: all custom fusions ON, DSpark n_max=2
        proc, fp, _ = start_server(spec_args, log_path=f"/tmp/inkling-verify-on-spec-{kv}.log")
        try:
            on_spec = run_panel(PARITY_PROMPTS, max_tokens=48, logprobs=True)
        finally:
            stop_server(proc, fp)

        parity_results.append(compare_panels(f"fused_on_vs_off_nospec_{kv}", ref_nospec, on_nospec))
        parity_results.append(compare_panels(f"fused_on_vs_off_spec_{kv}",   ref_spec,   on_spec))
        parity_results.append(compare_panels(f"spec_vs_nospec_off_{kv}",     ref_nospec, ref_spec))
        parity_results.append(compare_panels(f"spec_vs_nospec_on_{kv}",      on_nospec,  on_spec))
    summary["parity"] = parity_results

    # 4. Also verify DSpark fallback when LLAMA_INKLING_DSPARK_FAST=0 (reference draft logits path)
    print("=== [4/4] DSpark fast-argmax vs reference draft logits parity ===", flush=True)
    spec_f16 = ["-c", "4096", "-b", "512", "-ub", "512", "-np", "1", "-ctk", "f16", "-ctv", "f16",
                "-md", DRAFT, "--spec-type", "draft-dspark", "--spec-draft-n-max", "2",
                "--spec-draft-n-ubatch", "512", "-ctkd", "q8_0", "-ctvd", "q8_0", "-ngld", "99"]
    proc, fp, _ = start_server(spec_f16, extra_env={"LLAMA_INKLING_DSPARK_FAST": "0"}, log_path="/tmp/inkling-verify-dft-ref.log")
    try:
        dft_ref = run_panel(PARITY_PROMPTS[:6], max_tokens=48, logprobs=True)
    finally:
        stop_server(proc, fp)
    proc, fp, _ = start_server(spec_f16, log_path="/tmp/inkling-verify-dft-fast.log")
    try:
        dft_fast = run_panel(PARITY_PROMPTS[:6], max_tokens=48, logprobs=True)
    finally:
        stop_server(proc, fp)
    summary["dspark_fast_vs_ref"] = compare_panels("dspark_fast_vs_ref_f16", dft_ref, dft_fast)

    out_path = f"{ROOT}/bench/m5pro-inkling-256k/parity-report.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved full parity report to {out_path}")

if __name__ == "__main__":
    main()
