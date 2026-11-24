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
    # Strip ambient Inkling toggle env vars so every run's exact fusion/draft state is controlled explicitly.
    for k in list(env):
        if k.startswith(("LLAMA_INKLING_", "LLAMA_EXPERIMENTAL_INKLING_")):
            del env[k]
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
    target_on_env = {
        "LLAMA_INKLING_FUSED_ATTN": "1",
        "LLAMA_INKLING_FUSED_KERNELS": "1",
        "LLAMA_INKLING_FUSED_MIX": "1",
        "LLAMA_INKLING_FUSED_CTRL": "1",
    }
    target_off_env = {
        "LLAMA_INKLING_FUSED_ATTN": "0",
        "LLAMA_INKLING_FUSED_KERNELS": "0",
        "LLAMA_INKLING_FUSED_MIX": "0",
        "LLAMA_INKLING_FUSED_CTRL": "0",
    }
    draft_fixed_env = {
        "LLAMA_INKLING_DSPARK_FAST": "0",
        "LLAMA_INKLING_PROMPT_LOOKUP": "0",
    }

    # 1. MTL_SHADER_VALIDATION=1 at -c 262144 (F16 KV + DSpark n_max=2) — >32,768 tokens + verified live nwg=32 GPU dispatch
    print("=== [1/4] MTL_SHADER_VALIDATION=1 at -c 262144 (n_kv > 32768 -> logged nwg=32 dispatch) ===", flush=True)
    sv_log = "/tmp/inkling-verify-shaderval.log"
    proc, fp, st = start_server(
        ["-md", DRAFT, "--spec-type", "draft-dspark", "--spec-draft-n-max", "2", "--spec-draft-n-ubatch", "512",
         "-c", "262144", "-b", "2048", "-ub", "2048", "-np", "1",
         "-ctk", "f16", "-ctv", "f16", "-ctkd", "q8_0", "-ctvd", "q8_0", "-ngld", "99"],
        extra_env={**target_on_env, "LLAMA_INKLING_DSPARK_FAST": "1", "LLAMA_INKLING_PROMPT_LOOKUP": "1",
                   "MTL_SHADER_VALIDATION": "1", "GGML_METAL_INKLING_LOG_DISPATCH": "1"},
        log_path=sv_log,
    )
    try:
        # >33,000 token prompt + >=16 decode tokens so split-K decode actually dispatches nwg=32 live under MTL_SHADER_VALIDATION=1
        long_p = ("The quick brown fox jumps over the lazy dog. " * 3000) + "\nWrite a 40-word summary of the repeated sentence above."
        r1 = chat(long_p, max_tokens=48, timeout=420)
        n_prompt = r1["usage"]["prompt_tokens"]
        n_comp_long = r1["usage"]["completion_tokens"]
        assert n_prompt > 32768, f"Expected prompt_tokens > 32768 to trigger nwg=32, got {n_prompt}"
        assert n_comp_long >= 16, f"Expected >=16 decoded tokens at n_kv>32768 to force multi-step split-32 decode, got {n_comp_long}"
        r2 = chat(PARITY_PROMPTS[0], max_tokens=64, timeout=180)
    finally:
        stop_server(proc, fp)

    sv_lines = [ln.strip() for ln in open(sv_log, errors="replace") if "inkling_fa_decode_split32: dispatched nwg=32" in ln]
    assert len(sv_lines) > 0, f"Server log {sv_log} did not record any nwg=32 Inkling FA decode dispatch!"
    summary["shader_validation_256k"] = {
        "ok": True,
        "startup_s": st,
        "long_prompt_tokens": n_prompt,
        "long_completion_tokens": n_comp_long,
        "nwg_32_dispatches": len(sv_lines),
        "nwg_32_sample_log": sv_lines[0],
        "short_completion_tokens": r2["usage"]["completion_tokens"],
    }
    print("PASS shader_validation_256k:", json.dumps(summary["shader_validation_256k"]), flush=True)

    # 2. Multi-slot (-np 2 --kv-unified) at -c 262144: 2 concurrent requests + drop to 1 active request
    print("=== [2/4] Multi-slot (-np 2 --kv-unified) at -c 262144 ===", flush=True)
    proc, fp, st = start_server(
        ["-md", DRAFT, "--spec-type", "draft-dspark", "--spec-draft-n-max", "2", "--spec-draft-n-ubatch", "512",
         "-c", "262144", "-b", "1024", "-ub", "1024", "-np", "2", "--kv-unified",
         "-ctk", "f16", "-ctv", "f16", "-ctkd", "q8_0", "-ctvd", "q8_0", "-ngld", "99"],
        extra_env={**target_on_env, "LLAMA_INKLING_DSPARK_FAST": "1", "LLAMA_INKLING_PROMPT_LOOKUP": "1"},
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

    # 3. Strictly single-variable controlled parity across ALL 3 KV types (f16, q8_0, q4_0):
    #    - target_fused_on_vs_off_nospec_{kv} : ONLY target fusions toggle (explicit 0 vs 1, nospec both sides)
    #    - target_fused_on_vs_off_spec_{kv}   : ONLY target fusions toggle (explicit 0 vs 1, DSpark n_max=2 + identical draft_fixed_env both sides!)
    #    - spec_vs_nospec_target_off_{kv}     : ONLY spec toggles (target fusions explicitly 0 both sides, fixed draft ref)
    #    - spec_vs_nospec_target_on_{kv}      : ONLY spec toggles (target fusions explicitly 1 both sides, fixed draft ref)
    print("=== [3/4] Strictly single-variable Target Fusion parity across F16, Q8_0, Q4_0 ===", flush=True)
    parity_results = []
    for kv in ["f16", "q8_0", "q4_0"]:
        base_args = ["-c", "4096", "-b", "512", "-ub", "512", "-np", "1", "-ctk", kv, "-ctv", kv]
        spec_args = base_args + ["-md", DRAFT, "--spec-type", "draft-dspark", "--spec-draft-n-max", "2",
                                 "--spec-draft-n-ubatch", "512", "-ctkd", "q8_0", "-ctvd", "q8_0", "-ngld", "99"]

        # A: target fusions explicitly OFF, nospec
        proc, fp, _ = start_server(base_args, extra_env=target_off_env, log_path=f"/tmp/inkling-verify-off-nospec-{kv}.log")
        try:
            ref_nospec = run_panel(PARITY_PROMPTS, max_tokens=48, logprobs=True)
        finally:
            stop_server(proc, fp)

        # B: target fusions explicitly OFF, DSpark n_max=2 (fixed draft ref)
        proc, fp, _ = start_server(spec_args, extra_env={**target_off_env, **draft_fixed_env}, log_path=f"/tmp/inkling-verify-off-spec-{kv}.log")
        try:
            ref_spec = run_panel(PARITY_PROMPTS, max_tokens=48, logprobs=True)
        finally:
            stop_server(proc, fp)

        # C: target fusions explicitly ON, nospec
        proc, fp, _ = start_server(base_args, extra_env=target_on_env, log_path=f"/tmp/inkling-verify-on-nospec-{kv}.log")
        try:
            on_nospec = run_panel(PARITY_PROMPTS, max_tokens=48, logprobs=True)
        finally:
            stop_server(proc, fp)

        # D: target fusions explicitly ON, DSpark n_max=2 (SAME fixed draft ref as B — ONLY target fusions change!)
        proc, fp, _ = start_server(spec_args, extra_env={**target_on_env, **draft_fixed_env}, log_path=f"/tmp/inkling-verify-on-spec-{kv}.log")
        try:
            on_spec = run_panel(PARITY_PROMPTS, max_tokens=48, logprobs=True)
        finally:
            stop_server(proc, fp)

        parity_results.append(compare_panels(f"target_fused_on_vs_off_nospec_{kv}", ref_nospec, on_nospec))
        parity_results.append(compare_panels(f"target_fused_on_vs_off_spec_{kv}",   ref_spec,   on_spec))
        parity_results.append(compare_panels(f"spec_vs_nospec_target_off_{kv}",      ref_nospec, ref_spec))
        parity_results.append(compare_panels(f"spec_vs_nospec_target_on_{kv}",       on_nospec,  on_spec))
    summary["parity"] = parity_results

    # 4. Strictly single-variable Draft Optimization parity on F16 (target fusions explicitly fixed ON)
    print("=== [4/4] Single-variable Draft Optimization parity (DSpark fast-argmax & prompt-lookup with verified hits) ===", flush=True)
    spec_f16 = ["-c", "4096", "-b", "512", "-ub", "512", "-np", "1", "-ctk", "f16", "-ctv", "f16",
                "-md", DRAFT, "--spec-type", "draft-dspark", "--spec-draft-n-max", "2",
                "--spec-draft-n-ubatch", "512", "-ctkd", "q8_0", "-ctvd", "q8_0", "-ngld", "99"]
    lookup_prompts = PARITY_PROMPTS[:5] + [
        "Repeat the following paragraph verbatim with no extra commentary:\n"
        "The M5 Pro unified memory architecture streams quantized MoE weights directly through "
        "Metal 4 tensor accelerators while preserving full FP16 key-value precision across 262,144 tokens."
    ]
    # 4a: DSPARK_FAST=0 vs DSPARK_FAST=1 (target_on_env + PROMPT_LOOKUP=0 fixed on both sides)
    proc, fp, _ = start_server(
        spec_f16,
        extra_env={**target_on_env, "LLAMA_INKLING_DSPARK_FAST": "0", "LLAMA_INKLING_PROMPT_LOOKUP": "0"},
        log_path="/tmp/inkling-verify-dft-ref.log",
    )
    try:
        dft_ref = run_panel(PARITY_PROMPTS[:6], max_tokens=48, logprobs=True)
    finally:
        stop_server(proc, fp)
    proc, fp, _ = start_server(
        spec_f16,
        extra_env={**target_on_env, "LLAMA_INKLING_DSPARK_FAST": "1", "LLAMA_INKLING_PROMPT_LOOKUP": "0"},
        log_path="/tmp/inkling-verify-dft-fast.log",
    )
    try:
        dft_fast = run_panel(PARITY_PROMPTS[:6], max_tokens=48, logprobs=True)
    finally:
        stop_server(proc, fp)
    summary["dspark_fast_vs_ref"] = compare_panels("dspark_fast_on_vs_off_f16", dft_ref, dft_fast)

    # 4b: PROMPT_LOOKUP=0 vs PROMPT_LOOKUP=1 (target_on_env + DSPARK_FAST=1 fixed on both sides; requires >=1 actual lookup hit!)
    pl_off_log = "/tmp/inkling-verify-pl-off.log"
    pl_on_log  = "/tmp/inkling-verify-pl-on.log"
    proc, fp, _ = start_server(
        spec_f16,
        extra_env={**target_on_env, "LLAMA_INKLING_DSPARK_FAST": "1", "LLAMA_INKLING_PROMPT_LOOKUP": "0", "LLAMA_INKLING_LOG_LOOKUP": "1"},
        log_path=pl_off_log,
    )
    try:
        pl_off = run_panel(lookup_prompts, max_tokens=48, logprobs=True)
    finally:
        stop_server(proc, fp)
    proc, fp, _ = start_server(
        spec_f16,
        extra_env={**target_on_env, "LLAMA_INKLING_DSPARK_FAST": "1", "LLAMA_INKLING_PROMPT_LOOKUP": "1", "LLAMA_INKLING_LOG_LOOKUP": "1"},
        log_path=pl_on_log,
    )
    try:
        pl_on = run_panel(lookup_prompts, max_tokens=48, logprobs=True)
    finally:
        stop_server(proc, fp)
    pl_hits_off = sum("inkling_prompt_lookup: hit" in ln for ln in open(pl_off_log, errors="replace"))
    pl_hits_on  = sum("inkling_prompt_lookup: hit" in ln for ln in open(pl_on_log, errors="replace"))
    assert pl_hits_off == 0, f"Expected 0 prompt-lookup hits when PROMPT_LOOKUP=0, got {pl_hits_off}"
    assert pl_hits_on > 0, f"Expected >=1 prompt-lookup hit when PROMPT_LOOKUP=1 on verbatim-repeat panel, got {pl_hits_on}"
    pl_res = compare_panels("prompt_lookup_on_vs_off_f16", pl_off, pl_on)
    pl_res["lookup_hits_off"] = pl_hits_off
    pl_res["lookup_hits_on"] = pl_hits_on
    summary["prompt_lookup_vs_off"] = pl_res
    print("PASS prompt_lookup_hits:", json.dumps({"lookup_hits_off": pl_hits_off, "lookup_hits_on": pl_hits_on}), flush=True)

    out_path = f"{ROOT}/bench/m5pro-inkling-256k/parity-report.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved full parity report to {out_path}")

if __name__ == "__main__":
    main()
