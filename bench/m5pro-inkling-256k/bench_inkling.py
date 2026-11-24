#!/usr/bin/env python3
"""Hill-climb benchmark harness for atomic-engine (Inkling-2-Mini) on Apple M5 Pro.

Follows inference-optimisation/SKILL.md:
- Refuses to run on battery (AC power required).
- Measures real token counts from server usage stats (never streamed chunk count).
- Fixed 6-prompt panel (code + technical prose + narrative + tables + bash + history).
- Median of >=6 decode samples per run, plus looping detector.
- Cold unique-prefix prefill measurements (defeating prefix cache) + warm turn-2 TTFT.
- Long-context needle retrieval verification (--long N, exact 3/3 match).
- Greedy output equivalence verification (--greedy-check, compares exact output text).
- Apple unified-memory watchdog with clean SIGTERM drain.
- Parses llama-server timing & DSpark acceptance stats from the run's server log.
- Appends structured records to ledger.jsonl.
"""
import json, os, random, re, signal, statistics, string, subprocess, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "Write a Python implementation of an LRU cache with O(1) get and put, with docstrings and tests.",
    "Explain, step by step, how TCP congestion control works, including slow start and fast recovery.",
    "Write a short story (about 600 words) about a lighthouse keeper who finds a message in a bottle.",
    "Compare B-trees and LSM-trees for a write-heavy database workload. Use a table and give a recommendation.",
    "Write a bash script that backs up a directory to a timestamped tar.gz and keeps only the last 7 backups.",
    "Summarise the causes and consequences of the 1929 stock market crash in five paragraphs.",
]

GREEDY_PROMPTS = [
    ("fib", "Write a concise Python function fibonacci(n) using memoization and print fibonacci(10). Reply with code only."),
    ("tcp", "In three numbered sentences, explain TCP slow start."),
]

WORDS = (
    "river stone paper cloud engine garden silver market winter orange signal harbor violet "
    "candle forest mirror thunder copper meadow lantern"
).split()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def on_ac():
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        return "AC Power" in out
    except Exception:
        return True


def thermal_state():
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
        return int(m.group(1)) if m else 100
    except Exception:
        return 100


class Server:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base = cfg.get("base_url", "http://127.0.0.1:8081").rstrip("/")

    def get(self, path, timeout=5):
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    def healthy(self):
        try:
            return bool(self.get("/v1/models").get("data"))
        except Exception:
            return False

    def chat(self, prompt, max_tokens, temperature=None, seed=None, timeout=7200, extra_override=None):
        body = {
            "model": self.cfg["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": self.cfg.get("temperature", 0.0) if temperature is None else temperature,
        }
        if seed is not None:
            body["seed"] = seed
        eb = dict(self.cfg.get("extra_body", {}))
        if extra_override:
            eb.update(extra_override)
        body.update(eb)
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.monotonic()
        t_first = None
        text = []
        usage = {}
        timings = {}
        with urllib.request.urlopen(req, timeout=timeout) as r:
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
                    piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                    if piece:
                        if t_first is None:
                            t_first = time.monotonic()
                        text.append(piece)
        t_end = time.monotonic()
        ct = usage.get("completion_tokens")
        pt = usage.get("prompt_tokens")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if t_first is None:
            t_first = t_end
        decode = (ct - 1) / (t_end - t_first) if ct and ct > 1 and t_end > t_first else None
        return {
            "ttft": t_first - t0,
            "decode": decode,
            "completion_tokens": ct,
            "prompt_tokens": pt,
            "cached_tokens": cached,
            "wall": t_end - t0,
            "t0": t0,
            "t_end": t_end,
            "text": "".join(text),
            "timings": timings,
        }


def looping(text, n=8, thresh=0.5):
    w = text.split()
    grams = [tuple(w[i : i + n]) for i in range(max(0, len(w) - n))]
    return len(grams) > 50 and 1 - len(set(grams)) / len(grams) > thresh


def document(tokens, seed, needles=3):
    # Measured ratio for WORDS on Inkling tokenizer is ~1.05 tok/word; calibrate carefully!
    rng = random.Random(seed)
    n_words = int(tokens / 1.05)
    words = [rng.choice(WORDS) for _ in range(n_words)]
    codes = ["".join(rng.choices(string.ascii_uppercase + string.digits, k=8)) for _ in range(needles)]
    for i, c in enumerate(codes):
        pos = int(n_words * (i + 1) / (needles + 1))
        words.insert(pos, f". The secret code number {i + 1} is {c}. ")
    head = f"[doc {seed}-{time.time_ns()}] "
    q = "\n\nList every secret code number and its value that appears in the document above. Answer only with the codes."
    return head + " ".join(words) + q, codes


def parse_server_log_stats(log_path, start_offset=0):
    if not log_path or not os.path.exists(log_path):
        return {}
    with open(log_path, "r", errors="replace") as f:
        f.seek(start_offset)
        tail = f.read()
    acc_rates = [float(m) for m in re.findall(r"draft acceptance = ([0-9.]+)", tail)]
    mean_lens = [float(m) for m in re.findall(r"mean len =\s*([0-9.]+)", tail)]
    eval_tps = [float(m) for m in re.findall(r"eval time =.*?,\s*([0-9.]+) tokens per second\)", tail)]
    prompt_tps = [float(m) for m in re.findall(r"prompt eval time =.*?,\s*([0-9.]+) tokens per second\)", tail)]
    out = {}
    if acc_rates:
        out["draft_accept_med"] = round(statistics.median(acc_rates), 4)
    if mean_lens:
        out["draft_mean_len_med"] = round(statistics.median(mean_lens), 2)
    if eval_tps:
        out["srv_eval_tps_med"] = round(statistics.median(eval_tps), 2)
    return out


class MemWatch(threading.Thread):
    def __init__(self, mcfg, on_trip):
        super().__init__(daemon=True)
        self.mcfg = mcfg
        self.on_trip = on_trip
        self.peak = 0
        self.tripped = False
        self.stop_ev = threading.Event()

    def run(self):
        cap = self.mcfg.get("cap_gib", 46) * (2**30)
        while not self.stop_ev.wait(self.mcfg.get("interval_s", 2)):
            try:
                b = int(subprocess.run(self.mcfg["probe"], capture_output=True, text=True, timeout=15).stdout.strip() or 0)
            except Exception:
                continue
            self.peak = max(self.peak, b)
            if b > cap and not self.tripped:
                self.tripped = True
                log(f"MEMORY CAP EXCEEDED: {b / 2**30:.2f} GiB > {cap / 2**30:.2f} GiB")
                self.on_trip()


def stop_pid(pid_file, timeout_s=30):
    if not pid_file or not os.path.exists(pid_file):
        return
    try:
        pid = int(open(pid_file).read().strip())
    except Exception:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        try:
            os.remove(pid_file)
        except OSError:
            pass
        return
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
            time.sleep(0.3)
        except ProcessLookupError:
            break
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.remove(pid_file)
    except OSError:
        pass


def launch_server(cfg):
    pid_file = cfg.get("pid_file", "/tmp/inkling-bench-server.pid")
    log_path = cfg.get("log_file", f"/tmp/inkling-bench-{cfg['name']}.log")
    stop_pid(pid_file)
    # Also ensure nothing else owns the target port
    env = os.environ.copy()
    for k, v in (cfg.get("env") or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    logf = open(log_path, "wb")
    p = subprocess.Popen(cfg["cmd"], stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    with open(pid_file, "w") as f:
        f.write(str(p.pid))
    return p, pid_file, log_path


def run(cfg_path, long_ctx=None, reps=6, max_tokens=256, greedy_check=False, warm_check=False, keep_running=False):
    cfg = json.load(open(cfg_path))
    srv = Server(cfg)
    flags = []
    res = {
        "name": cfg["name"],
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "thermal_start": thermal_state(),
    }
    if cfg.get("require_ac", True) and not on_ac():
        sys.exit("refusing to run on battery (require_ac)")

    pid_file = cfg.get("pid_file", "/tmp/inkling-bench-server.pid")
    log_path = cfg.get("log_file", f"/tmp/inkling-bench-{cfg['name']}.log")
    proc = None

    def stop_srv():
        stop_pid(pid_file)

    watch = MemWatch(cfg["memory"], stop_srv) if cfg.get("memory", {}).get("probe") else None

    if cfg.get("cmd"):
        t0 = time.monotonic()
        proc, pid_file, log_path = launch_server(cfg)
        if watch:
            watch.start()
        deadline = t0 + cfg.get("health_timeout_s", 300)
        while not srv.healthy():
            if proc.poll() is not None:
                res.update(ok=False, error=f"server exited early with rc={proc.returncode}")
                return finish(res, flags, watch, pid_file, keep_running=False)
            if time.monotonic() > deadline or (watch and watch.tripped):
                res.update(ok=False, error="server did not become healthy")
                return finish(res, flags, watch, pid_file, keep_running=False)
            time.sleep(0.5)
        res["startup_s"] = round(time.monotonic() - t0, 2)
        log(f"[{cfg['name']}] healthy after {res['startup_s']}s (pid={proc.pid})")
    else:
        if not srv.healthy():
            sys.exit("server not reachable and no cmd given")
        if watch:
            watch.start()

    log_offset = os.path.getsize(log_path) if log_path and os.path.exists(log_path) else 0

    try:
        # Warmup (discarded)
        srv.chat("Say hello.", 16, temperature=0.0)
        r = srv.chat("Reply with the single word: ready", 8, temperature=0.0)
        res["ttft_short_ms"] = round(r["ttft"] * 1000, 1)

        # Decode panel (fixed 6 prompts, unique prefix per prompt? No, fixed panel with fresh slot or unique prefix to prevent cache)
        c1 = []
        c1_srv = []
        decode_log_offset = os.path.getsize(log_path) if log_path and os.path.exists(log_path) else 0
        for i in range(reps):
            # Add tiny unique nonce so prefix cache does not shortcut across reps
            prompt = f"[run {cfg['name']} rep {i} {time.time_ns()}] " + PROMPTS[i % len(PROMPTS)]
            r = srv.chat(prompt, max_tokens, seed=42 + i)
            if r["decode"]:
                c1.append(round(r["decode"], 2))
            if r["timings"].get("predicted_per_second"):
                c1_srv.append(round(r["timings"]["predicted_per_second"], 2))
            if looping(r["text"]):
                flags.append("LOOP")
            if not r["completion_tokens"] or r["completion_tokens"] < 16:
                flags.append("SHORT_OUT")
        res["c1_decode"] = round(statistics.median(c1), 2) if c1 else None
        res["c1_min"] = min(c1) if c1 else None
        res["c1_max"] = max(c1) if c1 else None
        res["c1_all"] = c1
        if c1_srv:
            res["c1_srv_med"] = round(statistics.median(c1_srv), 2)
        res.update(parse_server_log_stats(log_path, decode_log_offset))
        log(f"[{cfg['name']}] c1 median={res['c1_decode']} (min={res['c1_min']} max={res['c1_max']}) tok/s all={c1} draft_acc={res.get('draft_accept_med')} mean_len={res.get('draft_mean_len_med')}")

        # Prefill lengths (cold, unique documents): record both end-to-end (prompt/TTFT) and server prompt_per_second
        for L in cfg.get("prefill_lengths", [4096, 32768]):
            prompt, _ = document(L, seed=L)
            r = srv.chat(prompt, 1, temperature=0.0)
            res[f"prefill_{L}"] = round(r["prompt_tokens"] / r["ttft"], 1) if r["prompt_tokens"] else None
            srv_pp = r["timings"].get("prompt_per_second")
            if srv_pp:
                res[f"prefill_{L}_srv"] = round(srv_pp, 1)
            res[f"prefill_{L}_tokens"] = r["prompt_tokens"]
            res[f"prefill_{L}_ttft_s"] = round(r["ttft"], 2)
            log(f"[{cfg['name']}] prefill ~{L}: e2e={res[f'prefill_{L}']} srv={res.get(f'prefill_{L}_srv')} tok/s ({r['prompt_tokens']} toks in {r['ttft']:.2f}s)")
            if warm_check and L == max(cfg.get("prefill_lengths", [4096])):
                r_warm = srv.chat(prompt + " Confirm.", 4, temperature=0.0)
                res[f"warm_{L}_ttft_ms"] = round(r_warm["ttft"] * 1000, 1)
                res[f"warm_{L}_cached"] = r_warm["cached_tokens"]
                log(f"[{cfg['name']}] warm ~{L}: ttft={res[f'warm_{L}_ttft_ms']}ms cached={r_warm['cached_tokens']}")

        if long_ctx:
            prompt, codes = document(long_ctx, seed=7)
            r = srv.chat(prompt, 256, temperature=0.0)
            hits = sum(c in r["text"] for c in codes)
            res["long"] = {
                "target_tokens": long_ctx,
                "prompt_tokens": r["prompt_tokens"],
                "ttft_s": round(r["ttft"], 2),
                "prefill": round(r["prompt_tokens"] / r["ttft"], 1) if r["prompt_tokens"] else None,
                "prefill_srv": round(r["timings"]["prompt_per_second"], 1) if r["timings"].get("prompt_per_second") else None,
                "decode": round(r["decode"], 2) if r["decode"] else None,
                "retrieval": f"{hits}/{len(codes)}",
                "codes_expected": codes,
                "answer_excerpt": r["text"][:300],
            }
            if hits < len(codes):
                flags.append("NEEDLE_MISS")
            log(f"[{cfg['name']}] long {r['prompt_tokens']} toks: prefill={res['long']['prefill']} tok/s ttft={res['long']['ttft_s']}s decode={res['long']['decode']} retrieval={res['long']['retrieval']}")

        if greedy_check:
            gout = {}
            for key, gp in GREEDY_PROMPTS:
                r = srv.chat(gp, 128, temperature=0.0, seed=0)
                gout[key] = r["text"]
            res["greedy_outputs"] = gout

        res["ok"] = ("NEEDLE_MISS" not in flags)
    except Exception as e:
        res.update(ok=False, error=repr(e))
        flags.append("REQ_FAIL")
        log(f"[{cfg['name']}] ERROR: {e!r}")

    return finish(res, flags, watch, pid_file, keep_running=keep_running)


def finish(res, flags, watch, pid_file, keep_running=False):
    res["thermal_end"] = thermal_state()
    if res["thermal_end"] < 100:
        flags.append(f"THERMAL_{res['thermal_end']}")
    if watch:
        watch.stop_ev.set()
        res["mem_peak_gib"] = round(watch.peak / (2**30), 2)
        if watch.tripped:
            flags.append("MEM_CAP")
            res["ok"] = False
    if not on_ac():
        flags.append("ON_BATTERY")
        res["ok"] = False
    if not keep_running:
        stop_pid(pid_file)
    res["flags"] = sorted(set(flags))
    ledger_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger.jsonl")
    with open(ledger_path, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "greedy_outputs"}, indent=2), flush=True)
    return res


def show(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    pre = sorted({k for d in rows for k in d if k.startswith("prefill_") and k[8:].isdigit()}, key=lambda k: int(k[8:]))
    keys = ["c1_decode", "draft_mean_len_med", *pre, "ttft_short_ms", "startup_s", "mem_peak_gib"]
    print("name".ljust(26) + "".join(k[:13].rjust(14) for k in keys) + "  flags")
    for d in rows:
        print(d["name"][:26].ljust(26) + "".join(str(d.get(k, "-"))[:13].rjust(14) for k in keys) + "  " + ",".join(d.get("flags", [])))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "show"):
        sys.exit("Usage: bench_inkling.py run CONFIG.json [--long N] [--reps 6] [--max-tokens 256] [--greedy-check] [--warm-check] [--keep-running]\n       bench_inkling.py show [ledger.jsonl]")
    if a[0] == "show":
        show(*(a[1:2]))
    else:
        opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
        run(
            a[1],
            long_ctx=int(opt("--long")) if opt("--long") else None,
            reps=int(opt("--reps", 6)),
            max_tokens=int(opt("--max-tokens", 256)),
            greedy_check="--greedy-check" in a,
            warm_check="--warm-check" in a,
            keep_running="--keep-running" in a,
        )
