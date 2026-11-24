# Optimization Plan & Experiment Log: Inkling-2-Mini 256K+ on Apple M5 Pro

## Targets & Constraints
- **Target model:** `Inkling-2-Mini-AD-Q4_K.gguf` + `dspark-Inkling-2-Mini-Q8_0.gguf` on `atomic-engine` (forked from `AtomicBot-ai/atomic-engine` @ `e110381` -> `Shaderzx/atomic-engine`, branch `optim/m5pro-inkling-256k`).
- **Hard requirement:** **256K context window minimum** (`n_ctx >= 262,144`), stable with exact 3/3 needle retrieval at ~50% (128K) and ~90% (230K–250K) context, plus live OpenCode tool-calling & reasoning verified end-to-end.
- **Memory cap:** **46 GiB** unified memory watchdog limit (out of 64 GB RAM / 51.84 GiB recommended working set).
- **Quality constraint:** Target forward pass & verification remain **100% lossless** (exact greedy output match against unmodified baseline). Draft-side optimizations (shortlist, quantization, lookup) are verified by the exact target model so target output distribution is unchanged.
- **Metrics:**
  1. `c1_decode`: median of 6 fixed-panel prompts (server usage token count)
  2. `prefill_4096`, `prefill_32768`, `prefill_256k` (cold unique documents)
  3. `warm_ttft`: prefix-cache hit turn-2 TTFT
  4. `ttft_short_ms`, `startup_s`, `mem_peak_gib`
  5. `long` (128K & ~240K needle retrieval + decode speed at 256K context)
- **Council target:** Score >= **9.5/10** from the 3-member council (Claude Opus 5.5, GPT-6.2 Sol, Muse Wonder Full 1.0) on linked environment `fm-fsn-04-ci`.

## Ranked Hypothesis Backlog (by ceiling gap × likelihood)
1. **H1 — Native On-the-Fly Inkling Relative-Bias FlashAttention (`fa_tensor` Metal 4 prefill + `fa_vec` decode):**
   - Eliminates 51.5 GiB `kq_b` OOM at 256K prefill, eliminates 1.07 GiB 2D `self_rel_pos` CPU fill, eliminates 800 MB/step `kq_b` traffic at 256K decode, unlocks Metal 4 `matmul2d` Neural Accelerators for Inkling prefill at ALL lengths up to 1M!
2. **H2 — Unlock & Generalize Resident Inkling Metal Kernels (`clip`, `conv_residual`, `control`, `situ`, `vocab_add`):**
   - Currently gated on `expert_executor != nullptr` and/or `n_tokens == 4`. Generalize to ALL `n_tokens` and resident mode (`expert_executor == nullptr`). Cuts ~3,500 graph ops/step -> ~430 fused dispatches.
3. **H3 — Grouped Expert Kernel (`inkling_group_project`) for Resident Q4_K + Q5_0 Verify (`M=2..8`):**
   - Currently gated on `GGML_METAL_INKLING_GROUP_BANK` and skips `Q4_K`! Add resident support + fast `Q4_K` grouped dot (and benchmark vs `mul_mv_id` / `mul_mm_id` at `M=2..8`) so shared experts across verify tokens are loaded ONCE instead of `M` times!
4. **H4 — DSpark Drafter Acceleration (100% Lossless Target Verification):**
   - Raise draft `n_ubatch` (32 -> 512/1024) + zero-copy/device target feature injection (skip unnecessary full-prompt host copies during prefill!).
   - Project ONLY the `n_max` active mask rows through `target->output` (skip row 0 and unused rows `n_max+1..7`).
   - Pruned draft vocab shortlist (top-K by `markov_w1` L2 norm / hybrid vocab, e.g., 32K–49K) + quantized `markov_w2` (BF16 -> Q8_0) + single-pass fused GPU argmax (zero 200K logit CPU copy!).
   - Sweep draft depth `n_max` (3, 4, 5, 6, 7) on the fast drafter.
5. **H5 — Smart Prompt Suffix Lookup + DSpark Hybrid:**
   - Enable suffix lookup across full 256K history for any temperature/greedy where exact continuation matches (proposing longer exact blocks on repeated/edited code, falling back to DSpark otherwise).
6. **H6 — KV Cache Dtype & Batch/Concurrency Tuning at 256K (`Q4_0` vs `Q8_0` vs `F16` KV, `-b`/`-ub`, `GGML_METAL_CONCURRENCY_DISABLE`, `-t`):**
   - Measure actual tradeoffs at 4K, 32K, and 256K under the 46 GiB cap once H1 unlocks `fa_tensor` for all KV dtypes!
7. **H7 — Target LM Head Optimization (if still a large share after H1–H4):**
   - Profile whether `output.weight` (`Q8_0`, 492 MB) is worth two-stage/fused acceleration or stays well-amortized at `M=5..8`.

## Experiment Ledger

| Run | Mode / Flags | c1 decode (tok/s) | Prefill 4K (tok/s) | Prefill 32K (tok/s) | Warm 32K TTFT (ms) | Short TTFT (ms) | Startup (s) | Mem Peak (GiB) | Notes / Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `base-64k-nospec` | Unmodified `e110381`, `-c 65536`, no spec | 31.49 | 374.2 | — | — | 174.1 | 1.52 | — | Baseline nospec (before mem_macos fix) |
| `base-64k` | Unmodified `e110381`, `-c 65536`, DSpark 3/32 | 37.00 | 370.1 | 224.3 (173.7s) | 411.0 | 176.7 | 1.52 | — | Baseline spec (`mean_len=2.22`) |
| `base-256k` | Unmodified `e110381`, `-c 262144` | FAIL | FAIL | FAIL | — | — | hang | 42.97 | OOM/hang in `sched_reserve` (51.5 GiB `kq_b`) |
| `w1-256k-nospec` | Wave 1 fused Inkling FA (`-c 262144`, Q4_0 KV, no spec) | 32.39 | 676.4 (+80.8%) | — | — | 168.3 | 2.53 | 18.39 | **ACCEPTED**: 256K unlocked, 1.81x prefill |
| `w1-256k-q4kv` | Wave 1 fused Inkling FA (`-c 262144`, Q4_0 KV, DSpark 3/32) | 35.73 | 650.8 (+75.8%) | 591.7 (65.8s, 2.64x) | 280.9 (-32%) | 171.4 | 1.53 | 20.38 | **ACCEPTED**: 256K works in 20.4 GiB, 2.64x 32K prefill |
| `ctl-w2-off` (A) | Wave 2 control (`LLAMA_INKLING_FUSED_KERNELS=0`) | 34.80 | 652.8 | — | — | 179.9 | 2.03 | 20.30 | Interleaved control A |
| `w2-fused` | Wave 2 resident fused kernels (`control`, `clip`, `conv`, `situ`, `vocab_add`) | 37.90 (+8.1%) | 731.9 (+11.7%) | 649.2 (60.0s, 2.89x) | — | 152.5 | 1.53 | 20.49 | **ACCEPTED**: +8.1% c1, +11.7% 4K, 2.89x 32K vs base |
| `ctl-w2-off` (B) | Wave 2 control (`LLAMA_INKLING_FUSED_KERNELS=0`) | 35.33 | 657.8 | — | — | 174.8 | 1.53 | 20.47 | Interleaved control B |
| `w2-fused-nospec` | Wave 2 resident fused kernels, `-c 262144`, no spec | **38.18** (+21.2%) | **771.2** (+106%) | — | — | 142.2 | 1.02 | 18.71 | Plain nospec now beats unmodified DSpark 3/32! |
