# Theoretical Ceilings: Inkling-2-Mini (AD-Q4_K + DSpark) on Apple M5 Pro at 256K+ Context

All ceilings below are grounded in direct hardware measurements on this exact machine (Apple M5 Pro, 20-core Metal 4 GPU, 64 GB unified LPDDR5X, AC Power, High Power mode `powermode=2`) and exact tensor byte counts parsed from the GGUF headers.

## 1. Measured Hardware Truth (2026-11-24, AC Power, Thermal 100%)

Measured directly via Metal probe (`/tmp/m5_probe.mm`, median of 10 timed runs after 2 warmups):

- **Device:** Apple M5 Pro (`MTLGPUFamilyApple10`, `MTLGPUFamilyMetal4`, `has_tensor = true`)
- **Memory:** 64 GB unified LPDDR5X (`recommendedMaxWorkingSetSize = 51.84 GiB`, `maxBufferLength = 38.88 GiB`)
- **Measured Copy Bandwidth (1 GiB -> 1 GiB, 2 GiB traffic):** **264.2 GB/s** median (247.5–270.0 GB/s)
- **Measured Read-Only Streaming Bandwidth (1 GiB, models decode reads):** **280.3 GB/s** median (257.3–289.5 GB/s)
- **Measured FP16 GEMM Throughput (MPS / Metal 4 Neural Accelerators):**
  - `4096³`: **30.16 TFLOPS** median (28.35–30.37 TFLOPS)
  - `8192³`: **29.14 TFLOPS** median (28.75–29.20 TFLOPS, only 3.4% thermal droop)
- **Practical ceilings use:** `BW = 280.3 GB/s` (read-only) / `264.2 GB/s` (mixed read/write) for decode, and `0.5 × 29.14 = 14.57 TFLOPS` effective (50% MFU) for prefill.

## 2. Exact 48-Visit Target Byte Model (`Inkling-2-Mini-AD-Q4_K.gguf`)

Inkling-2-Mini (`inkling2_loop`) has 24 physical blocks (`block_count = 24`), with a 12-layer loop core (`layers 6..17`) executed `loop.count = 3` times per token -> **48 layer visits** per forward pass:
- **40 Sliding-Window Attention (SWA) visits** (`sliding_window = 1024`)
- **8 Full-Attention (Global) visits** (physical layers 5, 11×3, 17×3, 23)
- **128 routed experts, top-8 active per visit** (`ffn_gate_up_exps` in `Q4_K`, `ffn_down_exps` in `Q5_0`)
- **Native context length:** `1,048,576` tokens (1M)

Exact byte counts parsed from `Inkling-2-Mini-AD-Q4_K.gguf` (16.51 GB on disk):

| Component | Disk (MB) | 48-visit M=1 Read (MB) | Notes |
|---|---:|---:|---|
| Routed experts (top-8 of 128) | 14,778.63 | **1,847.33** | 1,146.6 MB `Q4_K` gate/up + 700.7 MB `Q5_0` down |
| Attention projections (`Q8_0`) | 690.54 | **1,381.08** | 48 visits (`wq`, `wk`, `wv`, `wo`, `wqkv_gate`, `attn_rel`, `sink`) |
| Target LM Head (`output.weight`, `Q8_0`) | 492.11 | **492.11** | `2304 × 201,024` |
| Loop controllers (`Q8_0`, passes 1 & 2) | 30.52 | **61.05** | 24 looped visits × 4 controllers (rank 64) |
| MoE Router (`F32`) | 28.32 | **56.65** | 48 visits (`2304 × 128`) |
| Norms & shortconv weights (`F32`) | 1.35 | **2.68** | 48 visits |
| Token embedding (`Q8_0`) | 492.11 | ~0.00 | Single row lookup |
| **Total M=1 Target Weights (excl KV)** | **16,513.58** | **3,840.89** | **13.70 ms floor @ 280.3 GB/s -> 73.0 tok/s ceiling** (68.8 @ 264.2 GB/s) |

## 3. KV Cache & Memory Sizing at 256K (`n_ctx = 262,144`)

Target has **8** global KV layers (8 KV heads × 128 dim × K & V) and **40** SWA KV layers (`size_swa = 2048` cells with `ub = 1024`).
DSpark (`dspark-Inkling-2-Mini-Q8_0.gguf`, 492 MB) has **1** global KV layer (`blk.1`) and **2** SWA layers (`blk.0, blk.2`).

| KV Type | Target Global (8L @ 256K) | Target SWA (40L @ 2K) | DSpark Global (1L @ 256K) | Total KV (GiB) | Total Footprint w/ 17 GiB Weights | KV Read / Step @ 256K |
|---|---:|---:|---:|---:|---:|---:|
| `Q4_0` | 2.25 GiB | 0.09 GiB | 0.28 GiB | **2.62 GiB** | **~21.5 GiB** | **2.42 GB** (+0.09 SWA) |
| `Q8_0` | 4.25 GiB | 0.17 GiB | 0.53 GiB | **4.95 GiB** | **~23.8 GiB** | **4.56 GB** (+0.17 SWA) |
| `F16` | 8.00 GiB | 0.31 GiB | 1.00 GiB | **9.31 GiB** | **~28.2 GiB** | **8.59 GB** (+0.31 SWA) |

**CRITICAL 256K DISCOVERY (Relative Bias Tensor Explosion in Unmodified Code):**
In unmodified `src/models/inkling2.cpp:378-396` & `src/llama-kv-cache.cpp:2010`:
- `set_input_relative_pos` allocates & fills `[n_kv, n_tokens]` in I32 on CPU: at `262,144 × 1024`, that is **1.07 GB** per ubatch and **34 billion CPU iterations** over a 256K prefill!
- `kq_b` materializes `[n_kv, n_tokens, 24]` in F32 (without `rel_mask_kernel`) = **25.77 GB × 2 = 51.5 GB** (instant OOM/crash!), or in F16 (with `rel_mask_kernel`) = **12.88 GB**!
- Even at decode (`M=4`, 256K context), materializing `kq_b` writes+reads **100 MB × 8 layers = 800 MB** of pure wasted bandwidth per verify step!
- **Fix:** Pass 1D `k_pos[n_kv]` (1.0 MB max!) + 1D `q_pos[n_tokens]` (4 KB) + `table[1025, 24, n_tokens]` (98 KB at decode) directly into FlashAttention (`fa_tensor` for Metal 4 prefill, `fa_vec` for decode). Because `n_relative_extent = 1024`, 99.6% of keys at 256K share the loop-invariant constant bias `table[qh, 1024] - sink[qh]`! Zero `kq_b`, zero 2D `rel_pos`, zero 2D `causal` mask!

## 4. Speculative Decoding Economics (DSpark + Prompt Lookup)

Measured expert union `U(M)` for Inkling-2-Mini (48 visits, top-8 of 128):
- `M=1`: `U=8.0` -> experts = 1,847 MB
- `M=4` (k=3 drafts): `U≈19.0` -> experts = 4,387 MB; target verify = `1,994 + 4,387 = 6,381 MB` (22.8 ms @ 280.3 GB/s)
- `M=5` (k=4 drafts): `U≈21.5` -> experts = 4,965 MB; target verify = `1,994 + 4,965 = 6,959 MB` (24.8 ms @ 280.3 GB/s)
- `M=8` (k=7 drafts): `U≈28.0` -> experts = 6,466 MB; target verify = `1,994 + 6,466 = 8,460 MB` (30.2 ms @ 280.3 GB/s)

**DSpark Draft Cost Analysis (`dspark-Inkling-2-Mini-Q8_0.gguf`):**
- **Current unmodified draft (k=3):**
  - 3 draft layers (`Q8_0`): `278 MB`
  - `target->output` (`Q8_0`, ALL 8 block rows × 201K vocab!): `492 MB`
  - `markov_w2` (`BF16`, 200K vocab × 3 steps): `307 MB`
  - 6-pass 200K-vocab argmax + 2.4 MB CPU logit copy + CPU argmax
  - **Current draft total:** `~1,077 MB` + overhead = **~8–12 ms per step**!
- **Optimized draft (shortlisted + quantized/fused GPU argmax, 100% lossless target verification):**
  - 3 draft layers (`Q8_0`): `278 MB` (or `Q4_0`: `150 MB`, still exact target!)
  - Draft shortlist (e.g., 32K–49K vocab rows): `target->output` drops `492 -> 80–120 MB`, `markov_w2` drops `307 -> 15–25 MB` (Q8_0/Q4_0)
  - Fused GPU argmax returns ONLY `k` int32 IDs (zero 200K logit copy to CPU!)
  - **Optimized draft total:** `~380 MB` (~1.5 ms) — saves **~7–10 ms EVERY step**!

**Combined Short-Context & 256K Decode Ceilings (`BW = 280.3 GB/s`, `AD-Q4_K`):**

| Mode | Bytes / step (short) | Floor ms (short) | Ceiling (short, `a=2.5 / 3.2 / 4.5`) | Floor ms @ 256K (Q4_0 KV) | Ceiling @ 256K (`a=2.5 / 3.2`) |
|---|---:|---:|---|---:|---|
| Plain `M=1` (no spec) | 3.84 GB | 13.7 ms | **73.0 tok/s** | 22.3 ms | **44.8 tok/s** |
| DSpark `k=3` (current unpruned draft) | 7.46 GB | 26.6 ms | 94 / 120 tok/s | 35.3 ms | 71 / 91 tok/s |
| DSpark `k=3` (shortlist + fused draft) | 6.76 GB | 24.1 ms | **104 / 133 tok/s** | 32.8 ms | **76 / 98 tok/s** |
| DSpark `k=4` (shortlist + fused draft) | 7.34 GB | 26.2 ms | **122 / 153 tok/s** (`a=3.2/4.0`) | 34.8 ms | **92 / 115 tok/s** |
| DSpark `k=7` (shortlist + fused draft) | 8.84 GB | 31.5 ms | **143 / 222 tok/s** (`a=4.5/7.0`) | 40.2 ms | **112 / 174 tok/s** |
| Prompt Lookup (`M=16..32` on copy/code) | ~10–12 GB | ~40 ms | **300–500+ tok/s** | ~50 ms | **250–400+ tok/s** |

## 5. Prefill Ceilings (Compute Bound, 50% MFU = 14.57 TFLOPS)

Active linear FLOPs per token (48 visits):
- Attention projections (`Q, K, V, O, Gate, Rel`): `48 × (3072 + 1024 + 1024 + 3072 + 3072 + 384) × 2304 × 2 = 2.57 GFLOP/tok`
- Routed MoE (`top-8`, `gate_up + down`): `48 × 8 × 3 × 2304 × 1152 × 2 = 6.11 GFLOP/tok`
- Loop controllers + router: `~0.14 GFLOP/tok`
- **Total linear FLOPs/tok:** **8.82 GFLOP/tok**
- Full-attention quadratic FLOPs/tok (8 global visits, 24 heads × 128 dim, causal avg pos `L/2`):
  `4 × 8 × 24 × 128 × (L/2) = 49,152 × L` FLOPs/tok
  - At `L = 4,096`: `0.20 GFLOP/tok` -> total `9.02 GFLOP/tok` -> **1,615 tok/s practical ceiling** (3,230 peak)
  - At `L = 32,768`: `1.61 GFLOP/tok` -> total `10.43 GFLOP/tok` -> **1,397 tok/s practical ceiling**
  - At `L = 131,072`: `6.44 GFLOP/tok` -> total `15.26 GFLOP/tok` -> **955 tok/s practical ceiling**
  - At `L = 262,144`: `12.88 GFLOP/tok` -> total `21.70 GFLOP/tok` -> **671 tok/s practical ceiling**

Notice how **HUGE** the prefill headroom is! Current baseline was only ~345 tok/s at 6.5K tokens (only **21%** of practical ceiling!).
Why? Because:
1. Unfused Inkling ops (`clip`, `conv_residual`, `control`, `situ`, `rel_mask`) + materialized `kq_b`
2. DSpark `n_ubatch = 32` forcing 8,192 tiny CPU-copied injections at 256K
3. Non-optimal `ubatch` / `Q4_0` KV dequantization overhead vs `F16` / `Q8_0` KV and Metal 4 `fa_tensor`!
