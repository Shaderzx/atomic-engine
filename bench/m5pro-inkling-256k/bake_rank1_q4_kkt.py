#!/usr/bin/env python3
"""Projection-matched KKT-optimal Q4_0 rank-1 LoRA baker for Inkling-2-Mini.

Bakes a rank-1 F32 abliteration LoRA (e.g. AtomicChat Inkling-2-Mini-Uncensored-LoRA.gguf)
directly into a Q4_0 GGUF base model (e.g. Inkling-2-Mini-M5-Q4_0Pure.gguf) while:
  1. Preserving ALL untouched tensors byte-for-byte via APFS CoW clone (`cp -c`).
  2. Preserving exact Q4_0 tensor types, shapes, GGUF metadata, and byte offsets (zero size growth).
  3. Solving the exact KKT Lagrange problem per input column / token row with FIXED Q4_0 block scales:
       min ||W_q - (W_0 + dW)||_2^2   subject to   r^T W_q = r^T (W_0 + dW)
     achieving ~100.0% retention of the rank-1 projection while modifying only ~1-2% of nibbles
     (vs naive Q4_0 requantization which erases ~78% of the abliteration direction!).
"""
import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0, "/Users/shadmehr/code/atomic-engine/gguf-py")
from gguf import GGUFReader, GGMLQuantizationType, dequantize


def unpack_q4_0(raw_bytes: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    k = shape[-1]
    assert k % 32 == 0
    nb = k // 32
    lead = shape[:-1]
    blk = raw_bytes.reshape(*lead, nb, 18)
    d = np.frombuffer(np.ascontiguousarray(blk[..., :2]).tobytes(), dtype=np.float16).reshape(*lead, nb).astype(np.float32)
    qs = blk[..., 2:]
    lo = (qs & 0x0F).astype(np.int8) - 8
    hi = (qs >> 4).astype(np.int8) - 8
    q = np.concatenate([lo, hi], axis=-1).reshape(*lead, k)
    return d, q


def pack_q4_0(d: np.ndarray, q: np.ndarray) -> np.ndarray:
    lead = q.shape[:-1]
    k = q.shape[-1]
    nb = k // 32
    q_u = np.clip(q, -8, 7).astype(np.int16) + 8
    q_blk = q_u.reshape(*lead, nb, 32)
    qs = ((q_blk[..., :16] & 0x0F) | ((q_blk[..., 16:] & 0x0F) << 4)).astype(np.uint8)
    d_bytes = np.ascontiguousarray(d.astype(np.float16)).view(np.uint8).reshape(*lead, nb, 2)
    return np.ascontiguousarray(np.concatenate([d_bytes, qs], axis=-1).reshape(*lead, nb * 18))


def bake_matrix_col_r(raw_bytes: np.ndarray, shape: tuple[int, int], a: np.ndarray, b: np.ndarray, scale: float, iters: int = 18):
    """W shape (N, K), a shape (1, K), b shape (N, 1), dW = scale * (b @ a), unit r along N."""
    n, k = shape
    d, q0 = unpack_q4_0(raw_bytes, shape)
    d_exp = np.repeat(d, 32, axis=-1)
    inv_d = np.where(d_exp != 0.0, 1.0 / d_exp, 0.0)
    w0 = q0.astype(np.float32) * d_exp
    dw = (scale * b) @ a  # (N, K)
    bn = float(np.linalg.norm(b))
    if bn < 1e-12:
        return raw_bytes.copy(), 1.0, 0.0
    r = (b[:, 0] / bn).astype(np.float32)
    w_exact = w0 + dw
    target_r = r @ w_exact  # (K,)
    u_exact = q0.astype(np.float32) + dw * inv_d
    r_over_d = r[:, None] * inv_d
    rd = r[:, None] * d_exp

    # Bracket wide enough for any block scale
    max_step = float(np.max(np.abs(d))) * 6.0 + 0.05
    lo = np.full(k, -max_step, dtype=np.float32)
    hi = np.full(k,  max_step, dtype=np.float32)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        q_try = np.clip(np.rint(u_exact + r_over_d * mid[None, :]), -8, 7)
        cur_r = np.sum(q_try * rd, axis=0)
        too_low = cur_r < target_r
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    mid = 0.5 * (lo + hi)
    q_opt = np.clip(np.rint(u_exact + r_over_d * mid[None, :]), -8, 7).astype(np.int8)
    packed = pack_q4_0(d, q_opt)
    wq = q_opt.astype(np.float32) * d_exp
    d_act = r @ (wq - w0)
    d_tgt = r @ dw
    ret = float(np.dot(d_act, d_tgt) / (np.dot(d_tgt, d_tgt) + 1e-12))
    chg = float(np.mean(q_opt != q0))
    return packed, ret, chg


def bake_embd_row_r(raw_bytes: np.ndarray, shape: tuple[int, int], a: np.ndarray, b: np.ndarray, scale: float, iters: int = 18):
    """token_embd: W shape (V, E), a shape (V, 1), b shape (E, 1), dW = scale * (a @ b.T), unit r along E."""
    v, e = shape
    bn = float(np.linalg.norm(b))
    if bn < 1e-12:
        return raw_bytes.copy(), 1.0, 0.0
    r = (b[:, 0] / bn).astype(np.float32)  # (E,)
    row_bytes = (e // 32) * 18
    out = np.empty((v, row_bytes), dtype=np.uint8)
    raw_2d = raw_bytes.reshape(v, row_bytes)
    chunk = 8192
    num_ret = 0.0
    den_ret = 0.0
    tot_chg = 0
    for s in range(0, v, chunk):
        t = min(v, s + chunk)
        cv = t - s
        d, q0 = unpack_q4_0(raw_2d[s:t], (cv, e))
        d_exp = np.repeat(d, 32, axis=-1)
        inv_d = np.where(d_exp != 0.0, 1.0 / d_exp, 0.0)
        w0 = q0.astype(np.float32) * d_exp
        dw = (scale * a[s:t, 0:1]) * b[:, 0][None, :]  # (cv, E)
        w_exact = w0 + dw
        target_r = w_exact @ r  # (cv,)
        u_exact = q0.astype(np.float32) + dw * inv_d
        r_over_d = r[None, :] * inv_d
        rd = r[None, :] * d_exp
        max_step = float(np.max(np.abs(d))) * 6.0 + 0.05
        lo = np.full(cv, -max_step, dtype=np.float32)
        hi = np.full(cv,  max_step, dtype=np.float32)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            q_try = np.clip(np.rint(u_exact + r_over_d * mid[:, None]), -8, 7)
            cur_r = np.sum(q_try * rd, axis=1)
            too_low = cur_r < target_r
            lo = np.where(too_low, mid, lo)
            hi = np.where(too_low, hi, mid)
        mid = 0.5 * (lo + hi)
        q_opt = np.clip(np.rint(u_exact + r_over_d * mid[:, None]), -8, 7).astype(np.int8)
        out[s:t] = pack_q4_0(d, q_opt)
        wq = q_opt.astype(np.float32) * d_exp
        d_act = (wq - w0) @ r
        d_tgt = dw @ r
        num_ret += float(np.dot(d_act, d_tgt))
        den_ret += float(np.dot(d_tgt, d_tgt))
        tot_chg += int(np.sum(q_opt != q0))
    return out, num_ret / (den_ret + 1e-12), tot_chg / (v * e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=14)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    t0 = time.monotonic()
    if os.path.exists(args.out):
        os.remove(args.out)
    # Fast APFS CoW clone so untouched 655 tensors share disk blocks and copy in ~10ms
    cp = subprocess.run(["cp", "-c", args.base, args.out], capture_output=True, text=True)
    if cp.returncode != 0:
        shutil.copy2(args.base, args.out)
    print(f"Cloned base -> {args.out} in {time.monotonic() - t0:.2f}s", flush=True)

    lora = GGUFReader(args.lora)
    base = GGUFReader(args.base)
    alpha_f = lora.fields.get("adapter.lora.alpha") or lora.fields.get("adapter.lora_alpha")
    alpha = float(alpha_f.contents()) if alpha_f is not None else 0.0
    lmap = {t.name: t for t in lora.tensors}
    bmap = {t.name: t for t in base.tensors}

    # Find base tensor names covered by LoRA
    bases_to_bake = sorted({t.name[:-7] for t in lora.tensors if t.name.endswith(".lora_a")})
    print(f"Baking {len(bases_to_bake)} tensors (alpha={alpha}, scale={args.scale}, iters={args.iters})", flush=True)

    rets = []
    chgs = []
    with open(args.out, "r+b") as fout:
        for idx, name in enumerate(bases_to_bake):
            bt = bmap[name]
            assert bt.tensor_type == GGMLQuantizationType.Q4_0, f"{name} is {bt.tensor_type}, expected Q4_0"
            ta = lmap[name + ".lora_a"]
            tb = lmap[name + ".lora_b"]
            a_np = ta.data.astype(np.float32)
            b_np = tb.data.astype(np.float32)
            # rank = ggml ne[0] of lora_b (which is last dim in numpy)
            rank = int(tb.shape[0])
            eff_scale = (args.scale * alpha / rank) if alpha > 0.0 else args.scale
            assert rank == 1, f"{name} has rank={rank}, expected rank-1"

            raw = bt.data.view(np.uint8)
            ts = time.monotonic()
            if name == "token_embd.weight":
                e_dim, v_dim = int(bt.shape[0]), int(bt.shape[1])
                packed, ret, chg = bake_embd_row_r(raw, (v_dim, e_dim), a_np, b_np, eff_scale, iters=args.iters)
            elif len(bt.shape) == 2:
                k_dim, n_dim = int(bt.shape[0]), int(bt.shape[1])
                packed, ret, chg = bake_matrix_col_r(raw, (n_dim, k_dim), a_np, b_np, eff_scale, iters=args.iters)
            elif len(bt.shape) == 3:
                k_dim, n_dim, n_exp = int(bt.shape[0]), int(bt.shape[1]), int(bt.shape[2])
                exp_bytes = n_dim * (k_dim // 32) * 18
                raw_3d = raw.reshape(n_exp, exp_bytes)
                packed = np.empty_like(raw_3d)
                def _do_exp(e):
                    pe, er, ec = bake_matrix_col_r(raw_3d[e], (n_dim, k_dim), a_np[e], b_np[e], eff_scale, iters=args.iters)
                    return e, pe.reshape(-1), er, ec
                e_rets = [0.0] * n_exp
                e_chgs = [0.0] * n_exp
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                    for e, pe, er, ec in ex.map(_do_exp, range(n_exp)):
                        packed[e] = pe
                        e_rets[e] = er
                        e_chgs[e] = ec
                ret = float(np.mean(e_rets))
                chg = float(np.mean(e_chgs))
            else:
                raise ValueError(f"Unsupported shape {bt.shape} for {name}")

            pbytes = packed.tobytes()
            assert len(pbytes) == bt.n_bytes, f"size mismatch {len(pbytes)} != {bt.n_bytes}"
            fout.seek(bt.data_offset)
            fout.write(pbytes)
            rets.append(ret)
            chgs.append(chg)
            print(f"  [{idx+1:2d}/{len(bases_to_bake)}] {name:34s} ret={ret:7.2%} chg={chg:5.2%} ({time.monotonic()-ts:.2f}s)", flush=True)

    print(f"DONE in {time.monotonic()-t0:.1f}s | mean_retention={np.mean(rets):.2%} (min={np.min(rets):.2%}) | mean_nibbles_changed={np.mean(chgs):.2%}", flush=True)


if __name__ == "__main__":
    main()
