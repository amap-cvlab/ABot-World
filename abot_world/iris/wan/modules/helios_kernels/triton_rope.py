import os
import atexit
import time
import torch
import triton
import triton.language as tl

from .utils import calculate_settings, torch_gpu_device

# ------------------------------- RoPE benchmark -------------------------------
# Set ROPE_BENCHMARK=1 to log call_count and total_ms to rope_benchmark.log.
# Set ROPE_USE_ORIG=1 with ROPE_BENCHMARK=1 to benchmark original implementation.
# Otherwise benchmarks Flash RoPE. Stats printed at process exit.
_ROPE_BENCHMARK = os.environ.get("ROPE_BENCHMARK", "0") == "1"
_ROPE_USE_ORIG = os.environ.get("ROPE_USE_ORIG", "0") == "1"
_rope_bench_state = {"call_count": 0, "total_ms": 0.0, "impl": None}


def _rope_bench_log_path():
    root = os.environ.get("PROJECT_ROOT", os.getcwd())
    return os.path.join(root, "rope_benchmark.log")


def _flush_rope_benchmark_log():
    if _rope_bench_state["impl"] is None or _rope_bench_state["call_count"] == 0:
        return
    p = _rope_bench_log_path()
    line = (
        f"[RoPE] impl={_rope_bench_state['impl']} "
        f"call_count={_rope_bench_state['call_count']} "
        f"total_ms={_rope_bench_state['total_ms']:.2f} "
        f"avg_ms={_rope_bench_state['total_ms']/max(1,_rope_bench_state['call_count']):.4f}\n"
    )
    with open(p, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"[RoPE benchmark] {line.strip()}")


def _make_timed_rope_apply(fn, name):
    def timed_fn(x, grid_sizes, freqs):
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(x, grid_sizes, freqs)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _rope_bench_state["call_count"] += 1
        _rope_bench_state["total_ms"] += elapsed_ms
        _rope_bench_state["impl"] = name
        return out
    return timed_fn


def _make_timed_causal_rope(fn, name):
    def timed_fn(x, grid_sizes, freqs, start_frame=0):
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(x, grid_sizes, freqs, start_frame=start_frame)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _rope_bench_state["call_count"] += 1
        _rope_bench_state["total_ms"] += elapsed_ms
        _rope_bench_state["impl"] = name
        return out
    return timed_fn


def _make_timed_any(fn, name):
    """Timed wrapper for arbitrary-signature RoPE fns (relative / refimg)."""
    def timed_fn(*args, **kwargs):
        x = args[0] if args else None
        do_sync = torch.is_tensor(x) and x.device.type == "cuda"
        if do_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        if do_sync:
            torch.cuda.synchronize()
        _rope_bench_state["call_count"] += 1
        _rope_bench_state["total_ms"] += (time.perf_counter() - t0) * 1000
        _rope_bench_state["impl"] = name
        return out
    return timed_fn


# ------------------------------- replace funtion -------------------------------


def apply_rotary_emb_transposed_flash(x, freqs_cis):
    return Flash_RoPE_Transposed.apply(x, freqs_cis)


def _wan_freqs_to_freqs_cis(freqs_split, grid_sizes, start_frame=0):
    """
    Convert Wan rope_params freqs (after split) + grid_sizes to freqs_cis for Flash kernel.
    freqs_split: tuple of 3 tensors (freqs_t, freqs_h, freqs_w), each complex (polar)
    Returns: freqs_cis [B, seq_len, head_dim*2] in Flash interleaved layout:
      cos at positions 0,2,4,...,126 (first 128), sin at positions 129,131,...,255 (second 128)
    """
    freqs_t, freqs_h, freqs_w = freqs_split
    output_list = []
    c = freqs_t.shape[1] + freqs_h.shape[1] + freqs_w.shape[1]
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        f, h, w = int(f), int(h), int(w)
        seq_len = f * h * w
        freqs_i = torch.cat([
            freqs_t[start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, -1)
        cos = freqs_i.real.float()  # [seq_len, c]
        sin = freqs_i.imag.float()  # [seq_len, c]
        # Flash kernel expects freqs_cis [B, seq, head_dim*2] with cos at even indices
        # of first half, sin at odd indices of second half (head_dim = 2*c)
        head_dim = 2 * c
        freqs_cis_i = torch.zeros(seq_len, head_dim * 2, dtype=cos.dtype, device=cos.device)
        freqs_cis_i[:, 0:head_dim:2] = cos
        freqs_cis_i[:, head_dim + 1:head_dim * 2:2] = sin
        output_list.append(freqs_cis_i)
    return torch.stack(output_list)


def flash_rope_apply_wan(x, grid_sizes, freqs):
    """
    Flash RoPE for Wan's rope_apply interface.
    x: [B, L, n_heads, head_dim], grid_sizes: [B, 3], freqs: from rope_params (single tensor)
    """
    n_heads = x.shape[2]
    head_dim = x.shape[3]
    c = head_dim // 2
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    B = x.shape[0]
    output_list = []
    for i in range(B):
        gs = grid_sizes[i : i + 1]
        f, h, w = int(gs[0, 0].item()), int(gs[0, 1].item()), int(gs[0, 2].item())
        seq_len = f * h * w

        x_i = x[i : i + 1, :seq_len].contiguous()
        freqs_cis_i = _wan_freqs_to_freqs_cis(freqs_split, gs, start_frame=0)
        freqs_cis_i = freqs_cis_i.to(device=x.device, dtype=x.dtype)

        out_i = Flash_RoPE_Transposed.apply(x_i, freqs_cis_i)

        if seq_len < x.shape[1]:
            out_i = torch.cat([out_i[0], x[i, seq_len:]], dim=0)
        else:
            out_i = out_i[0]
        output_list.append(out_i)

    return torch.stack(output_list).type_as(x)


def flash_causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """
    Flash RoPE for Wan's causal_rope_apply interface.
    """
    head_dim = x.shape[3]
    c = head_dim // 2
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    B = x.shape[0]
    output_list = []
    for i in range(B):
        gs = grid_sizes[i : i + 1]
        f, h, w = int(gs[0, 0].item()), int(gs[0, 1].item()), int(gs[0, 2].item())
        seq_len = f * h * w

        x_i = x[i : i + 1, :seq_len].contiguous()
        freqs_cis_i = _wan_freqs_to_freqs_cis(freqs_split, gs, start_frame=start_frame)
        freqs_cis_i = freqs_cis_i.to(device=x.device, dtype=x.dtype)

        out_i = Flash_RoPE_Transposed.apply(x_i, freqs_cis_i)

        if seq_len < x.shape[1]:
            out_i = torch.cat([out_i[0], x[i, seq_len:]], dim=0)
        else:
            out_i = out_i[0]
        output_list.append(out_i)

    return torch.stack(output_list).type_as(x)


# ---------------- Relative / ref-image Flash RoPE (causal_model) --------------
# Both variants are constant per (grid, ids, dim, device, dtype) in steady state
# (fixed sliding window + block-relative frame ids), so the kernel-ready
# freqs_cis tensor is cached and reused instead of rebuilt every block.
_FLASH_REL_FREQS_CIS_CACHE: dict = {}
_FLASH_REFIMG_FREQS_CIS_CACHE: dict = {}


def _build_relative_freqs_cis(freqs_split, f, h, w, t_index):
    """Build freqs_cis [1, seq_len, head_dim*2] from clamped relative frame ids."""
    freqs_t, freqs_h, freqs_w = freqs_split
    c = freqs_t.shape[1] + freqs_h.shape[1] + freqs_w.shape[1]
    seq_len = f * h * w
    freqs_i = torch.cat([
        freqs_t[t_index].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(seq_len, -1)
    cos = freqs_i.real.float()  # [seq_len, c]
    sin = freqs_i.imag.float()
    head_dim = 2 * c
    freqs_cis = torch.zeros(seq_len, head_dim * 2, dtype=cos.dtype, device=cos.device)
    freqs_cis[:, 0:head_dim:2] = cos
    freqs_cis[:, head_dim + 1:head_dim * 2:2] = sin
    return freqs_cis.unsqueeze(0)  # [1, seq_len, head_dim*2]


def _get_relative_freqs_cis(freqs_split, f, h, w, frame_indices, device, dtype):
    if frame_indices is None:
        t_index = torch.arange(f, device=device, dtype=torch.long)
    else:
        t_index = frame_indices[:f].to(device=device, dtype=torch.long)
    # Block-relative RoPE keeps temporal ids inside the visible local window
    # (local_attn_size=21 -> max valid id 20), matching relative_rope_apply.
    t_index = torch.clamp(t_index, min=0, max=20)
    c = freqs_split[0].shape[1] + freqs_split[1].shape[1] + freqs_split[2].shape[1]
    key = (int(f), int(h), int(w), tuple(t_index.tolist()), int(c), str(device), str(dtype))
    cached = _FLASH_REL_FREQS_CIS_CACHE.get(key)
    if cached is not None:
        return cached
    freqs_cis = _build_relative_freqs_cis(freqs_split, f, h, w, t_index).to(
        device=device, dtype=dtype
    )
    _FLASH_REL_FREQS_CIS_CACHE[key] = freqs_cis
    return freqs_cis


def flash_relative_rope_apply(x, grid_sizes, freqs, frame_indices=None):
    """Flash RoPE for causal_model.relative_rope_apply (block-relative frame ids).

    x: [B, L, n_heads, head_dim]; grid_sizes: [B, 3]; freqs: rope_params tensor.
    """
    head_dim = x.shape[3]
    c = head_dim // 2
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    B = x.shape[0]
    output_list = []
    for i in range(B):
        gs = grid_sizes[i : i + 1]
        f, h, w = int(gs[0, 0].item()), int(gs[0, 1].item()), int(gs[0, 2].item())
        seq_len = f * h * w

        x_i = x[i : i + 1, :seq_len].contiguous()
        freqs_cis_i = _get_relative_freqs_cis(
            freqs_split, f, h, w, frame_indices, x.device, x.dtype
        )
        out_i = Flash_RoPE_Transposed.apply(x_i, freqs_cis_i)

        if seq_len < x.shape[1]:
            out_i = torch.cat([out_i[0], x[i, seq_len:]], dim=0)
        else:
            out_i = out_i[0]
        output_list.append(out_i)

    return torch.stack(output_list).type_as(x)


def _get_refimg_freqs_cis(freqs, head_dim, device, dtype):
    """Convert precomputed complex ref freqs [S, 1, c] to freqs_cis [1, S, head_dim*2].

    ref freqs are already cached upstream (_build_ref_freqs), so the tensor
    identity is stable and data_ptr is a valid cache key.
    """
    key = (freqs.data_ptr(), int(freqs.shape[0]), int(head_dim), str(device), str(dtype))
    cached = _FLASH_REFIMG_FREQS_CIS_CACHE.get(key)
    if cached is not None:
        return cached
    fc = freqs.reshape(freqs.shape[0], -1)  # [S, c]
    c = fc.shape[1]
    assert 2 * c == head_dim, f"ref freqs dim {c} incompatible with head_dim {head_dim}"
    cos = fc.real.float()  # [S, c]
    sin = fc.imag.float()
    S = fc.shape[0]
    freqs_cis = torch.zeros(S, head_dim * 2, dtype=torch.float32, device=cos.device)
    freqs_cis[:, 0:head_dim:2] = cos
    freqs_cis[:, head_dim + 1:head_dim * 2:2] = sin
    freqs_cis = freqs_cis.unsqueeze(0).to(device=device, dtype=dtype)  # [1, S, head_dim*2]
    _FLASH_REFIMG_FREQS_CIS_CACHE[key] = freqs_cis
    return freqs_cis


def flash_rope_apply_with_refimg(x, freqs, num_heads):
    """Flash RoPE for causal_model.rope_apply_with_refimg (ref-image tokens).

    x: [B, S, D] or [B, S, n_heads, head_dim]; freqs: precomputed complex [S, 1, c].
    """
    if x.dim() == 3:
        b, s, _ = x.shape
        x = x.view(b, s, num_heads, -1)
    B, S, n_heads, head_dim = x.shape
    freqs_cis = _get_refimg_freqs_cis(freqs, head_dim, x.device, x.dtype)  # [1, S, head_dim*2]
    if B > 1:
        freqs_cis = freqs_cis.expand(B, -1, -1)
    out = Flash_RoPE_Transposed.apply(x.contiguous(), freqs_cis)
    return out.type_as(x)


def replace_rope_with_flash_rope():
    from .. import model as wan_model
    from .. import causal_model as wan_causal_model

    _rope_orig = wan_model.rope_apply
    _causal_orig = wan_causal_model.causal_rope_apply
    _relative_orig = wan_causal_model.relative_rope_apply
    _refimg_orig = wan_causal_model.rope_apply_with_refimg

    if _ROPE_BENCHMARK:
        atexit.register(_flush_rope_benchmark_log)
        if _ROPE_USE_ORIG:
            wan_model.rope_apply = _make_timed_rope_apply(_rope_orig, "orig")
            wan_causal_model.rope_apply = _make_timed_rope_apply(_rope_orig, "orig")
            wan_causal_model.causal_rope_apply = _make_timed_causal_rope(_causal_orig, "orig")
            wan_causal_model.relative_rope_apply = _make_timed_any(_relative_orig, "orig")
            wan_causal_model.rope_apply_with_refimg = _make_timed_any(_refimg_orig, "orig")
            print("Patched RoPE (orig with benchmark) - ROPE_USE_ORIG=1")
        else:
            wan_model.rope_apply = _make_timed_rope_apply(flash_rope_apply_wan, "flash")
            wan_causal_model.rope_apply = _make_timed_rope_apply(flash_rope_apply_wan, "flash")
            wan_causal_model.causal_rope_apply = _make_timed_causal_rope(flash_causal_rope_apply, "flash")
            wan_causal_model.relative_rope_apply = _make_timed_any(_relative_orig, "orig")
            wan_causal_model.rope_apply_with_refimg = _make_timed_any(_refimg_orig, "orig")
            print("Patched Flash_RoPE (with benchmark) - log to rope_benchmark.log")
    else:
        wan_model.rope_apply = flash_rope_apply_wan
        wan_causal_model.rope_apply = flash_rope_apply_wan
        wan_causal_model.causal_rope_apply = flash_causal_rope_apply
        wan_causal_model.relative_rope_apply = _relative_orig
        wan_causal_model.rope_apply_with_refimg = _refimg_orig
        print("Patched Flash_RoPE (rope_apply, causal_rope_apply); retained native relative/reference rotations")


# ------------------------------- layer norm -------------------------------


@triton.jit
def _apply_rope_transposed_kernel(
    X,
    Out,
    cos,
    sin,
    n_heads: tl.constexpr,
    stride_x: tl.constexpr,
    stride_out: tl.constexpr,
    stride_freq: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    freq_row_idx = row_idx // n_heads

    half_head_dim = head_dim // 2
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < half_head_dim

    x_ptr = X + row_idx * stride_x
    out_ptr = Out + row_idx * stride_out
    cos_ptr = cos + freq_row_idx * stride_freq
    sin_ptr = sin + freq_row_idx * stride_freq

    x_real = tl.load(x_ptr + col_offsets * 2, mask=mask, other=0.0)
    x_imag = tl.load(x_ptr + col_offsets * 2 + 1, mask=mask, other=0.0)
    cos_even = tl.load(cos_ptr + col_offsets * 2, mask=mask, other=0.0)
    sin_odd = tl.load(sin_ptr + col_offsets * 2 + 1, mask=mask, other=0.0)

    out_even = x_real * cos_even - x_imag * sin_odd
    out_odd = x_real * sin_odd + x_imag * cos_even

    tl.store(out_ptr + col_offsets * 2, out_even, mask=mask)
    tl.store(out_ptr + col_offsets * 2 + 1, out_odd, mask=mask)


class Flash_RoPE_Transposed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, freqs_cis):
        # x: [B, seq_len, n_heads, head_dim]
        # freqs_cis: [B, seq_len, head_dim*2]

        B, seq_len, n_heads, head_dim = x.shape

        x_flat = x.reshape(-1, head_dim).contiguous()
        device = x_flat.device
        out = torch.empty_like(x_flat)

        freqs_flat = freqs_cis.reshape(B * seq_len, -1).contiguous()
        half_dim = freqs_flat.shape[-1] // 2
        cos = freqs_flat[:, :half_dim].contiguous()  # [B*seq_len, head_dim]
        sin = freqs_flat[:, half_dim:].contiguous()  # [B*seq_len, head_dim]

        n_rows = x_flat.shape[0]  # B*seq_len*n_heads
        BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2)

        with torch_gpu_device(device):
            _apply_rope_transposed_kernel[(n_rows,)](
                x_flat,
                out,
                cos,
                sin,
                n_heads,
                x_flat.stride(0),
                out.stride(0),
                cos.stride(0),
                head_dim,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        out = out.reshape(B, seq_len, n_heads, head_dim)

        ctx.save_for_backward(cos, sin)
        ctx.n_heads = n_heads
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.head_dim = head_dim

        return out

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors

        B, seq_len, n_heads, head_dim = grad_output.shape
        grad_flat = grad_output.reshape(-1, head_dim).contiguous()
        device = grad_flat.device
        grad_x = torch.empty_like(grad_flat)

        sin_neg = -sin

        n_rows = grad_flat.shape[0]

        with torch_gpu_device(device):
            _apply_rope_transposed_kernel[(n_rows,)](
                grad_flat,
                grad_x,
                cos,
                sin_neg,
                ctx.n_heads,
                grad_flat.stride(0),
                grad_x.stride(0),
                cos.stride(0),
                ctx.head_dim,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=ctx.num_warps,
            )

        grad_x = grad_x.reshape(B, seq_len, n_heads, head_dim)
        return grad_x, None


# ------------------------------- For test -------------------------------
def test_zero_error():
    def apply_rotary_emb_transposed_orig(x, freqs_cis):
        cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
        x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)
        out = torch.empty_like(x)
        out[..., 0::2] = x_real * cos[..., 0::2] - x_imag * sin[..., 1::2]
        out[..., 1::2] = x_real * sin[..., 1::2] + x_imag * cos[..., 0::2]
        return out

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = torch.randn(1, 128, 12, 128, device="cuda", dtype=dtype)
        freqs_cis = torch.randn(1, 128, 256, device="cuda", dtype=dtype)

        out_orig = apply_rotary_emb_transposed_orig(x, freqs_cis)
        out_fast = apply_rotary_emb_transposed_flash(x, freqs_cis)

        diff = (out_orig - out_fast).abs().max()

        eps = torch.finfo(dtype).eps
        print(f"{dtype}: max_diff={diff.item():.2e}, machine_eps={eps:.2e}")

        if diff < eps * 100:
            print(f"  ✅ Essentially zero error for {dtype}")
        else:
            print(f"  ⚠️ Significant error: {diff / eps:.1f}x machine epsilon")


def test_comparison():
    def apply_rotary_emb_transposed_orig(x, freqs_cis):
        cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
        x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)
        out = torch.empty_like(x)
        out[..., 0::2] = x_real * cos[..., 0::2] - x_imag * sin[..., 1::2]
        out[..., 1::2] = x_real * sin[..., 1::2] + x_imag * cos[..., 0::2]
        return out

    x = torch.randn(1, 14040, 12, 128, device="cuda", dtype=torch.float32)
    freqs_cis = torch.randn(1, 14040, 256, device="cuda", dtype=torch.float32)

    out_orig = apply_rotary_emb_transposed_orig(x, freqs_cis)
    out_fast = apply_rotary_emb_transposed_flash(x, freqs_cis)

    diff = (out_orig - out_fast).abs().max()
    print(f"Max difference: {diff.item():.6e}")

    if diff < 1e-5:
        print("✅ Test passed!")
        print(f"Input shapes: x={x.shape}, freqs_cis={freqs_cis.shape}")
        print(f"Output shape: {out_fast.shape}")
    else:
        print(f"❌ Test failed! Max diff: {diff.item()}")


def test_backward_comparison():
    def apply_rotary_emb_transposed_orig(x, freqs_cis):
        cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
        x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)
        out = torch.empty_like(x)
        out[..., 0::2] = x_real * cos[..., 0::2] - x_imag * sin[..., 1::2]
        out[..., 1::2] = x_real * sin[..., 1::2] + x_imag * cos[..., 0::2]
        return out

    x1 = torch.randn(1, 128, 12, 128, device="cuda", requires_grad=True)
    x2 = x1.clone().detach().requires_grad_(True)
    freqs_cis = torch.randn(1, 128, 256, device="cuda")

    out_orig = apply_rotary_emb_transposed_orig(x1, freqs_cis)
    out_fast = apply_rotary_emb_transposed_flash(x2, freqs_cis)

    grad_output = torch.randn_like(out_orig)

    out_orig.backward(grad_output)
    out_fast.backward(grad_output)

    grad_diff = (x1.grad - x2.grad).abs()
    max_diff = grad_diff.max().item()
    mean_diff = grad_diff.mean().item()

    print("Gradient comparison:")
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-5:
        print("✅ Backward gradients match!")
    else:
        print(f"⚠️ Gradients differ by {max_diff:.6e}")
        max_idx = grad_diff.argmax()
        print(f"  Max diff location: {torch.unravel_index(max_idx, grad_diff.shape)}")
        print(f"  Original grad: {x1.grad.flatten()[max_idx]:.6f}")
        print(f"  Fast grad: {x2.grad.flatten()[max_idx]:.6f}")


def test_backward():
    from torch.autograd import gradcheck

    B, seq_len, n_heads, head_dim = 2, 16, 4, 32
    x = torch.randn(B, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float64, requires_grad=True)
    freqs_cis = torch.randn(B, seq_len, head_dim * 2, device="cuda", dtype=torch.float64)

    test = gradcheck(
        Flash_RoPE_Transposed.apply,
        (x, freqs_cis),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )

    if test:
        print("✅ Backward pass is correct (gradcheck passed)")
    else:
        print("❌ Backward pass has errors")


def test_in_training_loop_comparison():
    def apply_rotary_emb_transposed_orig(x, freqs_cis):
        cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
        x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)
        out = torch.empty_like(x)
        out[..., 0::2] = x_real * cos[..., 0::2] - x_imag * sin[..., 1::2]
        out[..., 1::2] = x_real * sin[..., 1::2] + x_imag * cos[..., 0::2]
        return out

    class SimpleModel(torch.nn.Module):
        def __init__(self, use_fast=False):
            super().__init__()
            self.linear = torch.nn.Linear(128, 128, device="cuda")
            self.use_fast = use_fast

        def forward(self, x, freqs_cis):
            x = self.linear(x)
            if self.use_fast:
                x = apply_rotary_emb_transposed_flash(x, freqs_cis)
            else:
                x = apply_rotary_emb_transposed_orig(x, freqs_cis)
            return x.mean()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model_orig = SimpleModel(use_fast=False)
    model_fast = SimpleModel(use_fast=True)

    model_fast.load_state_dict(model_orig.state_dict())

    optimizer_orig = torch.optim.Adam(model_orig.parameters(), lr=1e-3)
    optimizer_fast = torch.optim.Adam(model_fast.parameters(), lr=1e-3)

    losses_orig = []
    losses_fast = []

    print("=" * 80)
    print("Training comparison: Original vs Optimized RoPE")
    print("=" * 80)
    print(f"{'Step':<6} {'Original Loss':<15} {'Fast Loss':<15} {'Diff':<12} {'Status':<10}")
    print("-" * 80)

    torch.manual_seed(42)
    inputs = [
        (torch.randn(1, 128, 12, 128, device="cuda"), torch.randn(1, 128, 256, device="cuda")) for _ in range(10)
    ]

    for step, (x, freqs_cis) in enumerate(inputs):
        optimizer_orig.zero_grad()
        loss_orig = model_orig(x.clone(), freqs_cis)
        loss_orig.backward()
        optimizer_orig.step()

        optimizer_fast.zero_grad()
        loss_fast = model_fast(x.clone(), freqs_cis)
        loss_fast.backward()
        optimizer_fast.step()

        has_nan_orig = any(p.grad is not None and torch.isnan(p.grad).any() for p in model_orig.parameters())
        has_nan_fast = any(p.grad is not None and torch.isnan(p.grad).any() for p in model_fast.parameters())

        if has_nan_orig or has_nan_fast:
            print(f"❌ Step {step}: Found NaN in gradients")
            return False

        loss_orig_val = loss_orig.item()
        loss_fast_val = loss_fast.item()
        losses_orig.append(loss_orig_val)
        losses_fast.append(loss_fast_val)

        diff = abs(loss_orig_val - loss_fast_val)
        rel_diff = diff / abs(loss_orig_val) if abs(loss_orig_val) > 1e-10 else 0

        if diff < 1e-6:
            status = "✅ Match"
        elif diff < 1e-4:
            status = "✓ Close"
        else:
            status = "⚠️ Differ"

        print(
            f"{step:<6} {loss_orig_val:<15.6f} {loss_fast_val:<15.6f} "
            f"{diff:<12.2e} {status:<10}"
            f"{rel_diff:<12.2e} {status:<10}"
        )

    print("-" * 80)

    avg_diff = sum(abs(o - f) for o, f in zip(losses_orig, losses_fast)) / len(losses_orig)
    max_diff = max(abs(o - f) for o, f in zip(losses_orig, losses_fast))

    print(f"\n{'Summary':<20} {'Original':<15} {'Optimized':<15} {'Difference':<15}")
    print("-" * 65)
    print(
        f"{'Initial loss:':<20} {losses_orig[0]:<15.6f} {losses_fast[0]:<15.6f} "
        f"{abs(losses_orig[0] - losses_fast[0]):<15.2e}"
    )
    print(
        f"{'Final loss:':<20} {losses_orig[-1]:<15.6f} {losses_fast[-1]:<15.6f} "
        f"{abs(losses_orig[-1] - losses_fast[-1]):<15.2e}"
    )
    print(
        f"{'Average loss:':<20} {sum(losses_orig) / len(losses_orig):<15.6f} "
        f"{sum(losses_fast) / len(losses_fast):<15.6f} {avg_diff:<15.2e}"
    )
    print(f"{'Max difference:':<20} {'':<15} {'':<15} {max_diff:<15.2e}")

    weight_diffs = []
    for (name_o, param_o), (name_f, param_f) in zip(model_orig.named_parameters(), model_fast.named_parameters()):
        diff = (param_o - param_f).abs().max().item()
        weight_diffs.append(diff)

    max_weight_diff = max(weight_diffs)
    print(f"{'Max weight diff:':<20} {'':<15} {'':<15} {max_weight_diff:<15.2e}")

    print("=" * 80)

    if max_diff < 1e-4 and max_weight_diff < 1e-4:
        print("✅ Training consistency test PASSED")
        print("   Original and optimized versions produce nearly identical results")
        return True
    elif max_diff < 1e-2:
        print("✓ Training consistency test ACCEPTABLE")
        print("   Small numerical differences detected (within tolerance)")
        return True
    else:
        print("⚠️ Training consistency test WARNING")
        print(f"   Differences detected: loss_diff={max_diff:.2e}, weight_diff={max_weight_diff:.2e}")
        return False


if __name__ == "__main__":
    test_zero_error()
    test_comparison()
    test_backward_comparison()
    test_in_training_loop_comparison()
    # test_backward()
