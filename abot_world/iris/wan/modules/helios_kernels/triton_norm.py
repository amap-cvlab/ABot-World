import torch
import triton
import triton.language as tl

from diffusers.models.normalization import FP32LayerNorm, LayerNorm, RMSNorm

from .fp32_rmsnorm import FP32RMSNorm
from .utils import calculate_settings, torch_gpu_device


# ------------------------------- replace funtion -------------------------------


def replace_all_norms_with_flash_norms(model):
    patched_count = {"LayerNorm": 0, "RMSNorm": 0}

    for name, module in model.named_modules():
        if isinstance(module, (LayerNorm, FP32LayerNorm)):
            if hasattr(module, "elementwise_affine") and module.elementwise_affine:
                module.forward = (lambda self, x: flash_layernorm(self, x)).__get__(module, module.__class__)
                patched_count["LayerNorm"] += 1

        if isinstance(module, (torch.nn.RMSNorm, RMSNorm, FP32RMSNorm)):
            module.forward = (lambda self, x: flash_rms_layernorm(self, x)).__get__(module, module.__class__)
            patched_count["RMSNorm"] += 1

    print(f"Patched {patched_count['LayerNorm']} Flash_LayerNorm modules\n")
    print(f"Patched {patched_count['RMSNorm']} Flash_RMSNorm modules\n")

    return model


# ------------------------------- layer norm -------------------------------


@triton.jit
def layernorm_forward(
    Y,
    Y_row_stride,
    X,
    X_row_stride,
    W,
    b,
    r,
    mu,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx
    mu += row_idx

    # According to https://pytorch.org/torchtune/stable/_modules/torchtune/modules/layer_norm.html#Fp32LayerNorm, all modules
    # are in float32!
    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0).to(tl.float32)

    mean_X = tl.sum(X_row, axis=0) / n_cols
    # (X[0] - mean) == -mean so we need to mask it out
    XX = tl.where(mask, X_row - mean_X, 0)
    row_var = tl.sum(XX * XX, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    tl.store(mu, mean_X)
    output = (XX * inv_var) * W_row + b_row
    tl.store(Y + col_offsets, output, mask=mask)


@triton.jit
def layernorm_backward(
    dY,
    dY_row_stride,
    X,
    X_row_stride,
    W,
    b,
    r,
    mu,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Approximately follows https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X += row_idx * X_row_stride
    r += row_idx
    mu += row_idx

    # According to https://pytorch.org/torchtune/stable/_modules/torchtune/modules/layer_norm.html#Fp32LayerNorm, all modules
    # are in float32!
    dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
    # b_row = tl.load(b + col_offsets, mask = mask, other = 0).to(tl.float32)

    inv_var = tl.load(r).to(tl.float32)
    mean = tl.load(mu).to(tl.float32)
    normed = (X_row - mean) * inv_var
    dY_W = dY_row * W_row
    dX_row = dY_W - tl.sum(dY_W, axis=0) / n_cols - normed * tl.sum(dY_W * normed, axis=0) / n_cols
    dX_row = dX_row * inv_var
    tl.store(dY + col_offsets, dX_row, mask=mask)


class Flash_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, eps):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        device = X.device
        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=device)
        r = torch.empty(n_rows, dtype=torch.float32, device=device)
        mu = torch.empty(n_rows, dtype=torch.float32, device=device)

        with torch_gpu_device(device):
            layernorm_forward[(n_rows,)](
                Y,
                Y.stride(0),
                X,
                X.stride(0),
                W,
                b,
                r,
                mu,
                n_cols,
                eps,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X, W, b, r, mu)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, W, b, r, mu = ctx.saved_tensors
        n_rows, n_cols = dY.shape

        with torch_gpu_device(dY.device):
            layernorm_backward[(n_rows,)](
                dY,
                dY.stride(0),
                X,
                X.stride(0),
                W,
                b,
                r,
                mu,
                n_cols,
                ctx.eps,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=ctx.num_warps,
            )
        dX = dY.view(*shape)
        return dX, None, None, None, None


def flash_layernorm(layernorm, X):
    assert layernorm.elementwise_affine is True
    W = layernorm.weight
    bias = layernorm.bias
    eps = layernorm.variance_epsilon if hasattr(layernorm, "variance_epsilon") else layernorm.eps
    out = Flash_Layernorm.apply(X, W, bias, eps)
    return out


# ------------------------------- layer norm -------------------------------


# ------------------------------- rms norm -------------------------------


@triton.jit
def _rms_layernorm_forward(
    Y,
    Y_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Flash RMS Layernorm kernel
    Inspiration from a Triton tutorial:
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)  # .to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype)  # Exact copy from HF
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask=mask)


def _rms_layernorm_backward(
    dY,
    dY_row_stride: tl.constexpr,
    dX,
    dX_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    # dW, dW_row_stride,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    GEMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Flash RMS Layernorm kernel for the backward pass
    Inspiration from a Triton tutorial:
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    if GEMMA:
        dX += row_idx * dY_row_stride
    else:
        dX = dY

    dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

    # Get saved row variance
    inv_var = tl.load(r).to(tl.float32)
    normed = X_row * inv_var

    if GEMMA:
        dY_W = dY_row * (W_row + 1.0)
    else:
        dY_W = dY_row * W_row

    rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
    output = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
    tl.store(dX + col_offsets, output, mask=mask)


_rms_layernorm_backward = triton.jit(_rms_layernorm_backward)
_rms_layernorm_backward = triton.heuristics(
    {
        "GEMMA": lambda args: bool(args["GEMMA"]),
    }
)(_rms_layernorm_backward)


@triton.jit
def _gemma_rms_layernorm_forward(
    Y,
    Y_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Copies https://github.com/google-deepmind/gemma/blob/main/gemma/layers.py#L31
    # and https://github.com/keras-team/keras-nlp/blob/v0.8.2/keras_nlp/models/gemma/rms_normalization.py#L33
    # exactly. Essentially all in float32!
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    output = normed * (W_row + 1.0)

    tl.store(Y + col_offsets, output, mask=mask)


class Flash_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor, eps: float, gemma: bool = False):
        shape = X.shape
        dim: int = shape[-1]
        X = X.reshape(-1, dim)
        n_rows: int
        n_cols: int
        n_rows, n_cols = X.shape
        BLOCK_SIZE: int
        num_warps: int
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        device = X.device

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=device)
        r = torch.empty(n_rows, dtype=torch.float32, device=device)

        fx = _gemma_rms_layernorm_forward if gemma else _rms_layernorm_forward
        with torch_gpu_device(device):
            fx[(n_rows,)](
                Y,
                Y.stride(0),
                X,
                X.stride(0),
                W,
                W.stride(0),
                r,
                r.stride(0),
                n_cols,
                eps,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.GEMMA = gemma
        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        shape = dY.shape
        dim: int = shape[-1]
        dY = dY.reshape(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows: int
        n_cols: int
        n_rows, n_cols = dY.shape
        # dW = X
        dX = torch.empty_like(dY) if ctx.GEMMA else dY

        with torch_gpu_device(dY.device):
            _rms_layernorm_backward[(n_rows,)](
                dY,
                dY.stride(0),
                dX,
                dX.stride(0),
                X,
                X.stride(0),
                W,
                W.stride(0),
                r,
                r.stride(0),
                # dW, dW.stride(0),
                n_cols,
                ctx.eps,
                GEMMA=ctx.GEMMA,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=ctx.num_warps,
            )
        dX = dX.view(*shape)
        return dX, None, None, None


# [TODO] Unsure why RMS Layernorm is not torch.compiling properly
@torch.compiler.disable
def flash_rms_layernorm(layernorm, X: torch.Tensor, gemma: bool = False):
    W: torch.Tensor = layernorm.weight
    eps: float = layernorm.variance_epsilon if hasattr(layernorm, "variance_epsilon") else layernorm.eps
    out = Flash_RMS_Layernorm.apply(X, W, eps, gemma)
    return out


# ------------------------------- rms norm -------------------------------
