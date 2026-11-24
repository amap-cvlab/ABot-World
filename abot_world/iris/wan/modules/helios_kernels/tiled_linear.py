import functools
import math
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, LinearActivation, SwiGLU
from diffusers.utils import deprecate


# ------------------------------- replace funtion -------------------------------


def replace_linear_with_tiled_linear(model, num_shards=None, patch_by_names=True, patch_by_types=True):
    target_names = ["to_q", "to_k", "to_v", "add_k_proj", "add_v_proj"]
    target_types = ["FeedForward"]

    patched_count = 0

    def tiled_forward(self, x):
        compute_params = list(self.parameters())
        return apply_tiled_linear(
            fn=lambda module, input: module._original_forward(input),
            mlp_module=self,
            x=x,
            num_shards=num_shards,
            compute_params=compute_params,
        )

    for name, module in model.named_modules():
        layer_name = name.rsplit(".", 1)[-1] if "." in name else name
        module_type = type(module).__name__

        should_patch = False
        if patch_by_types and module_type in target_types:
            should_patch = True
        if patch_by_names and layer_name in target_names and isinstance(module, torch.nn.Linear):
            should_patch = True

        if should_patch:
            module._original_forward = module.forward
            module.forward = tiled_forward.__get__(module, module.__class__)
            patched_count += 1
            # print(f"  Patched {module_type}: {name}")

    print(f"Patched {patched_count} FeedForward modules with TiledMLP\n")
    return model


# ------------------------------- Tiled MLP -------------------------------


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


class TiledLinear(torch.autograd.Function):
    """
    Based on DeepSpeed's TiledMLP:
    https://github.com/deepspeedai/DeepSpeed/blob/v0.18.2/deepspeed/runtime/sequence_parallel/ulysses_sp.py#L838

    Perform a tiled MLP computation to massively reduce memory usage needed to compute MLP
    when using very long sequence lengths.

    This module re-computes `forward` in the `backward`. So the `forward` occurs twice each iteration.
    And if you're using activation checkpointing it then occurs thrice.

    Args:
        fn: the function to call on sharded inputs (e.g., mlp.forward)
        mlp_module: the MLP nn.Module object
        x: the input to MLP.forward (hidden_states)
        shards: how many shards to use
        compute_params: a list of weights engaged in the compute

    Returns:
        the computed hidden_states
    """

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        fn: Callable,
        mlp_module: torch.nn.Module,
        x: torch.Tensor,
        shards: int,
        compute_params: Optional[List[torch.nn.Parameter]] = None,
    ) -> torch.Tensor:
        ctx.fn = fn
        ctx.mlp_module = mlp_module
        ctx.shards = shards
        ctx.save_for_backward(x)

        # x.shape could be [bs, seqlen, hidden_size] or [seqlen, hidden_size] (moe experts)
        x_shards = list(torch.chunk(x, chunks=shards, dim=-2))
        with torch.no_grad():
            output_shards = [fn(mlp_module, x_shard) for x_shard in x_shards]
        output_unsharded = torch.cat(output_shards, dim=-2)

        return output_unsharded

    @staticmethod
    @ensure_contiguous
    def backward(ctx, *grads) -> tuple:
        fn = ctx.fn
        (x,) = ctx.saved_tensors
        mlp_module = ctx.mlp_module
        shards = ctx.shards

        x_requires_grad = x.requires_grad
        x = x.detach()
        # detach() unsets x.requires_grad, so restore it
        x.requires_grad_(x_requires_grad)

        # x.shape could be [bs, seqlen, hidden_size] or [seqlen, hidden_size] (moe experts)
        hidden_size = x.shape[-1]
        x_shape_orig = x.shape

        # flatten bs+seqlen to avoid having stride issues when narrowing into seqlen w/ bs>1
        x = x.view(-1, hidden_size)
        incoming_grad = grads[0].view(-1, hidden_size)
        x_grad = torch.zeros_like(x)

        x_shards = list(torch.chunk(x, chunks=shards, dim=0))

        trainable_params = [p for p in mlp_module.parameters() if p.requires_grad]

        for i, x_shard in enumerate(x_shards):
            x_shard = x_shard.detach().requires_grad_(x_requires_grad)

            shard_step = x_shards[i].shape[0]
            shard_offset = i * x_shards[0].shape[0]

            incoming_grad_shard = incoming_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)

            with torch.enable_grad():
                output = fn(mlp_module, x_shard)

            grads_tuple = torch.autograd.grad(
                outputs=output,
                inputs=[x_shard] + trainable_params,
                grad_outputs=incoming_grad_shard,
                allow_unused=True,
                retain_graph=False,
            )

            x_grad.narrow(0, shard_offset, shard_step).copy_(grads_tuple[0])

            for param, grad in zip(trainable_params, grads_tuple[1:]):
                if grad is not None:
                    if param.grad is None:
                        param.grad = grad
                    else:
                        param.grad.add_(grad)

        # unflatten
        x_grad = x_grad.view(x_shape_orig)

        return (None, None, x_grad, None, None)


def apply_tiled_linear(
    fn: Callable,
    mlp_module: torch.nn.Module,
    x: torch.Tensor,
    num_shards: Optional[int] = None,
    compute_params: Optional[List[torch.nn.Parameter]] = None,
) -> torch.Tensor:
    """
    Apply tiled MLP computation for memory efficiency.

    Args:
        fn: the function to call on sharded inputs (e.g., lambda module, x: module(x))
        mlp_module: the MLP nn.Module object
        x: the input tensor with shape [bs, seqlen, hidden_size] or [seqlen, hidden_size]
        num_shards: number of shards to use. If None, automatically calculated as ceil(seqlen / hidden_size)
        compute_params: list of parameters for DeepSpeed ZeRO optimization

    Returns:
        output tensor with the same shape as input
    """
    if num_shards is None:
        # x.shape could be [bs, seqlen, hidden_size] or [seqlen, hidden_size]
        hidden_size = x.shape[-1]
        seqlen = x.shape[-2]
        num_shards = math.ceil(seqlen / hidden_size)

    # Ensure num_shards is at least 1
    num_shards = max(1, num_shards)

    return TiledLinear.apply(
        fn,
        mlp_module,
        x,
        num_shards,
        compute_params,
    )


# ------------------------------- Tiled FeedForward -------------------------------
class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "linear-silu":
            act_fn = LinearActivation(dim, inner_dim, bias=bias, activation="silu")

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class TiledFeedForward(nn.Module):
    """
    Memory-efficient FeedForward using tiled computation (diffusers compatible)
    Args:
        dim: Input dimension
        dim_out: Output dimension (default: dim)
        mult: Multiplier for inner dimension (default: 4)
        dropout: Dropout probability
        activation_fn: Activation function ('geglu', 'gelu', 'gelu-approximate')
        final_dropout: Apply dropout at the end
        inner_dim: Inner dimension (overrides mult if provided)
        bias: Use bias in linear layers
        num_shards: Number of shards for tiling (None = auto)
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim: Optional[int] = None,
        bias: bool = True,
        num_shards: Optional[int] = None,
    ):
        super().__init__()

        # Calculate dimensions
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.dim = dim
        self.inner_dim = inner_dim
        self.dim_out = dim_out
        self.activation_fn = activation_fn
        self.num_shards = num_shards

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "linear-silu":
            act_fn = LinearActivation(dim, inner_dim, bias=bias, activation="silu")

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def _mlp_forward(self, module, x):
        """Internal MLP forward for tiled computation"""
        for layer in module.net:
            x = layer(x)
        return x

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with tiled computation
        Args:
            hidden_states: [batch_size, seq_len, dim] or [seq_len, dim]
        Returns:
            Output tensor with same shape as input (but last dim = dim_out)
        """
        # Collect compute parameters
        compute_params = list(self.parameters())

        return apply_tiled_linear(
            fn=self._mlp_forward,
            mlp_module=self,
            x=hidden_states,
            num_shards=self.num_shards,
            compute_params=compute_params,
        )


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # 设置随机种子保证可重复性
    torch.manual_seed(42)

    # 创建测试输入
    batch_size, seq_len, hidden_dim = 2, 1024, 768
    x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

    # 方法1: replace
    model1 = FeedForward(dim=hidden_dim)
    # model1 = replace_linear_with_tiled_linear(model1, num_shards=4)
    out1 = model1(x)
    loss1 = out1.sum()
    loss1.backward()
    grad1 = x.grad.clone()

    # 方法2: TiledFeedForward
    x.grad = None
    # model2 = TiledFeedForward(dim=hidden_dim, num_shards=4)
    model2 = FeedForward(dim=hidden_dim)
    model2 = replace_linear_with_tiled_linear(model2, num_shards=4)
    # 复制权重确保完全一致
    model2.load_state_dict(model1.state_dict(), strict=True)
    out2 = model2(x)
    loss2 = out2.sum()
    loss2.backward()
    grad2 = x.grad.clone()

    # 比较结果
    print(f"Output diff: {(out1 - out2).abs().max().item()}")
    print(f"Gradient diff: {(grad1 - grad2).abs().max().item()}")
    print(f"Output allclose: {torch.allclose(out1, out2, atol=1e-6)}")
    print(f"Gradient allclose: {torch.allclose(grad1, grad2, atol=1e-6)}")
