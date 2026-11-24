"""Pack per-frame WASDIJKL inputs for the teacher action adapter."""
import torch


def pack_action_context(
    action: torch.Tensor,
    *,
    height: int,
    width: int,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Pack raw ``[T,8]`` controls as ``[1,32,1+(T-1)/4,H,W]``.

    The layout exactly matches the former cache-time ``preprocess_action``
    implementation, but spatial expansion stays a view until the adapter makes
    its own contiguous frame batch.
    """

    action = torch.as_tensor(action, device=device, dtype=dtype)
    if action.ndim != 2 or action.shape[1] != 8:
        raise ValueError(f"action must have shape [T,8], got {tuple(action.shape)}")
    if action.shape[0] % 4 != 1:
        raise ValueError(
            "raw action frame count must satisfy T % 4 == 1, got "
            f"T={action.shape[0]}"
        )
    packed = torch.cat([action[:1].expand(4, -1), action[1:]], dim=0)
    packed = packed.reshape(-1, 4, 8).permute(0, 2, 1).reshape(-1, 32)
    packed = packed.transpose(0, 1).reshape(1, 32, packed.shape[0], 1, 1)
    return packed.expand(-1, -1, -1, int(height), int(width))
