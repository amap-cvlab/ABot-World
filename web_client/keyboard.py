"""
web_client/keyboard.py — 键盘状态管理与冲突消解。
"""
import json
import sys
from pathlib import Path

# 确保 web_client 模块在路径中
_web_client_dir = Path(__file__).parent
if str(_web_client_dir) not in sys.path:
    sys.path.insert(0, str(_web_client_dir))

from web_client.config import KEY_ORDER, CONFLICT_GROUPS
from web_client.state import state
from abot_world.iris.utils.action_conditions import CONTROL_TEXT


def _resolve_conflicts(keys: set, high_priority: set) -> set:
    """冲突消解：对每个冲突组，若两个键都在 keys 中，保留 high_priority 里的那个。
    若两个都在或都不在 high_priority，保留两者（不强行丢弃）。
    """
    result = set(keys)
    for a, b in CONFLICT_GROUPS:
        if a not in result or b not in result:
            continue
        # 两个键都在，根据优先级决定去掉哪个
        a_is_high = a in high_priority
        b_is_high = b in high_priority
        if a_is_high and not b_is_high:
            result.discard(b)
        elif b_is_high and not a_is_high:
            result.discard(a)
        # 两个都在 high_priority 或都不在：不处理，留给上层逻辑
    return result


def _sample_key_snapshot():
    """在每个 block 开始前采样一次按键状态。

    合并规则：combined = pressed ∪ activated，冲突时 activated 优先于 pressed。
    采样后清空 activated（消费），pressed 不动（持续按住继续生效）。

    返回移动/视角字典，以及基础动作的按住和点按集合。
    """
    with state._key_lock:
        pressed = state.frontend_pressed
        activated = state.frontend_activated
        combined = pressed | activated
        combined = _resolve_conflicts(combined, high_priority=activated)
        held_actions = pressed & CONTROL_TEXT.keys()
        tapped_actions = activated & CONTROL_TEXT.keys()
        state.frontend_activated.clear()
    snapshot = {k: (k in combined) for k in KEY_ORDER}
    state.key_snapshot = snapshot
    return snapshot, held_actions, tapped_actions


def on_key_update(key_json: str):
    """接收前端上报的按键 JSON，更新 StreamState 中的按键状态。"""
    if not key_json or not key_json.strip():
        return
    try:
        data = json.loads(key_json)
        pressed = set(data.get("pressed", []))
        activated = set(data.get("activated", []))
    except Exception:
        return
    with state._key_lock:
        state.frontend_pressed = pressed
        # |= 合并防止覆盖短按；activated 优先级高，冲突时保留新上报的 activated
        merged = state.frontend_activated | activated
        state.frontend_activated = _resolve_conflicts(merged, high_priority=activated)
