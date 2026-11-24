# Iris-BiT inference

Run BiT from the repository root with the default Iris and Iris-BiT checkpoint
directories from the [model table](../../README.md#models).

```bash
python -m abot_world.bit.cli --check-weights
python -m abot_world.bit.cli --example 1 --skill 1 --frames 121 \
  --output outputs/iris-bit-demo
```

`--example` selects one of the six scenes; `--skill` selects its first or second
skill for four generation chunks. Omit the skill for movement alone. Use `--teacher-dir` or
`--shared-assets` to override the checkpoint directories.

For custom inputs, replace `--example` and `--skill` with `--first-frame`,
`--action`, and `--prompt` (and optionally `--references`).

- Action JSON: `fps: 16`, `total_frames`, and one row in `frames` per frame.
  Each row has `keys` with boolean values for W, A, S, D, I, J, K, and L.
- Prompt JSON: `mode: "chunkwise"`, `num_frames`, `frame_ranges`,
  `global_prompts`, and `action_prompts`. Use contiguous half-open frame
  ranges with one global and one action caption per range.
- References: named JPEG/PNG images in teacher slot order: `head`, `front`,
  `left`, `right`, `back`.

BiT supports only absolute RoPE and defaults to 48 FlowMatch steps with CFG 3.0.
The negative pass uses the teacher's negative prompt, disables action injection,
and preserves the first frame without reference images. Set `--guidance-scale 1`
to disable CFG.
Start with 121 frames; 321-frame dense attention needs more GPU memory.
The runtime is isolated from the causal student and uses the same software
environment. PyTorch 2.5+ and an attention backend supported by the runtime
are required.
