# Third-Party Notices

Unless otherwise noted, ABot-World source code is distributed under the Apache
License, Version 2.0. This file summarizes third-party components that are
included, adapted, or required by this repository.

## Wan / Wan2.2

- Upstream: https://github.com/Wan-Video/Wan2.2
- License: Apache License 2.0
- Copyright notice retained in source files:
  `Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.`
- Local usage: model, attention, text encoder, tokenizer, VAE, and related Wan
  modules under `abot_world/iris/wan/` and `abot_world/bit/wan/`.

## AngelSlim

- Upstream: https://github.com/tencent/AngelSlim
- License: Apache License 2.0
- Copyright notice retained in source files:
  `Copyright 2025 Tencent Inc. All Rights Reserved.`
- Local usage: quantization utilities under `quantizer/`.

## LightX2V

- Upstream: https://github.com/ModelTC/LightX2V
- Local usage: optional optimized low-precision inference operators used by the
  quantized DiT path. This dependency is installed separately and is not
  vendored in this repository.

## taehv

- Upstream: https://github.com/madebyollin/taehv
- Local usage: TAEW2.2 lightweight streaming VAE wrapper and related code paths.

## Helios

- Upstream: https://github.com/PKU-YuanGroup/Helios
- Local usage: optimized Triton RoPE and normalization kernels under
  `abot_world/iris/wan/modules/helios_kernels/`.

## Demo Assets

The sample images and pre-generated reference-image cache files under
`examples/first_frames/` and `examples/references/` are included for
running the demo presets. Replace or remove them before redistribution if your
release process requires separate asset provenance records.
