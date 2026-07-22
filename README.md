<div align="center">
  <img src="assets/gifs/ABot-World-0.gif">
  <h1>ABot-World: Infinite Interactive World Rollout on a Single Desktop GPU</h1>
</div>

[![Studio](https://img.shields.io/badge/Studio-ABot_World_Studio-green?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHZpZXdCb3g9IjAgMCAzMiAzMiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJtMjkgLjMtLjY0LjA4LS41OC0uMjYtLjU3LjI5LS42NC0uMS0uNjQuMzMtLjYyLS4wNS0xLjM1LjYtLjU5LjA0LTEuODcgMS4xMy4xNS4xNy45MS0uMDEgMS4zOS0uNzcgMS4yMi0uMjMuNjUtLjMzIDIuNTgtLjExIDEuMjguNzguNTUuNjkuMzcuOS4wMiAxLjQxLS4zNiAxLjczLS44IDIuMS0xLjkyIDMuNDMtMy4yNSA0LjUuMDIuMy40LjUuNDYuMjQuNTYtLjI4IDMuMTYtNC40LjUzLTEuMzEuODctMS4yOCAxLjY3LTQuNDJ2LTMuMjJsLS40Mi0uMjUtLjI1LS42OS0uNTEtLjY0em0tMS40MyA2LjYyLTMuNjctMi41OS0zLjc2LTEuMjItMi40Ny0uMTktMi42Ny4xOS0zLjcyIDEuMi0zLjY4IDIuNTgtMS4xOCAxLjM4LTEuOTIgMy4wNC0uNzIgMS44Ni0uNTMgMy4wOC0uMDUgMS40NC40NSAzLjA2LjYgMS44NiAxLjg1IDMuMjMgMS44NSAxLjg5LjM5LjE5LjgxLS4xOSAxLjI3LTEuMDYtMi4xOS0yLjM0LS45Mi0xLjM2LS45NS0yLjA4LS41Ny0zLjI3LjMxLTMgLjc0LTIuMjYgMS41OC0yLjU1IDEuMzEtMS40IDMuNDEtMi4xMSAzLjItLjc4aDIuNDVsLjY5LjIzaC44M2wyLjAzLjY5IDIuMyAxLjM5IDIuNSAyLjIzIDEuMTctMS42NS0uMDgtLjY5em0zLjc3IDYuMTQtLjQxLS43NS0uMzYtLjE1LTEuNDMgMi4zNi4xOCAyLjktLjEyIDEuNTgtLjQ0IDEuOTUtMS4xMiAyLjM5LTIuMTMgMi40NC0yIDEuNjEtMi4wMS45NC0xLjk3LjUtMi43MS4xMi0uNTQtLjItLjcyLjA2LTEuMy0uNC0uMTEtLjM2LjUyLS42IDMuNDItMi4zNyAxLjM2LTEuMjIuMDktLjMzLS4yNS0uNDUtLjgzLS40OS01Ljc1IDQuMzgtMy40NiAyLTMuOCAxLjMxLTIuNjMtLjAxLS45My0uNjYtLjY4LS44LS4yNS0xLjIuMTEtMS44MSAxLjU4LTQuMjItLjA2LS4zLS4zLS4xOS0uMzYuMzgtLjE4Ljc1LS42NiAxLjE3LS4zIDEuMzMtLjM0LjU4LS4wMi43OS0uNDUuNDcuMTYuNS0uMTYgMS4xLjIzLjIuMDIuNyAxLjE2IDEuNjkgMS41MS44Ny43LS4wMy41LjI5LjU4LS4yNi43MS4xMS41Ny0uMjkuNzEuMDEgNC4xOS0xLjU5IDEuNzEuNzUgMy4xMy44M2gzLjc4bDMuMTctLjgzIDIuNDUtMS4yIDEuMzMtLjk0IDIuNjQtMi42NyAxLjgzLTMuMjEuMjgtMS4yNi4zNi0uNjR2LS42OWwuNDItLjQxdi00LjI5bC0uMjYtLjI0LjAzLS41NC0uMzYtLjY4em0tOS44OCA0LjE0LS45OC40OS0uOTIuODctLjU1Ljk5LjAyIDEuMTguNzggMS4yNC44Ni43Mi43LjM0IDEuMzMtLjA4LjY3LS4zNCAxLjE3LTEuMjMuMjItLjcyLS4wMS0xLjI1LS4yMi0uNTUtMS4yMS0xLjI1LS42Mi0uMzR6IiBmaWxsPSIjZmZmIi8+PC9zdmc+Cg==)](https://abot-world.amap.com)
[![Playground](https://img.shields.io/badge/Reactor-ABot_World-E9E4C2?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyOCAyMCIgd2lkdGg9IjI4IiBoZWlnaHQ9IjIwIj4KICA8cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZmlsbD0iIzAwMDAwMCIgZD0iTTIzLjg1NjYgMC4zMzQ1NDNDMjYuMjU0OSAwLjMzNDU0MyAyNy4zMTgzIDEuNzQwNDggMjcuMzE4MyAzLjY2MDIzVjkuMzEwNjRDMjcuMzE4MyAxMS4xNzYgMjYuMjgyNyAxMi41ODE5IDIzLjkzODIgMTIuNTgxOUgyMy44ODIyQzIzLjI3MjMgMTIuNTg0IDIyLjkxMTEgMTMuMjYwNiAyMy4yNTIxIDEzLjc2MjFMMjcuMjYzNCAxOS42NjU1SDIyLjM5OTRMMTcuNzU2OSAxMy4wNDg4QzE3LjU1MTUgMTIuNzU2NCAxNy4yMTQ4IDEyLjU4MTkgMTYuODU1NyAxMi41ODE5QzE2LjI4OTEgMTIuNTgxOSAxNS43NTc3IDEzLjA2OTYgMTUuNzU3NyAxMy42NzA5VjE5LjY2NTVIMTEuNTU5NlYxMy42NzA5QzExLjU1OTYgMTMuMDY5NiAxMS4wNjgyIDEyLjU4MTkgMTAuNDYxNSAxMi41ODE5QzEwLjEwMTkgMTIuNTgxOSA5Ljc2NTc3IDEyLjc1NjQgOS41NjAzNSAxMy4wNDg4TDQuOTE3ODMgMTkuNjY1NUgwLjA1NDQyMzJMNC4wNjYyNyAxMy43NjIxQzQuNDA3MjEgMTMuMjYwNiA0LjA0NjUzIDEyLjU4NCAzLjQzNjEzIDEyLjU4MTlIMy4zODAxMUMxLjAzNjE4IDEyLjU4MTkgMCAxMS4xNzYgMCA5LjMxMDExVjMuNjU5N0MwIDEuNzQwNDggMS4wNjI4NSAwLjMzNDU0MyAzLjQ2MTc0IDAuMzM0NTQzSDIzLjg1NjZaTTQuNjg4NCA0LjExOTYzQzQuMzA2OSA0LjExOTYzIDQuMTk4MDYgNC4yNTQ2MiA0LjE5ODA2IDQuNjMzNDRWOC4zMzc0MkM0LjE5ODA2IDguNjg5MDQgNC4zMDY5IDguNzk2ODIgNC42NjE3MiA4Ljc5NjgySDIyLjY1NzdDMjMuMDEyIDguNzk2ODIgMjMuMTIwOCA4LjY4ODUxIDIzLjEyMDggOC4zMzc0MlY0LjYzMzQ0QzIzLjEyMDggNC4yNTUxNSAyMy4wMTIgNC4xMTk2MyAyMi42MzA1IDQuMTE5NjNINC42ODg0WiIvPgo8L3N2Zz4=)](https://reactor.inc/abot-world)
[![Project](https://img.shields.io/badge/🌐_Project-ABot_World-blue)](https://amap-cvlab.github.io/ABot-World/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2607.19191)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github)](https://github.com/amap-cvlab/ABot-World)
[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/acvlab/ABot-World-0-5B-LF)
[![Space](https://img.shields.io/badge/Space-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/spaces/acvlab/abot-world-interactive)
[![Paper](https://img.shields.io/badge/Paper-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/papers/2607.19191)
[![Model](https://img.shields.io/badge/Model-ModelScope-7061FF?logo=modelscope)](https://modelscope.cn/models/amap_cvlab/ABot-World-0-5B-LF)



<div align="center">
  <h3>ABot-World Team</h3> <br>
</div>  

> **TL;DR:** ABot-World turns a single NVIDIA RTX 5090 desktop GPU into a real-time interactive world simulator, enabling infinite action-conditioned world rollout at 720P, 16 FPS, 1.2s latency, and 19GB GPU memory.

## 🚀 Key Highlights

* 🎮 **Action-Driven World Control:** Responds to user actions in real time, enabling continuous exploration instead of passive video playback.
* ⚡ **Real-Time Desktop Inference:** Runs at 720p and 16 FPS on a single NVIDIA RTX 5090 desktop GPU, with 1.2s latency and 19GB GPU memory.
* ♾️ **Infinite World Rollout:** Supports open-ended interactive world generation beyond fixed video-length limits.
* 🧠 **Open-Ended World Imagination:** Expands the world with new scenes and dynamics during rollout, avoiding scene lock-in, without prompt switching, by our *LongForcing* training.

## 📢 News

- 2026
- 2026-07-13: ABot-World is now on [Reactor](https://reactor.inc/abot-world)!
- 2026-07-10: We have decided to open-source our `500-hour` video training dataset with accurate action annotations. Stay tuned—we plan to release it very soon.
- 2026-07-09: We release the causal student model `ABot-World-0-5B-LF`, inference code, our local gradio demo and online playground [ABot World Studio](https://abot-world.amap.com).

## 🛠️ Setup

> This installation was tested on: Ubuntu 22.04, CUDA 13.3, NVIDIA RTX 5090.

1. Clone the repository:

```bash
git clone https://github.com/amap-cvlab/ABot-World.git
cd ABot-World
```

2. Install dependencies using conda:

```bash
conda create -n aworld python=3.12 -y
conda activate aworld
pip install -r requirements.txt
```

3. Download checkpoints:

Download models using HuggingFace:

```bash
pip install -U "huggingface_hub"
hf download acvlab/ABot-World-0-5B-LF --local-dir ./checkpoints/ABot-World-0-5B-LF
```

Download models using ModelScope:

```bash
pip install -U "modelscope"
modelscope download "amap_cvlab/ABot-World-0-5B-LF" --local_dir ./checkpoints/ABot-World-0-5B-LF
```

After downloading, the project should have the following checkpoint structure:

```text
checkpoints/
└── ABot-World-0-5B-LF/
    ├── Wan2.2_VAE.pth
    ├── taew2_2.pth
    ├── models_t5_umt5-xxl-enc-bf16.pth
    ├── diffusion_pytorch_model.safetensors
    └── google/umt5-xxl/
```

The checkpoint paths are configured in `configs/long_forcing_dmd.yaml` and
`configs/default_config.yaml`. The distilled generator weights are already
merged into `ABot-World-0-5B-LF/diffusion_pytorch_model.safetensors`.

## 🤗 Gradio Demo

```bash
bash web_client/run.sh
```

Select a GPU with:

```bash
CUDA_ID=0 bash web_client/run.sh
```

## License

This project is released under the Apache License 2.0. See `LICENSE`, `NOTICE`,
and `THIRD_PARTY_NOTICES.md` for copyright and third-party attribution details.

## 🤝 Acknowledgement

This project builds on and is inspired by the following open-source projects: [Causal Forcing](https://github.com/thu-ml/Causal-Forcing), [AngelSlim](https://github.com/tencent/AngelSlim), [LightX2V](https://github.com/ModelTC/LightX2V), [taehv](https://github.com/madebyollin/taehv), [Wan2.2](https://github.com/Wan-Video/Wan2.2), [Helios](https://github.com/PKU-YuanGroup/Helios), from which the optimized Triton RoPE and normalization kernels in `wan/modules/helios_kernels` are derived.

## 🗓️ Roadmap
- [x] Interactive Web Playground (ABot World Studio)
- [x] Inference Code Release
- [x] Local Gradio Demo Release
- [x] Causal Student Model Release
- [x] Technical Report (Arxiv)
- [ ] Bidirectional Teacher Model Release
- [ ] 500-Hour Video Training Dataset with Accurate Action Annotations

## 📝 Citation
If you find our work helpful, please cite our paper:

```
@article{abot-world-0,
      title={ABot-World-0: Infinite Interactive World Rollout on a Single Desktop GPU}, 
      author={ABot-World Team},
      year={2026}
}
```

## 🛰 Contact Us via WeChat Group
Feel free to contact us!
<div align="center">
  <img src="http://amap-cvlab.oss-cn-zhangjiakou.aliyuncs.com/github/imgs/abot-world-wechat.jpg?v=0729" width=30%>
</div>



