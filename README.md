<div align="center">
  <img src="assets/gifs/ABot-World-0.gif">
  <h1>ABot-World: Infinite Interactive World Rollout on Single Desktop GPU</h1>
</div>

[![Studio](https://img.shields.io/badge/🎮%20%20Studio-ABot%20%20World%20%20Studio-green.svg)](https://abot-world.amap.com)
[![Project](https://img.shields.io/badge/🌐%20%20Project-ABot%20%20World-blue.svg)](https://amap-cvlab.github.io/ABot-World/)
[![Paper](https://img.shields.io/badge/Arxiv-Coming_Soon-red)](#)
[![Code](https://img.shields.io/badge/Code-GitHub-181717.svg?logo=GitHub)](https://github.com/amap-cvlab/ABot-World)
[![Model](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=HuggingFace&color=yellow)](https://huggingface.co/acvlab/ABot-World-0-5B-LF)
[![Model](https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Model&message=ModelScope&color=purple)](https://modelscope.cn/models/amap_cvlab/ABot-World-0-5B-LF)


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
- [ ] Bidirectional Teacher Model Release
- [ ] 500-Hour Video Training Dataset with Accurate Action Annotations
- [ ] Technical Report (Arxiv)

## 📝 Citation
If you find our work helpful, please cite our paper:

```
@article{abot-world-0,
      title={ABot-World-0: Infinite Interactive World Rollout on Single Desktop GPU}, 
      author={ABot-World Team},
      year={2026}
}
```

## 🛰 Contact Us via WeChat Group
Feel free to contact us!
<div align="center">
  <img src="http://amap-cvlab.oss-cn-zhangjiakou.aliyuncs.com/github/imgs/abot-world-wechat.jpg" width=30%>
</div>



