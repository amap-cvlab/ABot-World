# 🌍 ABot-World: Real-Time Interactive World Simulation on a Single Desktop GPU

[![Project](https://img.shields.io/badge/🌐%20%20Project-ABot%20%20World-blue.svg)](https://amap-cvlab.github.io/abot-world/)
[![Paper](https://img.shields.io/badge/Arxiv-Coming_Soon-red)](#)
[![Code](https://img.shields.io/badge/Code-Coming_Soon-181717.svg?logo=GitHub)](https://github.com/amap-cvlab/ABot-World)
[![Model](https://img.shields.io/badge/Weights-Coming_Soon-yellow)](#)

> **TL;DR:** ABot-World is a next-generation Action-Conditioned World Model that achieves **real-time interactive inference on a single consumer-grade Desktop GPU**.

We are currently preparing the technical report, code, and model weights for public release. In the meantime, explore our interactive demos below.

## 🚀 Key Highlights

* ⚡ **Real-Time Desktop Inference:** Breaking the latency barrier. Through highly optimized inference architecture and compute-efficient scaling, ABot-World enables fluid, real-time rollout generation on a single desktop GPU. You no longer need a data center to simulate the world.
* 🧠 **Native Spatial Intelligence**: To capture the infinite variations of the physical world, ABot-World extracts robust spatial structures and complex dynamics implicitly. By natively embedding physical priors into its generation process, it seamlessly unifies visual rendering and physical simulation within the latent space.
* ♾️ **Long-Horizon Consistency:** Exposure bias and compounding errors have long plagued auto-regressive generation. By introducing a novel Parametric Memory module that acts as a global state sink, ABot-World maintains strict spatial topology and physical consistency over extraordinarily long rollouts.
* 🏆 **Top-Tier Physical Reasoning:** Evaluated against state-of-the-art models, ABot-World achieves top-tier performance on industry-standard benchmarks including [**VBench 2.0**](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard) (annouced by [ABot-World v0.1](https://github.com/amap-cvlab/ABot-World)), [**WorldArena**](https://huggingface.co/spaces/WorldArena/WorldArena) & [**GigaBran**](https://huggingface.co/spaces/open-gigaai/CVPR-2026-WorldModel-Track-LeaderBoard) (announced by [ABot-PhysWorld](https://huggingface.co/spaces/WorldArena/WorldArena)) and [**WorldScore**](https://huggingface.co/spaces/Howieeeee/WorldScore_Leaderboard) (announced by [FantasyWorld 1.0](https://fantasy-amap.github.io/fantasy-world/)), particularly excelling in physics adherence and dynamic motion quality.

## 🕹️ Interactive Demonstrations

### 1. Open World Exploration
Experience fluid and consistent generation as you interact with the environment. Whether navigating via first-person camera adjustments or driving a character through diverse scenes, the simulation responds dynamically to your inputs. The model natively understands spatial depth, object permanence, and complex environmental interactions, delivering an immersive experience from any viewpoint.

*<img src="assets/gifs/open-world-exploration-main.gif" width="800" alt="More Open World Exploration Demo">*

#### More Demos
*<img src="assets/gifs/open-world-exploration.gif" width="800" alt="More Open World Exploration Demo">*

### 2. Long-Horizon World Exploration via a single Desktop GPU
A continuous, uninterrupted rollout showcasing the power of our Parametric Memory. Through a highly compute-efficient architecture, ABot-World sustains strict global state topology and physical consistency over extended durations, generating expansive environments locally on a single Desktop GPU.

*<img src="assets/gifs/long-horizon.gif" width="800" alt="Long Horizon World Exploration Demo">*

🔗 **[Explore our previous work FantasyWorld](https://fantasy-amap.github.io/fantasy-world/)**

### 3. Embodied Interaction & Physics Alignment (ABot-PhysWorld)
A specialized variant of our foundation model, **ABot-PhysWorld**, focuses on robotic manipulation. It uses a novel DPO-based post-training framework with decoupled discriminators to actively suppress unphysical behaviors like object penetration and anti-gravity motion. A parallel context block enables precise spatial action injection for cross-embodiment control.

🔗 **[Explore the ABot-PhysWorld Repository](https://github.com/amap-cvlab/ABot-PhysWorld)**

## 🗓️ Roadmap
- [x] Teaser & Demo Release
- [ ] Technical Report (Arxiv)
- [ ] Inference Code Release
- [ ] Model Weights (Open Source)
- [ ] Interactive Web Playground

## 📝 Citation
If you find our work helpful, please stay tuned for our upcoming paper.