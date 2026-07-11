# PF_Wan

Image-to-video (I2V) robot world model — the Wan2.2-A14B model from **PhysisForcing**
([paper](https://arxiv.org/abs/2606.28128) · [project](https://github.com/DAGroup-PKU/PhysisForcing)).
Given a first frame and a prompt describing the motion, it generates a short video
that continues from that frame.

Self-contained inference bundle: framework code + weights + example inputs.

```
.
├── project/            # inference code (DiT / VAE / T5, samplers, FSDP + Ulysses)
├── configs/            # inference config (generate/) + architecture configs (meta_model/)
├── tools/              # main.py entrypoint + infer_1node.sh launcher
├── checkpoints/PF_Wan/ # weights: backbone (.pth) + T5 + VAE + tokenizer
└── assets/examples/    # example manifest (prompt@@image) + conditioning images
```

## 1. Environment

Python 3.11 + CUDA 12.8, in a dedicated environment (keep it separate from `pf_cosmos` —
the two bundles need different PyTorch/CUDA versions). From the repository root:

```shell
conda create -n pf_wan python=3.11 -y   # or: python3.11 -m venv .venv && source .venv/bin/activate
conda activate pf_wan

# PyTorch + FlashAttention (CUDA-specific wheels)
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.7-cp311-cp311-linux_x86_64.whl

# Everything else
pip install -r requirements.txt
```

For a different CUDA version, swap the two `cu128` wheels for the matching build.

## 2. Weights

Download the bundle into `checkpoints/PF_Wan/` (not tracked in git):

```shell
huggingface-cli download DAGroup-PKU/PF_Wan --local-dir checkpoints/PF_Wan
```

The bundle contains everything the config needs — no separate downloads:

```
checkpoints/PF_Wan/
├── backbone.pth                        # PF_Wan DiT (a vanilla WanModel state_dict)
├── models_t5_umt5-xxl-enc-bf16.pth     # T5 text encoder
├── Wan2.1_VAE.pth                      # VAE
└── google/umt5-xxl/                    # tokenizer
```

## 3. Run inference

From the repository root (8 GPUs by default):

```shell
bash tools/infer_1node.sh
# equivalent to:
bash tools/infer_1node.sh configs/generate/pf_wan_i2v.jsonc 8 29510
```

- Args: `bash tools/infer_1node.sh [config] [nproc_per_node] [master_port]`. To change the
  GPU count, set `nproc_per_node` **and** `meta_model.ulysses_size` in the config to match.
- Outputs land at `samples/pf_wan_i2v/<timestamp>/<name>.mp4`; the run log goes to `logs/`.
- Runs in bf16 with FSDP + Ulysses sequence parallel.

## 4. Input format

Inference is driven by a **manifest** file (`inference.positive_prompt` in the config),
one sample per line:

```
<text prompt>@@<conditioning first-frame image>
```

- Left of `@@`: the positive prompt (the motion to generate).
- Right of `@@`: the first-frame image — a path relative to the repo root, or an `hdfs://...` URL.
- Clip settings come from the config: `num_frames: 81` (must satisfy the VAE constraint
  `4k + 1`), `fps: 16` → ~5 s; resolution capped by `max_area: 307200` (~0.3 MP).

A runnable example manifest (13 samples) is in [`assets/examples/manifest.txt`](assets/examples/manifest.txt).
Point the config at your own manifest to run a larger batch.

## Citation

```bibtex
@article{zhang2026physisforcing,
  title={PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation},
  author={Zhang, Peiwen and Deng, Yufan and Sun, Shangkun and Ma, Juncheng and
          Wang, Duomin and Du, Jonas and Pan, Zilin and Huang, Ye and Liang, Hao and
          Huang, Songyan and Zhang, Ruihua and Xie, Enze and Liu, Ming-Yu and Zhou, Daquan},
  journal={arXiv preprint arXiv:2606.28128},
  year={2026}
}
```

Framework code retains its upstream licenses (notably Wan2.2); the PF_Wan weights are
released under the MIT License.
