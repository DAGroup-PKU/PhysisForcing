# PF_Cosmos

Image-to-video (I2V) robot world model — the Cosmos3-Nano model from **PhysisForcing**
([paper](https://arxiv.org/abs/2606.28128) · [project](https://github.com/DAGroup-PKU/PhysisForcing)).
Given a first frame and a prompt describing the motion, it generates a short video
that continues from that frame.

Self-contained inference bundle: framework code + weights + example inputs.

```
.
├── cosmos_framework/        # inference code (built on NVIDIA/cosmos-framework)
├── checkpoints/PF_Cosmos/   # weights (safetensors) + VAE + config
├── inputs/robot_i2v/        # example inputs (JSON) + conditioning images
└── docs/                    # framework docs (setup, inference, ...)
```

## 1. Environment

From the recommended NGC base image `nvcr.io/nvidia/pytorch:25.09-py3`, run **from the
repository root**:

```shell
apt-get update
apt-get install -y --no-install-recommends curl ffmpeg git-lfs libx11-dev tree wget

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# CUDA 13.0 (recommended); for CUDA 12.8 use `--group=cu128-train`
uv sync --all-extras --group=cu130-train
source .venv/bin/activate && export LD_LIBRARY_PATH=
```

The empty `LD_LIBRARY_PATH` avoids a `torch._C` import error. Docker path, CUDA
variants, and troubleshooting are in [`docs/setup.md`](docs/setup.md).

## 2. Weights

Download the weights into `checkpoints/PF_Cosmos/` (not tracked in git):

```shell
huggingface-cli download DAGroup-PKU/PF_Cosmos --local-dir checkpoints/PF_Cosmos
```


## 3. Run inference

From the repository root (the first run also fetches the Qwen tokenizer from Hugging
Face, so keep network access; set `HF_TOKEN` if prompted):

```shell
torchrun --nproc-per-node=8 --master-port=50055 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  --no-guardrails \
  -i 'inputs/robot_i2v/*.json' \
  -o outputs/i2v \
  --checkpoint-path checkpoints/PF_Cosmos \
  --seed 0 --skip-invalid-samples
```

- `--parallelism-preset=latency` shards one sample across the N GPUs; use `throughput`
  to run independent samples in parallel. Single-GPU: `--nproc-per-node=1`.
- Outputs land at `outputs/i2v/<name>/vision.mp4`; re-running skips finished samples.
- Full sampling-argument reference: [`docs/inference.md`](docs/inference.md).

## 4. Input format

```json
{
  "model_mode": "image2video",
  "name": "example_001",
  "prompt": "A short description of the motion to generate ...",
  "vision_path": "images/example_001.jpg",
  "num_frames": 189,
  "fps": 24,
  "aspect_ratio": "16,9"
}
```

- `vision_path` resolves relative to the JSON file's own directory.
- `num_frames` must satisfy the VAE constraint `4k + 1` (e.g. 189, 193).
- `aspect_ratio` is `"W,H"`.

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

Framework code retains its upstream license (see [`LICENSE`](LICENSE)); the PF_Cosmos
weights are released under the MIT License.
