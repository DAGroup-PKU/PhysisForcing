# Bridge-v2 subset (example) — Cosmos3 SFT dataset

A tiny 4-clip sample of the public
[nvidia/bridge-v2-subset-synthetic-captions](https://huggingface.co/datasets/nvidia/bridge-v2-subset-synthetic-captions)
dataset, shipped so the JSONL schema and directory layout are self-explanatory
and the launch scripts have something to run against out of the box.

## Layout

    sft_dataset_bridge/
    └── train/
        ├── video_dataset_file.jsonl   # one JSON record per clip
        └── videos/*.mp4               # 256x256 clips referenced by `vision_path`

## JSONL schema (one object per line)

| field | meaning |
| ----- | ------- |
| `uuid` | clip id (matches the mp4 filename) |
| `duration`, `width`, `height` | clip metadata |
| `vision_path` | mp4 path relative to `sft_dataset_bridge/train/` |
| `t2w_windows[]` | list of `{start_frame, end_frame, temporal_interval, caption}` |

See [docs/dataset_jsonl.md](../../../docs/dataset_jsonl.md) for the full format
and how to caption your own videos.

## Full dataset

This 4-clip copy is only for smoke tests. Download the full dataset into this
same path for real runs:

```shell
uvx hf@latest download --repo-type dataset nvidia/bridge-v2-subset-synthetic-captions \
    --revision 46468e12ac0dd36901e9e3240d4fc7620942b5d7 \
    --local-dir examples/data/bridge-v2-subset-synthetic-captions --quiet
```
