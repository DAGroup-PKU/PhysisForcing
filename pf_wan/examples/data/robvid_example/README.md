# Example training dataset (pf-wan)

A tiny 4-clip example showing the CSV manifest schema the training dataloader
(`project/data/video_i18n.py`, `VideoI18NParquetDataset`) expects. The clips are
public samples from BridgeData V2; swap in your own robot videos for real runs.

## Layout

    robvid_example/
    ├── metadata.csv        # one row per clip
    └── videos/*.mp4

## `metadata.csv` columns

The training config selects columns by name via `data.tos_url_key` (video path)
and `data.text_key` (caption). The shipped `configs/train/full_ft/*.jsonc` use
`video_path` + `short_caption`; the remaining columns are metadata the loader
ignores.

| column | used by loader | meaning |
| ------ | -------------- | ------- |
| `video_path` | yes (`tos_url_key`) | mp4 path, relative to the repo root (cv2 opens it) |
| `short_caption` | yes (`text_key`) | prompt used for training |
| `task_caption`, `detailed_caption` | | alternative caption columns |
| `frames`, `fps`, `height`, `width` | | clip metadata |
| `unique_hash`, `source_csv`, `source_row_idx`, `file_size_MB`, `verb` | | bookkeeping |

To use your own data, point `data.paths` at your CSV and make sure
`tos_url_key` / `text_key` match your column names.
