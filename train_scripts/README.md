# HDFLIM — Training (`train/`)

Hyperdimensional learning pipeline that binds frozen vision (DINOv3) and language (Qwen3-4B) model representations into a shared HD space via image-caption pairs.

## Repository Layout

```
HDFLIM/
├── VisionModel_utils.py      # Frozen vision encoder (DINOv3) + HD projection
├── LangModel_utils.py         # Frozen language encoder (Qwen3-4B) + HD projection
└── train/
    ├── train_HD.py            # Main training script (this README)
    ├── train_dataloader.py    # WebDataset loader with caption preprocessing
    └── README.md
```

`train_HD.py` imports `VisionModel_utils` and `LangModel_utils` from the **parent directory** automatically via `sys.path`. No manual `PYTHONPATH` setup is needed — just run from inside `train/`.

## Quick Start

```bash
cd train/

python train_HD.py \
  --shard-pattern "/path/to/shards/{00000..00049}.tar" \
  --save-dir "/path/to/saved_HD_mats" \
  --vocab-file "/path/to/saved_HD_mats/vocab_HD_dict.dat"
```

## Arguments

### Required

| Argument | Description |
|---|---|
| `--shard-pattern` | Brace-expansion pattern pointing to WebDataset `.tar` shards. Each shard must contain image files (`jpg`/`png`/`jpeg`) paired with `.txt` caption files. Example: `/data/shards/{00000..01639}.tar` |
| `--save-dir` | Directory where HD matrices are saved/loaded (`img_LSH_matrix.pt`, `img_pos_HD.pt`, `LM_LSH_matrix.pt`). Created automatically if it doesn't exist. |
| `--vocab-file` | Path for the vocabulary HD dictionary memory-mapped file (`.dat`). This is the main training output. Created automatically if it doesn't exist. |

### Optional

| Argument | Default | Description |
|---|---|---|
| `--batch-size` | `50` | Batch size for the dataloader. |
| `--caption-size` | `21` | Max caption length in tokens (after tokenization). |
| `--vision-hidden-dim` | `1024` | DINOv3 patch hidden state dimension. |
| `--vision-num-patches` | `1025` | DINOv3 number of patches (1024 + 1 class token). |
| `--language-hidden-dim` | `2560` | Qwen3-4B last hidden state dimension. |
| `--automodel-causal` | `False` | Pass this flag to load the language model with `AutoModelForCausalLM` instead of `AutoModel`. |

### Fixed Constant

The HD vector dimension is hardcoded to **50,000** (`HD_DIM = 50000` in `train_HD.py`). This value is shared across all matrices and encoders.

## ⚠️ Disk Space Requirements

The vocabulary HD dictionary is a **memory-mapped `int32` array** with shape `(caption_size, vocab_size, HD_dim_size)`. With the Qwen3-4B tokenizer (`vocab_size ≈ 151,936`) this gets very large:

| `--caption-size` | Vocab File Size |
|---|---|
| 21 (default) | **~594 GB** |
| 43 | **~1.2 TB** |

The HD matrices saved in `--save-dir` add another **~440 MB** total:
- `img_LSH_matrix.pt` — `(1024, 50000)` bfloat16 → ~97 MB
- `img_pos_HD.pt` — `(1, 1025, 50000)` int16 → ~98 MB
- `LM_LSH_matrix.pt` — `(2560, 50000)` bfloat16 → ~244 MB

**Make sure `--vocab-file` points to a filesystem with sufficient free space.** On HPC clusters, use scratch or group storage — not your home directory.

## ⚠️ Paths You Must Update

The following hardcoded paths in **`VisionModel_utils.py`** are specific to the original development environment and **must be changed** before running:

1. **DINOv3 local repo path** — the first argument to `torch.hub.load()`:
   ```
   /storage/group/vuh14/default/Abhishek_files/dinov3txt_qwen3/dinov3_repo/dinov3
   ```
   → Change to wherever you cloned the [DINOv3 repo](https://github.com/facebookresearch/dinov3).

2. **DINOv3 backbone weights** — the `backbone_weights` argument:
   ```
   /storage/group/vuh14/default/Abhishek_files/dinov3txt_qwen3/meta_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
   ```
   → Change to your local path for the downloaded DINOv3 pretrained weights.

3. **DINOv3 text encoder weights** — the `weights` argument:
   ```
   meta_weights/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth
   ```
   → This is a relative path; ensure it resolves correctly from your working directory, or change to an absolute path.

The default paths in `LangModel_utils.py` and `VisionModel_utils.py` for the HD matrix files (`LM_LSH_matrix_path`, `img_LSH_matrix_path`, `img_pos_HD_path`) are **overridden by `train_HD.py`** via `--save-dir`, so those do not need manual editing.

## Data Format

Training data must be in [WebDataset](https://github.com/webdataset/webdataset) `.tar` format. Each sample in a shard should contain:
- An image file: `.jpg`, `.png`, or `.jpeg`
- A caption file: `.txt`

Images are resized (shortest side to 512px) and center-cropped to 512×512. Captions are lowercased, re-capitalized after periods, and prefixed with `"This image shows "` before tokenization.

## GPU and Memory

- **GPU**: A CUDA GPU is expected. The vision model runs with `torch.autocast('cuda', dtype=torch.bfloat16)` and is moved to CUDA explicitly. The language model also loads in bfloat16.
- **RAM**: The vocabulary memmap is disk-backed, so RAM usage stays manageable even for the ~594 GB file. However, the frozen DINOv3 and Qwen3-4B models both need to fit in GPU memory simultaneously.

## Resumability

- **HD matrices**: If `--save-dir` already contains the `.pt` files from a previous run, they are loaded rather than regenerated. This preserves the random projections across runs.
- **Vocab dictionary**: If `--vocab-file` already exists, training **accumulates on top of existing values** (the memmap is opened in `r+` mode). This means you can resume from a different shard range without losing prior progress. However, re-running the same shards will double-count those samples.
- **Shard tracking**: The current shard URL is printed every 20 batches. If a run fails, note the last shard printed and adjust `--shard-pattern` to resume from there.

## Hugging Face Authentication

If the Qwen3-4B model requires gated access, uncomment and fill in the `login(...)` call at the top of `LangModel_utils.py`, or run `huggingface-cli login` before training.