# HDFLIM

**HDFLIM (HyperDimensional computing with Frozen Language and Image Models)** — a framework for cross-modal alignment and image captioning that keeps pretrained vision and language models fully frozen using Hyperdimensional computing.

HD Computing is A.K.A: Binary Splatter Codes (BSC), Vector Symbolic Architectures (VSA), Holographic Reduced Representations (HRR). .... and many more

---

## Overview

HDFLIM establishes cross-modal mappings while keeping both vision and language models entirely frozen. Unimodal embeddings are projected into a shared hyperdimensional space, where lightweight symbolic operations — **binding**, **bundling**, and **similarity-based retrieval** — construct associative cross-modal representations in a single pass over the data. Caption generation emerges from HD memory retrieval rather than iterative gradient-based optimization.

It is highly recommended to explore the demo.ipynb notebook get familar with things. notebook to get familiar with the framework. Please download the required files from the [link](https://huggingface.co/adalvi/HDFLIM/tree/main) before running the demo or evals.

### Key Ideas

- **Frozen encoders**: Both the vision model (DINOv3) and LLM (Qwen3) are frozen — no fine-tuning required (even while learning/training).
- **HD projection via LSH**: Random LSH matrices project continuous hidden states from vison and LLM into binary HD vectors (50,0000 dimension space).
- **CLIP-guided sampling**: A `CLIPSemanticSampler` blends HDFLIM model probabilities with CLIP image-text alignment scores for more semantically grounded captions.
- **Efficient HD logits**: `HDLogitsComputer` uses pre-allocated GPU buffers and packed boolean tensors for fast vocabulary scoring.

---

## Inference Architecture

```
Image ──► DINOv3.txt (frozen) ──► HD Projection ───────────────┐
                                  (img_LSH_matrix)             │
                                                          HD Binding────────┐
                                                               │            │
Caption Prefix ──► Qwen3 (frozen) ──► HD Projection ───────────┘            │
                       ▲              (LM_LSH_matrix)                       │
                       │                                                    ▼
                       │                                        HD Logits Computation
                       │                                                    │
                       │                                                    ▼
                       │                                       CLIP-Guided Sampling
                       │                                                    │
                       │                                                    ▼
                       └──────────────── New Token ◄── Token Prediction ────┘
                         (appended to prefix,loop 
                          until max_len or EOS)
```

Generation is **autoregressive**: at each step the caption prefix (including all previously predicted tokens) is re-fed into Qwen3, its HD projection is binded with the image HD vector, HD logits are computed over the vocabulary, CLIP-guided sampling selects the next token, and that token is appended to the prefix for the next iteration.

### Components

| File | Description |
|---|---|
| `VisionModel_utils.py` | `FrozenVisionModel_Encoding` — wraps DINOv3 with HD projection |
| `LangModel_utils.py` | `FrozenLanguageModel_Encoding` — wraps Qwen3 with HD projection |
| `semantic_clip.py` | `CLIPSemanticSampler` — CLIP-guided token sampling with temperature, topk, top-p nucleus sampling, repeat penalty |
| `transform_inference.py` | Image preprocessing pipeline with aspect-ratio-aware resizing |
| `HD_eval.py` | Main evaluation script |
| `demo_notebook.ipynb` | Demo notebook |

---

## Hardware Requirements

| Resource | Recommended |
|---|---|
| GPU | NVIDIA A100 40GB |
| CPU RAM | 80–100 GB |

An A100 40GB GPU is required to run the models comfortably. 80–100 GB of CPU memory is recommended for efficient operation, particularly when loading large model weights and HD matrices.

## PCKGs/Models Requirements

- PyTorch (CUDA approperiate versions compatible with DINOv3 and Qwen3)
- `transformers`, `huggingface_hub`
- `torchvision`
- `Pillow`, `numpy`, `pandas`, `tqdm`
- **DINOv3** model weights — must be downloaded separately from Meta's GitHub: [github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- **Qwen3** (or compatible) language model weights from HuggingFace

> ⚠️ **VisionModel_utils.py requires DINOv3 with CLIP encoder model weights to be downloaded separately through Meta's website and GitHub repo.
> Make sure to change the path of the model weights in the torch.hub.load(...) line in VisionModel_utils.py**

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Abhishek-Dalvi410/HDFLIM.git
cd HDFLIM
```

### 2. Download DINOv3 weights

Follow the instructions at [https://github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) to obtain:
- `dinov3_vitl16_pretrain_lvd1689m-xyz.pth`
- `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-xyz.pth`

### 3. Download HD Matrices & Prototypes

Download the following pretrained `.pt` files and place them in `saved_HD_mats/`:

#### HD Projection Matrices

| File | Description | Download |
|---|---|---|
| `LM_LSH_matrix.pt` | LSH projection matrix for the language model | [Download](https://huggingface.co/adalvi/HDFLIM/tree/main) |
| `img_LSH_matrix.pt` | LSH projection matrix for the vision model | [Download](https://huggingface.co/adalvi/HDFLIM/tree/main) |
| `img_pos_HD.pt` | Positional HD vectors for image patches | [Download](https://huggingface.co/adalvi/HDFLIM/tree/main) |

#### Vocabulary HD Prototypes

Prototypes are precomputed HD vectors over the vocabulary used during HD logits computation. Choose the one that matches your use case:

| File | Description | Download |
|---|---|---|
| `vocab_HD_packed_COCO.pt` | Prototypes learned on the COCO dataset | [Download](https://huggingface.co/adalvi/HDFLIM/tree/main) |
| `vocab_HD_packed_13M.pt` | Prototypes learned on a 13M PixelProse image-caption dataset | [Download](https://huggingface.co/adalvi/HDFLIM/tree/main) |

> `vocab_HD_packed_COCO` is learned using for COCO-Karpathy train split; `vocab_HD_packed_13M` is learned using PixelProse Dataset.

Update the paths in `VisionModel_utils.py` and `LangModel_utils.py` to point to these files.

### 4. (Optional) HuggingFace Login

If using gated models, uncomment and add your token in `LangModel_utils.py`:

```python
# login("your_token_here")
```

---

## Usage

### Vision Encoder

```python
from VisionModel_utils import FrozenVisionModel_Encoding

vision_model = FrozenVisionModel_Encoding(
    device='cuda:0',
    HD_dim_size=50000,
    img_LSH_matrix_path='saved_HD_mats/img_LSH_matrix.pt',
    img_pos_HD_path='saved_HD_mats/img_pos_HD.pt'
)

hidden, class_tokens = vision_model.get_h_img(image_tensor)
img_hd_vec = vision_model.get_img_HD_vec(hidden)
```

### Language Encoder

```python
from LangModel_utils import FrozenLanguageModel_Encoding

lang_model = FrozenLanguageModel_Encoding(
    device='cuda:0',
    model_name='Qwen/Qwen3-4B',
    HD_dim_size=50000,
    LM_LSH_matrix_path='saved_HD_mats/LM_LSH_matrix.pt'
)

token_ids, hidden_caption = lang_model.get_h_caption(["A cat sitting on a mat"])
caption_hd_vec = lang_model.get_caption_HD_vec(hidden_caption)
```

### Image Transforms

```python
from transform_inference import make_transform

transform = make_transform(resize_size=512, crop_size=512)
image_tensor = transform(pil_image)
```

The transform pipeline intelligently handles aspect ratio: images with a high aspect ratio are resized to fit and then center-cropped, while near-square images are simply resized. You can experiment with different configurations — performance varies per use case. In the reported eval, its simply resize shortest to 512 and then centre crop to 512x512.

### CLIP-Guided Sampling

```python
from semantic_clip import CLIPSemanticSampler

sampler = CLIPSemanticSampler(lm_tokenizer, clip_model, clip_tokenizer)

next_token = sampler.sample_next_token(
    logits=logits,
    generated_tokens=generated_ids,
    clip_image_features=clip_img_feats,
    temperature=0.8,
    top_k=20,
    top_p=0.9,
    clip_weight=0.5
)
```

### Example for running evals (no batched inference possbile as  for now)

```bash
python HD_eval.py \
    --task coco \
    --aspect_ratio_threshold 1.0 \
    --HD_vocab_path /storage/group/vuh14/default/Abhishek_files/dinov3txt_qwen3/saved_HD_mats/vocab_HD_packed_13M.dat \
    --fixed_temp 1.0 \
    #-----
    --use_LLM_paraphraser True \
    #---- If caption is sort of incomplete, caption is fed into Frozen LLM with prompt to beautify.
    --top_k 80 \
    --caption_size 15 \
    --window_length 0 \
    --clip_weight 0.5 \
    --path_to_nocaps_val_images /scratch/abd5811/nocaps_val/val_by_id \
    --path_to_flickr_test_images /storage/work/abd5811/Flickr30k_KP_test_split/flickr30k_KP_test
```

---

## Notes

- `HDLogitsComputer` pre-allocates GPU buffers reused across all inference calls to minimize memory allocation overhead.
- The `pooling` parameter in `HDLogitsComputer` supports `'max'` or `'sum'` (not used sum in evals but optional kept for future work).
- Train/Learn Code is yet to be released!
- Will be uploading download links for LSH and prototypes (prototype files are pretty large).
- Experiments in this work were performed on the Pennsylvania State University’s Institute for Computational and Data Sciences’ ROAR supercomputer.
- ROAR supercomputer is a SLURM HPC cluster and it has infiniband storage, so disk(np.memmap) read/wrties are pretty quick.
- For `HDLogitsComputer` There is a possibility of implementing this with Triton or CUDA Kernels to optimize memory and performance. The LUT (Look-Up Table) for popcount followed by sum (highlighted with arrow comment in the demo code) is the main memory bottleneck.

---
## If any questions please feel free to reach out at abd5811@psu.edu :)

P.S I’m terrible at naming variables, functions and classes in a mnemonic way.

## License

See [LICENSE](LICENSE) for details.
