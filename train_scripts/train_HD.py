import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path for importing VisionModel_utils and LangModel_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from tqdm import tqdm
import faulthandler

from VisionModel_utils import FrozenVisionModel_Encoding
from LangModel_utils import FrozenLanguageModel_Encoding
from train_dataloader import create_loader

faulthandler.enable()


def create_vocab_HD_file(caption_size, vocab_size, HD_dim_size, filename):
    """Create or load a memory-mapped vocabulary HD dictionary file."""
    shape = (caption_size, vocab_size, HD_dim_size)
    dtype = np.int32

    total_elements = caption_size * vocab_size * HD_dim_size
    memory_size_mb = (total_elements * np.dtype(dtype).itemsize) / (1024**2)
    memory_size_gb = memory_size_mb / 1024

    print("=" * 70)
    print("VOCABULARY HYPERDIMENSIONAL DICTIONARY INITIALIZATION")
    print("=" * 70)
    print(f"\n  Caption size:    {caption_size}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  HD dimension:    {HD_dim_size}")
    print(f"  Shape:           {shape}")
    print(f"  Memory:          ~{memory_size_gb:.2f} GB" if memory_size_gb >= 1 else f"  Memory:          ~{memory_size_mb:.2f} MB")
    print(f"  File:            {filename}")

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if os.path.exists(filename):
        print(f"  Status: File exists — loading in r+ mode")
        vocab_HD = np.memmap(filename, dtype=dtype, mode='r+', shape=shape)
    else:
        print(f"  Status: Creating new file...")
        vocab_HD = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)

    vocab_HD.flush()
    del vocab_HD
    print("=" * 70)


def _init_matrix(path, shape, dtype, description, binary=False):
    """Helper to create or load a saved torch matrix."""
    print(f"\n  [{description}]")
    print(f"    Path:  {path}")
    print(f"    Shape: {shape}  Dtype: {dtype}")

    if os.path.exists(path):
        print(f"    Status: Already exists — skipping")
        return

    print(f"    Status: Creating...")
    if binary:
        mat = (2 * torch.randint(0, 2, size=shape, dtype=dtype)) - 1
    else:
        mat = torch.randn(size=shape, dtype=dtype)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(mat, path)
    print(f"    Saved ({mat.numel() * mat.element_size() / (1024**2):.1f} MB)")
    del mat


def init_HD_matrices(save_dir, HD_dim_size,
                     vision_hidden_dim=1024, vision_num_patches=1025,
                     language_hidden_dim=2560):
    """Initialise LSH and positional HD matrices for both vision and language models."""
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("HD MATRIX INITIALIZATION")
    print("=" * 70)
    print(f"  HD dimension: {HD_dim_size}")
    print(f"  Save dir:     {save_dir}")

    # Vision model matrices
    _init_matrix(
        os.path.join(save_dir, "img_LSH_matrix.pt"),
        (vision_hidden_dim, HD_dim_size),
        torch.bfloat16,
        "Image LSH Matrix",
    )
    _init_matrix(
        os.path.join(save_dir, "img_pos_HD.pt"),
        (1, vision_num_patches, HD_dim_size),
        torch.int16,
        "Image Position HD",
        binary=True,
    )

    # Language model matrix
    _init_matrix(
        os.path.join(save_dir, "LM_LSH_matrix.pt"),
        (language_hidden_dim, HD_dim_size),
        torch.bfloat16,
        "Language Model LSH Matrix",
    )

    print("\n" + "=" * 70)


def learn_HD(shard_pattern, vision_encoders, caption_encoders, vocab_file_name,
             batch_size=50):
    """Main training loop: encode images + captions into HD space and accumulate into vocab dictionary."""
    print(f"\nCreating DataLoader from: {shard_pattern}")
    dataloader = create_loader(
        shard_pattern=shard_pattern,
        batch_size=batch_size,
        resize_size=512,
        crop_size=512,
        aspect_ratio_threshold=1.1,
        num_workers=0,
        shuffle=False,
    )
    print(f"  Batch size: {batch_size}")

    # Vocab HD memmap
    shape = (caption_encoders.caption_size, caption_encoders.vocab_size, caption_encoders.HD_dim_size)
    create_vocab_HD_file(caption_encoders.caption_size, caption_encoders.vocab_size,
                         caption_encoders.HD_dim_size, vocab_file_name)
    vocab_HD = np.memmap(vocab_file_name, dtype=np.int32, mode='r+', shape=shape)

    print("Starting training...\n")

    for batch_num, (imgs, img_captions, shard_url) in enumerate(
        tqdm(dataloader, desc="Batches", dynamic_ncols=True), 1
    ):
        if batch_num % 20 == 0:
            tqdm.write(f"Batch {batch_num} — shard: {shard_url}")

        n = len(img_captions)

        # Vision encoding → HD
        hidden_imgs, _ = vision_encoders.get_h_img(imgs)
        del imgs
        HD_imgs = vision_encoders.get_img_HD_vec(hidden_imgs)
        del hidden_imgs

        # Caption encoding → HD (drop 2 prefix tokens for "This image")
        tokenized, hidden_caps = caption_encoders.get_h_caption(img_captions)
        tokenized = tokenized[:, 2:]
        hidden_caps = hidden_caps[:, 2:, :]
        del img_captions
        HD_caps = caption_encoders.get_caption_HD_vec(hidden_caps)
        del hidden_caps

        # Bind image and caption HD vectors
        HD_combined = (HD_imgs.unsqueeze(1) * HD_caps).to(torch.int32).cpu().numpy()
        del HD_imgs, HD_caps

        # Accumulate into vocab dictionary
        for i in tqdm(range(n), desc="Images", position=1, leave=False, dynamic_ncols=True):
            tokens = tokenized[i]
            hd_row = HD_combined[i]
            for j in range(tokens.shape[0] - 1):
                tok = tokens[j + 1].item()
                vocab_HD[j, tok, :] += hd_row[j]
                if tok == caption_encoders.eos_id:
                    break

        vocab_HD.flush()
        del HD_combined, tokenized

    del vocab_HD


def parse_args():
    p = argparse.ArgumentParser(description="HD learning pipeline for image-caption binding")

    # Paths
    p.add_argument("--shard-pattern", required=True,
                    help="Glob/brace pattern for webdataset tar shards, e.g. /data/shards/{00000..00049}.tar")
    p.add_argument("--save-dir", required=True,
                    help="Directory to save/load HD matrices (LSH, positional HD)")
    p.add_argument("--vocab-file", required=True,
                    help="Path for the int32 vocab HD dictionary memmap (.dat)")

    # HD / model dimensions
    p.add_argument("--hd-dim", type=int, default=50000,
                    help="Hyperdimensional vector dimension (default: 50000)")
    p.add_argument("--vision-hidden-dim", type=int, default=1024,
                    help="Vision model last hidden state dim (default: 1024)")
    p.add_argument("--vision-num-patches", type=int, default=1025,
                    help="Vision model number of patches incl. class token (default: 1025)")
    p.add_argument("--language-hidden-dim", type=int, default=2560,
                    help="Language model last hidden state dim (default: 2560)")
    p.add_argument("--caption-size", type=int, default=21,
                    help="Max caption token length (default: 21)")

    # Training
    p.add_argument("--batch-size", type=int, default=50,
                    help="Batch size for dataloader (default: 50)")

    # Model flags
    p.add_argument("--automodel-causal", action="store_true",
                    help="Use AutoModelForCausalLM flag in language model")

    return p.parse_args()


def run():
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Initialise HD matrices
    init_HD_matrices(
        save_dir=args.save_dir,
        HD_dim_size=args.hd_dim,
        vision_hidden_dim=args.vision_hidden_dim,
        vision_num_patches=args.vision_num_patches,
        language_hidden_dim=args.language_hidden_dim,
    )

    # Frozen model encoders
    F_VM = FrozenVisionModel_Encoding(device=device)
    F_LM = FrozenLanguageModel_Encoding(
        device=device,
        AutoModelForCausalLM_flag=args.automodel_causal,
        caption_size=args.caption_size,
        LM_LSH_matrix_path=os.path.join(args.save_dir, "LM_LSH_matrix.pt"),
    )

    # Train
    learn_HD(
        shard_pattern=args.shard_pattern,
        vision_encoders=F_VM,
        caption_encoders=F_LM,
        vocab_file_name=args.vocab_file,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    run()