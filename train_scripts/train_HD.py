import numpy as np
from VisionModel_utils import FrozenVisionModel_Encoding
from LangModel_utils import FrozenLanguageModel_Encoding
from train_dataloader import create_loader
import torch
from tqdm import tqdm
import os
from pathlib import Path
import faulthandler
faulthandler.enable()


def create_vocab_HD_file(caption_size, vocab_size, HD_dim_size, filename):
    shape = (caption_size, vocab_size, HD_dim_size)
    dtype = np.int32
    
    print("=" * 70)
    print("VOCABULARY HYPERDIMENSIONAL DICTIONARY INITIALIZATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Caption size: {caption_size}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  HD dimension size: {HD_dim_size}")
    print(f"  Data type: {dtype.__name__}")
    print(f"\nFile path: {filename}")
    print(f"Target shape: {shape}")
    
    # Calculate memory size
    total_elements = caption_size * vocab_size * HD_dim_size
    memory_size_mb = (total_elements * np.dtype(dtype).itemsize) / (1024**2)
    memory_size_gb = memory_size_mb / 1024
    
    if memory_size_gb >= 1:
        print(f"Memory size: ~{memory_size_gb:.2f} GB")
    else:
        print(f"Memory size: ~{memory_size_mb:.2f} MB")
    
    print("=" * 70)
    
    # Check and create memory-mapped file
    print(f"\n[Vocab HD Dictionary Memory-Mapped File]")
    
    if os.path.exists(filename):
        print(f"    Status: ✓ File already exists - LOADING in read/write mode")
        vocab_HD_dict_memmap = np.memmap(filename, dtype=dtype, mode='r+', shape=shape)
        print(f"    Existing file shape: {vocab_HD_dict_memmap.shape}")
        print(f"    Existing file dtype: {vocab_HD_dict_memmap.dtype}")
    else:
        print(f"    Status: ✗ File does not exist - CREATING...")
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        vocab_HD_dict_memmap = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
        print(f"    Status: ✓ Successfully created memory-mapped file")
    
    vocab_HD_dict_memmap.flush()
    print(f"    Action: Flushed changes to disk")
    del vocab_HD_dict_memmap
    print(f"    Action: Closed memory-mapped file")
    
    print("\n" + "=" * 70)
    print("PROCESS COMPLETED")
    print("=" * 70)


def learn_HD(shard_pattern, vision_encoders, caption_encoders, vocab_file_name):
    batch_size = 50
    
    # Create DataLoader using the provided function
    # Path to your shards
    print("Creating DataLoader....")
    print("Path to webdataset shards: ", shard_pattern)
    # Create DataLoader
    dataloader = create_loader(
        shard_pattern=shard_pattern,
        batch_size=batch_size,
        resize_size=512,
        crop_size=512,
        aspect_ratio_threshold = 1.1, # For training this seems good as not much squeezing
        num_workers=0,  # adjust for your machine
        shuffle=False
    )
    
    print("Batch Size = ", batch_size)
    
    # Initialize vocab HD file
    shape = (caption_encoders.caption_size, caption_encoders.vocab_size, caption_encoders.HD_dim_size)
    create_vocab_HD_file(caption_encoders.caption_size, caption_encoders.vocab_size, caption_encoders.HD_dim_size, vocab_file_name)
    vocab_HD_dict_memmap = np.memmap(vocab_file_name, dtype=np.int32, mode='r+', shape=shape)
    
    print("Now Starting Training.....")
    
    # Process batches using the DataLoader
    for batch_num, (imgs, img_captions, shard_url) in enumerate(tqdm(dataloader, desc="Processing Batches", position=0, dynamic_ncols=True), 1):
        
        # Print shard info every 10 batches or when it changes
        # Kept this as one can resume from the shard if error occurs
        if batch_num % 20 == 0:
            tqdm.write(f"\nBatch {batch_num} - Current shard: {shard_url}")
        
        # img: torch.Tensor of shape (batch_size, 3, 512, 512)
        # img_captions: list of strings

        len_img_captions = len(img_captions)
        
        hidden_batches_imgs, _ = vision_encoders.get_h_img(imgs)

        del imgs
        
        if batch_num < 10:
            print("Checks to see how img captions look like")
            print(img_captions[batch_num])

        batch_input_tokenized, hidden_batches_captions = caption_encoders.get_h_caption(img_captions)
        
        # Drop the first 2 tokens (prefix tokens for "This image")
        batch_input_tokenized = batch_input_tokenized[:, 2:]  # Drop first 2 tokens
        hidden_batches_captions = hidden_batches_captions[:, 2:, :]  # Drop first 2 positions

        del img_captions

        HD_batches_imgs = vision_encoders.get_img_HD_vec(hidden_batches_imgs)

        del hidden_batches_imgs

        HD_batches_captions = caption_encoders.get_caption_HD_vec(hidden_batches_captions)

        del hidden_batches_captions

        HD_batch_img_cap = HD_batches_imgs.unsqueeze(1) * HD_batches_captions

        HD_batch_img_cap = HD_batch_img_cap.to(torch.int32)

        HD_batch_img_cap = HD_batch_img_cap.cpu()

        HD_batch_img_cap = HD_batch_img_cap.numpy()

        del HD_batches_imgs
        del HD_batches_captions
        
        if batch_num <10:
            print("working till here for 10 batches")

        for i in tqdm(range(len_img_captions), desc="Processing each batch Image", position=1, leave=False, dynamic_ncols=True):
            img_cap_HD = HD_batch_img_cap[i]

            cap_tokenized = batch_input_tokenized[i]

            for j in range(cap_tokenized.shape[0]-1):

                token = cap_tokenized[j+1].item()
                vocab_HD_dict_memmap[j,token,:] = vocab_HD_dict_memmap[j,token,:] + img_cap_HD[j]

                if token == caption_encoders.eos_id:
                  break

        vocab_HD_dict_memmap.flush()
        del HD_batch_img_cap
        del batch_input_tokenized


    del vocab_HD_dict_memmap


def run():
    #### HD DIMENSION SIZE #####
    HD_dim_size = 50000
    #----------------------END--------------------#
    ########### CUDA DEVICE SETUP ###########
    device = "cpu"
    if torch.cuda.is_available():
        # Get the current CUDA device
        device = torch.device("cuda")
        print("CUDA is available. \n")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU. \n")
    #----------------------END--------------------#
    
    ########### IMAGE/VISION MODEL LSH AND HD SETUP ###########
    """
    Setting Up LSH matrix for IMAGE/VISION MODEL and also position HD matrix for tokens.
    Also, printing out info about these matrices.
    """
    
    # Configuration
    Vision_model_last_hidden_state_dim = 1024 # DINOv3.Txt patch hidden state
    Vision_model_num_patches = 1025  # DINOv3 TXT patches plus class patch (1024+1)
    
    # Define save paths
    save_dir = "/storage/group/vuh14/default/Abhishek_files/dinov3txt_qwen3/saved_HD_mats"
    img_LSH_matrix_path = os.path.join(save_dir, "img_LSH_matrix.pt")
    img_pos_HD_path = os.path.join(save_dir, "img_pos_HD.pt")
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("HYPERDIMENSIONAL MATRIX INITIALIZATION FOR FROZEN IMAGE/VISION MODEL")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  HD dimension size: {HD_dim_size}")
    print(f"  Vision model hidden state dim: {Vision_model_last_hidden_state_dim}")
    print(f"  Number of patches: {Vision_model_num_patches}")
    print(f"\nSave directory: {save_dir}")
    print("=" * 70)
    
    # Check and create img_LSH_matrix
    print(f"\n[1] Image LSH Matrix")
    print(f"    Path: {img_LSH_matrix_path}")
    print(f"    Shape: ({Vision_model_last_hidden_state_dim}, {HD_dim_size})")
    print(f"    Dtype: torch.bfloat16")
    
    if os.path.exists(img_LSH_matrix_path):
        print(f"    Status: ✓ File already exists - SKIPPING creation")
        # Load and print info about existing file
        existing_matrix = torch.load(img_LSH_matrix_path)
        print(f"    Existing file shape: {existing_matrix.shape}")
        print(f"    Existing file dtype: {existing_matrix.dtype}")
        del existing_matrix
    else:
        print(f"    Status: ✗ File does not exist - CREATING...")
        img_LSH_matrix = torch.randn(
            size=(Vision_model_last_hidden_state_dim, HD_dim_size), 
            dtype=torch.bfloat16
        )
        torch.save(img_LSH_matrix, img_LSH_matrix_path)
        print(f"    Status: ✓ Successfully created and saved")
        print(f"    Memory size: ~{(img_LSH_matrix.numel() * 2) / (1024**2):.2f} MB")
        del img_LSH_matrix
    
    # Check and create img_pos_HD
    print(f"\n[2] Image Position HD Vectors")
    print(f"    Path: {img_pos_HD_path}")
    print(f"    Shape: (1, {Vision_model_num_patches}, {HD_dim_size})")
    print(f"    Dtype: torch.int16")
    print(f"    Values: Binary {-1, +1}")
    
    if os.path.exists(img_pos_HD_path):
        print(f"    Status: ✓ File already exists - SKIPPING creation")
        # Load and print info about existing file
        existing_pos = torch.load(img_pos_HD_path)
        print(f"    Existing file shape: {existing_pos.shape}")
        print(f"    Existing file dtype: {existing_pos.dtype}")
        print(f"    Value range: [{existing_pos.min().item()}, {existing_pos.max().item()}]")
        del existing_pos
    else:
        print(f"    Status: ✗ File does not exist - CREATING...")
        img_pos_HD = (2 * torch.randint(
            0, 2, 
            size=(1, Vision_model_num_patches, HD_dim_size), 
            dtype=torch.int16
        )) - 1
        torch.save(img_pos_HD, img_pos_HD_path)
        print(f"    Status: ✓ Successfully created and saved")
        print(f"    Memory size: ~{(img_pos_HD.numel() * 2) / (1024**2):.2f} MB")
        print(f"    Unique values: {torch.unique(img_pos_HD).tolist()}")
        del img_pos_HD
    
    print("\n" + "=" * 70)
    print("PROCESS COMPLETED")
    print("=" * 70)
    
    #----------------------END--------------------#
    
    ########### LANGUAGE MODEL LSH AND HD SETUP ###########
    """
    Setting Up LSH matrix for LANGUAGE MODEL.
    Also, printing out info about this matrices.
    """
    
    # Configuration
    Language_model_last_hidden_state_dim = 2560  # LLM Last Hidden State
    
    # Define save paths
    save_dir = "/storage/group/vuh14/default/Abhishek_files/dinov3txt_qwen3/saved_HD_mats"
    LM_LSH_matrix_path = os.path.join(save_dir, "LM_LSH_matrix.pt")
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("HYPERDIMENSIONAL MATRIX INITIALIZATION FOR FROZEN LANGUAGE MODEL")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  HD dimension size: {HD_dim_size}")
    print(f"  Language model hidden state dim: {Language_model_last_hidden_state_dim}")
    print(f"\nSave directory: {save_dir}")
    print("=" * 70)
    
    # Check and create LM_LSH_matrix
    print(f"\n[1] Language Model LSH Matrix")
    print(f"    Path: {LM_LSH_matrix_path}")
    print(f"    Shape: ({Language_model_last_hidden_state_dim}, {HD_dim_size})")
    print(f"    Dtype: torch.bfloat16")
    
    if os.path.exists(LM_LSH_matrix_path):
        print(f"    Status: ✓ File already exists - SKIPPING creation")
        # Load and print info about existing file
        existing_matrix = torch.load(LM_LSH_matrix_path)
        print(f"    Existing file shape: {existing_matrix.shape}")
        print(f"    Existing file dtype: {existing_matrix.dtype}")
        print(f"    Memory size: ~{(existing_matrix.numel() * 2) / (1024**2):.2f} MB")
        del existing_matrix
    else:
        print(f"    Status: ✗ File does not exist - CREATING...")
        LM_LSH_matrix = torch.randn(
            size=(Language_model_last_hidden_state_dim, HD_dim_size), 
            dtype=torch.bfloat16
        )
        torch.save(LM_LSH_matrix, LM_LSH_matrix_path)
        print(f"    Status: ✓ Successfully created and saved")
        print(f"    Memory size: ~{(LM_LSH_matrix.numel() * 2) / (1024**2):.2f} MB")
        del LM_LSH_matrix
    
    print("\n" + "=" * 70)
    print("PROCESS COMPLETED")
    print("=" * 70)
    
    #----------------------END--------------------#
    F_VM_object= FrozenVisionModel_Encoding(device = device)
    F_LM_object = FrozenLanguageModel_Encoding(device = device, use_COCO_finetuned_model=False, AutoModelForCausalLM_flag=False, caption_size=21) # use_COCO_finetuned_model=True/False
    
    """
    Path to your shards
    Contains all tar files according to the pattern {00000..01639}.tar
    Tar files contain images and captions in webdataset format (https://github.com/webdataset/webdataset)
    """
    # tar_files = "/storage/group/vuh14/default/Abhishek_files/pix2prose_512/{01330..01639}.tar"
    # tar_files = "/storage/group/vuh14/default/Abhishek_files/pix2prose_512/{00000..01639}.tar"
    tar_files = "/scratch/abd5811/COCO_imgs_512_wds_merged/{00000..00049}.tar"
    
    vocab_file_name_path = "/storage/group/vuh14/default/Abhishek_files/clip_qwen3/saved_HD_mats/vocab_HD_dict.dat" # path to int32 vocab file to be created with shape of vocab file = (caption_size, vocab_size, HD_dim_size)
    
    learn_HD(shard_pattern = tar_files, vision_encoders=F_VM_object, caption_encoders=F_LM_object, vocab_file_name=vocab_file_name_path)


if __name__ == "__main__":
   run()