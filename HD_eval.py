#importing the libraries
import pandas as pd
import torch
from numpy import array
import numpy as np
import os
from PIL import Image
import requests
from tqdm import tqdm
from VisionModel_utils import FrozenVisionModel_Encoding
from LangModel_utils import FrozenLanguageModel_Encoding
from semantic_clip import CLIPSemanticSampler
import json
from transform_inference import make_transform
import argparse
from datetime import datetime
import time
from requests.exceptions import ConnectionError, RequestException

def download_image_with_retry(url, max_retries=5, initial_delay=1):
    """
    Download an image with retry logic and exponential backoff.
    
    Args:
        url: Image URL to download
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (will increase exponentially)
    
    Returns:
        PIL Image object
    """
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()  # Raise exception for bad status codes
            image = Image.open(response.raw).convert("RGB")
            return image
        except (ConnectionError, RequestException) as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Failed to download {url} after {max_retries} attempts")
                raise
            print(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff



def pack_boolean_tensor(token_hd_vec):
    """
    Convert a boolean tensor to packed uint8 format (NumPy fallback if torch.packbits is missing).
    Optimized for CPU.
    """
    token_hd_vec_uint8 = token_hd_vec.to(torch.uint8, copy=False)

    if hasattr(torch, "packbits"): # This was available in some PyTorch versions
        try:
            return torch.packbits(token_hd_vec_uint8, dim=-1)
        except Exception:
            pass  # fallback if torch.packbits fails (rare older builds)

    arr = token_hd_vec_uint8.numpy()
    packed = np.packbits(arr, axis=-1)
    return torch.from_numpy(packed)

class HDLogitsComputer:
    """Fast HD logits computation with pre-allocated buffers and max pooling"""
    def __init__(self, vocab_hd_packed, popcount_lut, device='cuda:0', pooling='max'):
        self.device = device
        self.vocab_hd_packed = vocab_hd_packed
        self.popcount_lut = popcount_lut
        self.vocab_size = vocab_hd_packed.shape[1]
        self.pooling = pooling  # 'max' or 'sum'
        
        # Pre-allocate buffers (reused across ALL calls)
        self.HD_logits_gpu = torch.zeros(self.vocab_size, device=device, dtype=torch.float32)
        self.HD_logits_cpu = torch.zeros(self.vocab_size, device='cuda:0', dtype=torch.float32)
        
    def compute(self, token_hd_vec, i, window_size=4, vocab_chunk_size=30000):
        start_pos = max(0, i)
        end_pos = min(self.vocab_hd_packed.shape[0], i + window_size + 1)
        window_len = end_pos - start_pos
        
        # Move token to GPU
        token_hd_vec = token_hd_vec.to(self.device, non_blocking=True)
        token_hd_expanded = token_hd_vec.unsqueeze(0).unsqueeze(0)
        
        # Reset buffer
        self.HD_logits_gpu.zero_()
        
        # Optimization: Load entire window once (window_size=4 means max 9 positions)
        if window_len <= 10:
            vocab_window = self.vocab_hd_packed[start_pos:end_pos].to(self.device, non_blocking=True)
            
            for v_start in range(0, self.vocab_size, vocab_chunk_size):
                v_end = min(v_start + vocab_chunk_size, self.vocab_size)
            
                if v_end == v_start:
                    continue  # nothing to process
            
                xor_result = torch.bitwise_xor(vocab_window[:, v_start:v_end, :], token_hd_expanded)
                hamming_distances = self.popcount_lut[xor_result.long()].sum(dim=2, dtype=torch.int32)
            
                if self.pooling == 'max':
                    self.HD_logits_gpu[v_start:v_end] = (50000 - hamming_distances).max(dim=0)[0].float()

                else:
                    # Sum pooling: sum similarities across window
                    self.HD_logits_gpu[v_start:v_end] = (50000 - hamming_distances).sum(dim=0, dtype=torch.float32)
            
            del vocab_window
        else:
            """"
            Currently Else part is not needed as window size is small
            Also, still working on this part so might not work when used.
            Additonally, should be very slow.... 
            """
            # Fallback for larger windows
            # Fallback for larger windows
            for v_start in range(0, self.vocab_size, vocab_chunk_size):
                v_end = min(v_start + vocab_chunk_size, self.vocab_size)
                
                vocab_chunk = self.vocab_hd_packed[start_pos:end_pos, v_start:v_end, :].to(self.device, non_blocking=True)
                xor_result = torch.bitwise_xor(vocab_chunk, token_hd_expanded)
                hamming_distances = self.popcount_lut[xor_result.long()].sum(dim=2, dtype=torch.int32)
                
                # Apply pooling operation
                if self.pooling == 'max':
                    self.HD_logits_gpu[v_start:v_end] = (50000 - hamming_distances).max(dim=0)[0].float()
                else:
                    self.HD_logits_gpu[v_start:v_end] = (50000 - hamming_distances).mean(dim=0, dtype=torch.float32)
                
                del vocab_chunk, xor_result, hamming_distances
        
        # Normalize in-place (adjust divisor for max pooling)
        if self.pooling == 'max':
            self.HD_logits_gpu.div_(1.0)  # No window length normalization needed
        else:
            self.HD_logits_gpu.div_(1.0)
        
        # Copy to cuda:0 (reuse buffer)
        self.HD_logits_cpu.copy_(self.HD_logits_gpu, non_blocking=False)
        
        del token_hd_expanded
        torch.cuda.empty_cache()
        
        return self.HD_logits_gpu




def LLM_get_next_token_logits(model, tokenizer, tokens, device):
    """
    Get probabilities for the next token given a list of tokens.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        tokens: List of token IDs or a string
        device: Device to run on
    
    Returns:
        next_token_logits: Tensor of logits for all tokens in vocabulary
    """
    # Convert to token IDs if input is a string
    if isinstance(tokens, str):
        tokens = tokenizer.encode(tokens, return_tensors='pt').to(device)
    elif isinstance(tokens, list):
        tokens = torch.tensor([tokens]).to(device)
    else:
        tokens = tokens.to(device)
    
    # Get model output
    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)
    
    # Get logits for the last token (next token prediction)
    next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)
    
    return next_token_logits


def inferer_captions_using_HD(img, top_k, caption_size, window_length, fixed_temp, clip_weight):
    """
    The core function to infer captions using HD representations.
    Args:
        img: Input image (PIL Image)
        top_k: Top-k value for sampling
        caption_size: Maximum caption size
        window_length: Window length for HD computation
        fixed_temp: Fixed temperature for sampling
        clip_weight: Weight for CLIP similarity in logits computation
    Returns:
        pred_caption: Generated caption (string)
    """
    img_tensor = transform(img).unsqueeze(0)
    
    hidden_batches_imgs, clip_image_features = F_VM_object.get_h_img(img_tensor)
    clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
    h_img = hidden_batches_imgs
    img_HD = F_VM_object.get_img_HD_vec(h_img)
    img_HD = torch.sign(img_HD)[0]
    pred_caption_tokens = [1986, 2168, 4933]
    
    i = 0
    while i < caption_size:
        
        i = len(pred_caption_tokens)-3
        if i>41:
            break
        toks, LM_rep_hidden = F_LM_object.get_h_caption(
            F_LM_object.tokenizer.decode(pred_caption_tokens)
        )

        LM_rep_HD = torch.sign(torch.matmul(LM_rep_hidden[0][-1], F_LM_object.LM_LSH_matrix))
        
        token_hd_vec = torch.sign(torch.mul(LM_rep_HD, img_HD))
        token_hd_vec = token_hd_vec > 0
        token_hd_vec = token_hd_vec.cpu()
        token_hd_vec = pack_boolean_tensor(token_hd_vec)

        HD_logits = hd_computer.compute(token_hd_vec, i, window_size=window_length)

        LLM_logits = LLM_get_next_token_logits(F_LM_object.model, F_LM_object.tokenizer, pred_caption_tokens, device)[:151669]

        HD_logits = HD_logits/HD_logits.max() + 0.15*LLM_logits/LLM_logits.max()
        
        predict_token_i = sampler.sample_next_token(
            logits=HD_logits,
            generated_tokens=pred_caption_tokens,
            clip_image_features=clip_image_features,
            temperature=fixed_temp, # CAN CHANGE PARARMS
            top_k=top_k,
            top_p=0.95,
            min_candidates= top_k,
            repetition_penalty=1.1,
            clip_weight=clip_weight
        )
                
        pred_caption_tokens.append(predict_token_i)
        
        if i > 1 and (pred_caption_tokens[-1] == F_LM_object.tokenizer.eos_token or pred_caption_tokens[-1] == 151645 or pred_caption_tokens[-1] == 13):
            break
    
    pred_caption = F_LM_object.tokenizer.decode(pred_caption_tokens)
    return pred_caption



def clean_caption_after_HD_inference(dirty_caption, use_LLM_paraphraser=False):
    
    # Remove EOS Tag
    clean_caption = dirty_caption.replace("<|im_end|>", "")

    # Prompt Template
    prompt_temp = f"""You are a precise image captioning assistant. You have a predicted noisy caption for the image:

        Caption (predicted caption): {clean_caption}
        
        Your task: Create ONE accurate, grammatically correct caption that:
        1. Removes any contradictions or repetitions
        2. Correct Grammar
        3. Uses natural, fluent English
        4. Starts with "This image shows" or "The image shows"
        5. Is ONE sentence only (no multiple sentences)
        6. Keeps all relevant objects, people, and actions mentioned
        7. Replace all proper nouns with generic terms and no geolocation name specific information should be present.
        
        Output ONLY the final caption, nothing else.
        
        Final caption:"""

    # Grammar Correct/Beautify with prompt
    if use_LLM_paraphraser:
        clean_caption = F_LM_object.generate_with_prompt(text="", prompt=prompt_temp, max_length=2500)
        
    # Remove "This image shows " prefix if present
    prefix = "This image shows "
    if clean_caption.startswith(prefix):
            clean_caption = clean_caption[len(prefix):]
            # Capitalize the first letter
            clean_caption = clean_caption[0].upper() + clean_caption[1:]

    # Remove "The image shows " prefix if present
    prefix = "The image shows "
    if clean_caption.startswith(prefix):
        clean_caption = clean_caption[len(prefix):]
        # Capitalize the first letter
        clean_caption = clean_caption[0].upper() + clean_caption[1:]

    return clean_caption
    
################################################################

def get_coco_karpathy_test_predictions(fixed_temp, top_k, caption_size, window_length, HD_vocab_in_use, use_LLM_paraphraser, clip_weight):
    splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'restval': 'data/restval-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/yerevann/coco-karpathy/" + splits["test"])
    

    results = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"preds_with_{HD_vocab_in_use}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("Now predicting captions for COCO Karpathy test Split")

    for i in tqdm(range(0, 5000), desc="COCO Test dataset Caption generation progress"):
    
        imgid = int(str(df['imgid'][i].astype(int)))
        
        url = df['url'][i]

        # image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        
        image = download_image_with_retry(url)
        
        caption = inferer_captions_using_HD(image, top_k=top_k, window_length=window_length, caption_size = caption_size, fixed_temp=fixed_temp, clip_weight=clip_weight)

        caption = clean_caption_after_HD_inference(caption, use_LLM_paraphraser) # Clean the Caption
        
        results.append({
            "image_id": imgid,
            "caption": caption
        })
        
        if i<5:
            print("Prints to check first 5 are working")
            print(imgid)
            print(caption)
            print("-------")
            with open(f"{output_dir}/COCO_predictions_mix_logits.json", "w") as f:
                json.dump(results, f, indent=2)
        
        # Save every 5 images
        if (i + 1) % 5 == 0:
            with open(f"{output_dir}/COCO_predictions_mix_logits.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved {i + 1} captions")
    
    # Final save
    with open(f"{output_dir}/COCO_predictions_mix_logits.json", "w") as f:
        json.dump(results, f, indent=2)


    print("Prediction of captions for COCO Karpathy test Split is Complete!")

################################################################

def get_nocaps_val_predictions(fixed_temp, top_k, caption_size, window_length, HD_vocab_in_use, use_LLM_paraphraser, clip_weight):

    print("Now predicting captions for NoCaps Val Split")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"preds_with_{HD_vocab_in_use}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []

    for i in tqdm(range(0, 4500), desc="NoCaps dataset Caption generation progress"):

        imgid = i
        
        filepath = os.path.join(no_caps_base_dir, f"{i}.jpg")
        image = Image.open(filepath).convert("RGB")
        
        caption = inferer_captions_using_HD(image, top_k=top_k, window_length=window_length, caption_size = caption_size, fixed_temp=fixed_temp, clip_weight=clip_weight)
        
        caption = clean_caption_after_HD_inference(caption, use_LLM_paraphraser) # Clean the Caption

        results.append({
            "image_id": imgid,
            "caption": caption
        })
        
        # Save every 5 images
        if (i + 1) % 5 == 0:
            with open(f"{output_dir}/no_caps_predictions_mix_logits.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved {i + 1} captions")
    
    # Final save
    with open(f"{output_dir}/no_caps_predictions_mix_logits.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Prediction of captions for NoCaps Val Split is Complete!")

#################################################################

def get_flickr_karpathy_test_predictions(fixed_temp, top_k, caption_size, window_length, HD_vocab_in_use, use_LLM_paraphraser, clip_weight):

    print("Now predicting captions for FLICKR Karpathy Test Split")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"preds_with_{HD_vocab_in_use}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []

    # Get all .jpg files in directory, sorted for reproducibility
    image_files = sorted([
        f for f in os.listdir(flickr_test_base_dir)
        if f.lower().endswith(".jpg")
    ])

    print(f"Found {len(image_files)} images.")

    for idx, filename in tqdm(enumerate(image_files), total=len(image_files),
                            desc="FLICKR dataset Caption generation progress"):

        filepath = os.path.join(flickr_test_base_dir, filename)

        # extract the numeric ID (filename without extension)
        image_id = os.path.splitext(filename)[0]

        image = Image.open(filepath).convert("RGB")


        caption = inferer_captions_using_HD(image, top_k=top_k, window_length=window_length, caption_size = caption_size, fixed_temp=fixed_temp, clip_weight=clip_weight)
        
        caption = clean_caption_after_HD_inference(caption, use_LLM_paraphraser) # Clean the Caption

        results.append({
            "image_id": image_id,
            "caption": caption
        })
        
        # Save every 5 images
        if (idx + 1) % 5 == 0:
            with open(f"{output_dir}/flickr30k_predictions.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved {idx + 1} captions")
        
    # Final save
    with open(f"{output_dir}/flickr30k_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Prediction of captions for Flickr30k Test Split is Complete!")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Eval Captions')
        
    # Add arguments - all required
    parser.add_argument('--task', type=str, choices=['coco', 'nocaps', 'flickr'], 
                        required=True, help='Which task to run')
    parser.add_argument('--aspect_ratio_threshold', type=float, required=True, 
                        help='Aspect ratio threshold')
    parser.add_argument('--HD_vocab_path', type=str, required=True, 
                        help='Path to vocab HD packed file')
    parser.add_argument('--fixed_temp', type=float, required=True, 
                        help='Fixed Temperature value for sampling')
    parser.add_argument('--use_LLM_paraphraser', type=str2bool, required=True,
                        help='If True, uses LLM paraphraser to clean captions after HD inference')
    parser.add_argument('--top_k', type=int, required=True, 
                        help='Top-k value for sampling')
    parser.add_argument('--caption_size', type=int, required=True, 
                        help='Maximum Caption size to generate')
    parser.add_argument('--window_length', type=int, required=True, 
                        help='Window length for HD computation')
    parser.add_argument('--clip_weight', type=float, required=True, 
                        help='Weight for CLIP similarity in logits computation')
    parser.add_argument('--path_to_nocaps_val_images', type=str, required=True, 
                        help='Path to NoCaps validation images')
    parser.add_argument('--path_to_flickr_test_images', type=str, required=True, 
                        help='Path to Flickr Kaprthy Test set images')

    args = parser.parse_args()

    global device, F_VM_object, F_LM_object, packed_memmap, vocab_hd_packed
    global vocab_hd_packed_pinned, popcount_lut_gpu, hd_computer, transform, sampler
    global no_caps_base_dir, flickr_test_base_dir
    global POPCOUNT_TABLE  # just in case needed this globally too
    
    # Define device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Frozen Vision Model
    F_VM_object= FrozenVisionModel_Encoding(device = device)
    
    # Initialize Frozen Language Model
    F_LM_object = FrozenLanguageModel_Encoding(device = device, AutoModelForCausalLM_flag=True)

    # Load packed vocab predictors
    packed_memmap = np.memmap(args.HD_vocab_path, dtype=np.uint8, mode='r', shape=(43, 151669, 6250))

    # Short to see what HD_vcoab is being used
    HD_vocab_in_use = args.HD_vocab_path.split("/")[-1].replace(".dat", "")
    print(f"Using HD Vocab: {HD_vocab_in_use}")
    
    # Convert to torch tensor (creates view, no memory copy)
    vocab_hd_packed = torch.from_numpy(packed_memmap[:]).clone()  # .clone() copies to RAM
    vocab_hd_packed = vocab_hd_packed[:args.caption_size+1]

    # Create a lookup table for popcount of all byte values (0-255)
    POPCOUNT_TABLE = torch.tensor([bin(i).count('1') for i in range(256)], dtype=torch.uint8)

    # ============= OPTIMIZATION: Initialize once, reuse =============
    vocab_hd_packed_pinned = vocab_hd_packed.pin_memory()
    popcount_lut_gpu = POPCOUNT_TABLE.to(device)  # Move once, keep on GPU

    # Initialize the computer once with max pooling - REUSED ACROSS ALL IMAGES
    hd_computer = HDLogitsComputer(vocab_hd_packed_pinned, popcount_lut_gpu, pooling='max')

    transform = make_transform(resize_size=512, aspect_ratio_threshold=args.aspect_ratio_threshold)

    # Initialize sampler once (reuse across multiple captions)
    sampler = CLIPSemanticSampler(
        lm_tokenizer=F_LM_object.tokenizer,
        clip_model=F_VM_object.model,
        clip_tokenizer=F_VM_object.clip_tokenizer.tokenize
    )

    no_caps_base_dir = args.path_to_nocaps_val_images
    flickr_test_base_dir = args.path_to_flickr_test_images

    if args.task in ['coco']:
        print("-------RUNNING KARPATHY TEST SPLIT COCO PREDICTIONS-------")
        get_coco_karpathy_test_predictions(fixed_temp=args.fixed_temp, 
                                           top_k=args.top_k, caption_size=args.caption_size, 
                                           window_length=args.window_length, HD_vocab_in_use=HD_vocab_in_use, 
                                           use_LLM_paraphraser=args.use_LLM_paraphraser, clip_weight=args.clip_weight)

    if args.task in ['flickr']:
        print("-------RUNNING KARPATHY TEST SPLIT FLICKR PREDICTIONS-------")
        get_flickr_karpathy_test_predictions(fixed_temp=args.fixed_temp, 
                                             top_k=args.top_k, caption_size=args.caption_size, 
                                             window_length=args.window_length, HD_vocab_in_use=HD_vocab_in_use, 
                                             use_LLM_paraphraser=args.use_LLM_paraphraser, clip_weight=args.clip_weight)
    
    if args.task in ['nocaps']:
        print("-------RUNNING NOCAPS VAL SPLIT PREDICTIONS-------")
        get_nocaps_val_predictions(fixed_temp=args.fixed_temp, 
                                           top_k=args.top_k, caption_size=args.caption_size, 
                                           window_length=args.window_length, HD_vocab_in_use=HD_vocab_in_use, 
                                           use_LLM_paraphraser=args.use_LLM_paraphraser, clip_weight=args.clip_weight)