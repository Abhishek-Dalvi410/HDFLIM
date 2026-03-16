import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as Tv2
import webdataset as wds

# -------------------------
# Transform
# -------------------------

def make_transform(resize_size: int = 512, crop_size: int = 512, aspect_ratio_threshold: float = 1.4):
    to_tensor = Tv2.ToImage()  # converts PIL/ndarray -> ImageTensor
    resize = Tv2.Resize(resize_size, interpolation=Tv2.InterpolationMode.BICUBIC, antialias=True)  # resizes shortest side to resize_size
    crop = Tv2.CenterCrop(crop_size)  # crops to square
    to_float = Tv2.ToDtype(torch.float32, scale=True)
    normalize = Tv2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        )
    return Tv2.Compose([to_tensor, resize, crop, to_float, normalize])

# -------------------------
# Collate function
# -------------------------
def collate_batch(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    captions = list(captions)
    
    # Process each caption
    processed_captions = []
    for caption in captions:
        # Convert to lowercase
        caption = caption.lower()
        
        # Split by periods, keeping track of them
        parts = caption.split('.')
        
        # Filter out empty strings and strip whitespace
        parts = [p.strip() for p in parts if p.strip()]
        
        # Capitalize first letter of each part (except the first one) and join with ". "
        if len(parts) > 0:
            capitalized_parts = [parts[0]]  # Keep first part as is
            for p in parts[1:]:
                capitalized_parts.append(p[0].upper() + p[1:] if len(p) > 0 else p)
            caption = ". ".join(capitalized_parts)
        
        # Add prefix and full stop at the end
        caption = "This image shows " + caption + "."
        
        processed_captions.append(caption)
    
    return images, processed_captions
    

def collate_batch_with_url(batch):
    images, captions, urls = zip(*batch)
    images = torch.stack(images, dim=0)
    captions = list(captions)
    
    # Process each caption (your existing code)
    processed_captions = []
    for caption in captions:
        caption = caption.lower()
        parts = caption.split('.')
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 0:
            capitalized_parts = [parts[0]]
            for p in parts[1:]:
                capitalized_parts.append(p[0].upper() + p[1:] if len(p) > 0 else p)
            caption = ". ".join(capitalized_parts)
        caption = "This image shows " + caption + "."
        processed_captions.append(caption)
    
    # Return the first URL as shard identifier
    shard_url = urls[0] if urls else "unknown"
    
    return images, processed_captions, shard_url

# -------------------------
# Decode captions helper
# -------------------------
def decode_caption(x):
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace").strip()
    if isinstance(x, str):
        return x.strip()
    try:
        data = x.read()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace").strip()
        return str(data).strip()
    except Exception:
        return str(x).strip()

# -------------------------
# Loader factory OLD Commenting
# -------------------------
# def create_loader(
#     shard_pattern: str,
#     batch_size: int = 32,
#     resize_size: int = 512,
#     crop_size: int = 512,
#     aspect_ratio_threshold: float = 1.2, # For Training kept as 1.2
#     num_workers: int = 1,
#     shuffle: bool = False
# ):
#     transform = make_transform(resize_size, crop_size, aspect_ratio_threshold)
# 
#     dataset = (
#         wds.WebDataset(shard_pattern, handler=wds.warn_and_continue)
#            .decode("pil")
#            .to_tuple("jpg;png;jpeg", "txt")
#            .map_tuple(lambda img: transform(img), decode_caption)
#     )
# 
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         collate_fn=collate_batch,
#         num_workers=num_workers,
#         pin_memory=True,
#     )
# 
#     return loader
#-----------------------

# -------------------------
# Loader factory NEW for tracking the shard number
# -------------------------
def create_loader(
    shard_pattern: str,
    batch_size: int = 32,
    resize_size: int = 512,
    crop_size: int = 512,
    aspect_ratio_threshold: float = 1.2,
    num_workers: int = 1,
    shuffle: bool = False
):
    transform = make_transform(resize_size, crop_size, aspect_ratio_threshold)
    
    dataset = (
        wds.WebDataset(shard_pattern, handler=wds.warn_and_continue, shardshuffle=shuffle)
           .decode("pil", handler=wds.warn_and_continue)
           .to_tuple("jpg;png;jpeg", "txt", "__url__")  # Add __url__ to track shard
           .map_tuple(
               lambda img: transform(img), 
               decode_caption,
               lambda url: url  # Pass through the URL
           )
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch_with_url,  # Use new collate function
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader