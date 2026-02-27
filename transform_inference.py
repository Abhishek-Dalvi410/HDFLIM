import torch
import torchvision.transforms.v2 as Tv2
from PIL import Image

"""
You can try out different image transformations
like Full resize, resize then crop, center crop, etc. and see 
which one works best for your use case.
Performance seems to vary a bit as per our observations.
The below configuration is the one used for evals
"""

def make_transform_normalize():
    """
    Transform that only normalizes the image.
    
    Returns:
        Composed transform
    """
    to_tensor = Tv2.ToImage()  # converts PIL/ndarray -> ImageTensor
    to_float = Tv2.ToDtype(torch.float32, scale=True)
    
    normalize = Tv2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    
    return Tv2.Compose([to_tensor, to_float, normalize])
    
def make_transform(resize_size: int = 512, crop_size: int = 512, aspect_ratio_threshold: float = 1.0):
    """
    Creates a transform pipeline that handles images intelligently based on aspect ratio.
    
    Args:
        resize_size: Target size for resizing
        crop_size: Target size for cropping (should typically equal resize_size for square output)
        aspect_ratio_threshold: If aspect ratio (max/min dimension) exceeds this, use center crop
                               to preserve aspect ratio. Default 1.4 means if one dimension is 
                               more than 1.4x the other, it will crop instead of distorting.
    """
    
    class AspectRatioAwareResize:
        def __init__(self, target_size, aspect_threshold):
            self.target_size = target_size
            self.aspect_threshold = aspect_threshold
            
        def __call__(self, img):
            # Get image dimensions
            if hasattr(img, 'shape'):
                # For tensor images (C, H, W)
                height, width = img.shape[-2:]
            else:
                # For PIL images
                width, height = img.size
            
            # Calculate aspect ratio
            aspect_ratio = max(width, height) / min(width, height)
            
            if aspect_ratio > self.aspect_threshold:
                # Image is too stretched - resize to fit and then center crop
                # Resize so the smaller dimension becomes target_size
                if width > height:
                    # Wide image - resize based on height
                    resize_height = self.target_size
                    resize_width = int(width * (resize_height / height))
                else:
                    # Tall image - resize based on width
                    resize_width = self.target_size
                    resize_height = int(height * (resize_width / width))
                
                # Apply resize maintaining aspect ratio
                resize_op = Tv2.Resize((resize_height, resize_width), 
                                      interpolation=Tv2.InterpolationMode.BICUBIC, 
                                      antialias=True)
                img = resize_op(img)
                
                # Then center crop to square
                crop_op = Tv2.CenterCrop(self.target_size)
                img = crop_op(img)
            else:
                # Aspect ratio is reasonable - just resize to square
                resize_op = Tv2.Resize((self.target_size, self.target_size), 
                                      interpolation=Tv2.InterpolationMode.BICUBIC, 
                                      antialias=True)
                img = resize_op(img)
            
            return img
    
    to_tensor = Tv2.ToImage()  # converts PIL/ndarray -> ImageTensor
    aspect_aware_resize = AspectRatioAwareResize(crop_size, aspect_ratio_threshold)
    to_float = Tv2.ToDtype(torch.float32, scale=True)
    normalize = Tv2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    
    return Tv2.Compose([to_tensor, aspect_aware_resize, to_float, normalize])