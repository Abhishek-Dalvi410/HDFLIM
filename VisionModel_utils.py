import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from io import BytesIO
import requests


#----|------------------------------------#
#----|    THIS FILE REQUIRES DINOv3       |
#----|  WITH CLIP ENCODER MODEL WEIGHTS   |
#----|   TO BE DOWNLOADED SEPARATELY      |
#----| THROUGH META'S WEBSITE AND GITHUB  |
#----|              REPO!                 |
#----| github.com/facebookresearch/dinov3 |
#----|------------------------------------#


class FrozenVisionModel_Encoding:
    """
    Used for instantiating of a Frozen Vision Encoder with methods to get hidden representations and projecting them to HD space.

    Constructor Parameters
    ----------------------
        model_name : String
            Defaults to "facebook/dino".
        device : String
            Defaults to "cpu".
        LSH_mat_on_disk : Bool
            Defaults to False.
        HD_dim_size : int
            Defaults to 50000.

    Methods
    -------
        get_h_img(imgs) 
            Returns the hidden representations of the image.
        get_HD_rep_img()
            Returns the HD representations of the image.
    """
    def __init__(
        self, 
        device, 
        model_name="Dinov3.txt", 
        HD_dim_size=50000, 
        last_hidden_state_dim = 1024, 
        num_patches = 1025,
        img_LSH_matrix_path= "/storage/group/vuh14/default/Abhishek_files/dinov3txt_qwen3/saved_HD_mats/img_LSH_matrix.pt", # Replace this with img_LSH_matrix.pt path
        img_pos_HD_path = "/storage/group/vuh14/default/Abhishek_files/dinov3txt_qwen3/saved_HD_mats/img_pos_HD.pt" # Replace this with img_pos_HD.pt path
        ):
        
        print("INSIDE Constructor/Initializer function of FrozenVisionModel_Encoding class")
        print("Loading Dinov3.txt ..... \n")
        print("Specified device to use: ", device, "\n")
        print("Specified HD dimension size: ", HD_dim_size, "\n")
        print("Specified last hidden state dimension size: ", last_hidden_state_dim, "\n")
        print("Specified number of patches: ", num_patches, "\n")

        self.device = device
        """
        #-----------------------------------------------------------------------#
        - Please see clone DINOv3 Github Repo to use and download the appropriate model weights!
        - https://github.com/facebookresearch/dinov3
        - After downloading the weights, please ensure that the paths specified in the torch.hub.load() function below are correct.
        - The Github repo's README.md has all the details and instructions about loading the model.
        #-----------------------------------------------------------------------#
        """
        self.model , self.clip_tokenizer = torch.hub.load("/storage/group/vuh14/default/Abhishek_files/dinov3txt_qwen3/dinov3_repo/dinov3", 
                                        'dinov3_vitl16_dinotxt_tet1280d20h24l', 
                                        source='local', 
                                        backbone_weights="/storage/group/vuh14/default/Abhishek_files/dinov3txt_qwen3/meta_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", 
                                        weights="meta_weights/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth")
        
        
        self.last_hidden_state_dim = last_hidden_state_dim
        self.num_patches = num_patches

        try:
            self.img_LSH_matrix = torch.load(img_LSH_matrix_path)
            self.img_pos_HD = torch.load(img_pos_HD_path)
            print("=" * 80)
            print("✓ Successfully loaded HD matrices from disk")
            print("=" * 80)
            print(f"  [1] img_LSH_matrix.pt")
            print(f"      Path: {img_LSH_matrix_path}")
            print(f"      Shape: {self.img_LSH_matrix.shape}")
            print(f"      Dtype: {self.img_LSH_matrix.dtype}")
            print(f"\n  [2] img_pos_HD.pt")
            print(f"      Path: {img_pos_HD_path}")
            print(f"      Shape: {self.img_pos_HD.shape}")
            print(f"      Dtype: {self.img_pos_HD.dtype}")
            print("=" * 80 + "\n")
        except FileNotFoundError as e:
            print("=" * 80)
            print("✗ HD matrices not found on disk")
            print("=" * 80)
            print(f"  Looking for:")
            print(f"    [1] {img_LSH_matrix_path}")
            print(f"    [2] {img_pos_HD_path}")
            print(f"\n  Error: {str(e)}")
            print(f"\n  Please run the initialization script to create these files.")
            print("=" * 80 + "\n")
            raise
            
        self.img_LSH_matrix = self.img_LSH_matrix.to(torch.bfloat16).to(device)
        self.img_pos_HD = self.img_pos_HD.to(device)
        self.model = self.model.eval().cuda()

        print("Frozen Vision Model: ",model_name, "with HD LSH matrix instantiated succesfully! \n")

    def get_h_img(self, images):
        """
        Compute the concat hidden representations (last layer output) of the images.
        Parameters
        ----------
        images : torch.Tensor
        A batch of preprocessed images to process with shape
        (batch_size, 512, 512) # For Dinov3
        Returns
        -------
        last_hidden_states : torch.Tensor
        Hidden representations of the batched images with shape
        (batch_size, num_patches, hidden_dim).
        """
        images = images.to(self.device)

        # Get the model output
        with torch.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                image_class_tokens, image_patch_tokens, _ = self.model.encode_image_with_patch_tokens(images)
        
        image_class_tokens_half = image_class_tokens[:, :1024]
        
        # 2. Expand class tokens and concat along sequence dimension
        final_tokens_hidden_states = torch.cat([image_class_tokens_half.unsqueeze(1), image_patch_tokens], dim=1)  # [N, 1025, 1024]

        final_tokens_hidden_states = final_tokens_hidden_states.to(torch.bfloat16)
            
        return final_tokens_hidden_states, image_class_tokens
    
    def get_img_HD_vec(self, hidden_img):
        """
        Compute the HD representations of the batched images

        Parameters
        ----------
        hidden_img: torch.Tensor
            Hidden representation of the batch of images 
            with shape: (batch_size, num_patches, hidden_dim)

        Returns
        -------
        img_hd : torch.Tensor
            HD representation of the batch of images 
            with shape: (batch_size, HD_dim_size)
        """
        
        img_HD = torch.sign(torch.matmul(hidden_img, self.img_LSH_matrix)) # shape: (batch_size, num_patches, HD_dim_size)
        
        img_HD = img_HD * self.img_pos_HD # Element wise multiplication with broadcasting; shape: [batch_size, num_patches, HD_dim_size]

        img_HD = torch.sign(torch.sum(img_HD, dim=1)) # Bundle across the patch dimension
        img_HD = img_HD.to(torch.int8)
        
        return img_HD