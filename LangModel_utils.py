from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch

from huggingface_hub import login

####login("...........")  # If not already logged in



class FrozenLanguageModel_Encoding:
    def __init__(
        self, 
        device,
        AutoModelForCausalLM_flag=False,
        model_name="Qwen/Qwen3-4B",
        HD_dim_size=50000,
        caption_size = 43, 
        LM_LSH_matrix_path = "/storage/group/vuh14/default/Abhishek_files/dinov3txt_qwen3/saved_HD_mats/LM_LSH_matrix.pt" # Replace this with LM_LSH_matrix.pt path
        ):
        
        print("INSIDE Constructor/Initializer function of FrozenLanguageModel_Encoding class")
        
        print("Loading ", model_name, ".....")
        self.device = device
        self.HD_dim_size = HD_dim_size
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        if AutoModelForCausalLM_flag:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, torch_dtype=torch.bfloat16).to(device)
            
        else:
            self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True, torch_dtype=torch.bfloat16).to(device)
        
        self.model.eval()  # Set to evaluation mode
        for param in self.model.parameters():
            param.requires_grad = False  # Explicitly freeze parameters
            
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        self.vocab_size = len(self.tokenizer)  # This includes ALL tokens including special ones
        
        self.eos_id = self.tokenizer.eos_token_id
        self.caption_size = caption_size
        self.hidden_state_dimension = self.model.config.hidden_size
        print("Frozen Language model ", model_name, "summary:")
        print("Frozen Language model vocabulary size:-")
        print(self.vocab_size)
        print("Frozen Language model tokenizer has the following special tokens map:-")
        print(self.tokenizer.special_tokens_map)
        print("Frozen Language model will truncate captions to size ", caption_size, "tokens")
        print("Last hidden state dimension :", self.hidden_state_dimension)
        print("Using device for Frozen Language model: ", device)
        print("Checking if saved_HD_mats/LM_LSH_matrix.pt exists on disk")
        
        try:
            self.LM_LSH_matrix = torch.load(LM_LSH_matrix_path)
            print("=" * 80)
            print("✓ Successfully loaded HD matrix from disk")
            print("=" * 80)
            print(f"  [1] LM_LSH_matrix.pt")
            print(f"      Path: {LM_LSH_matrix_path}")
            print(f"      Shape: {self.LM_LSH_matrix.shape}")
            print(f"      Dtype: {self.LM_LSH_matrix.dtype}")
        except FileNotFoundError as e:
            print("=" * 80)
            print("✗ HD matrix not found on disk")
            print("=" * 80)
            print(f"  Looking for:")
            print(f"    [1] {LM_LSH_matrix_path}")
            print(f"\n  Error: {str(e)}")
            print(f"\n  Please run the initialization script to create these files.")
            print("=" * 80 + "\n")
            raise
            
        self.LM_LSH_matrix = self.LM_LSH_matrix.to(torch.bfloat16).to(device)
        print("Frozen Language Model: ",model_name, "with HD LSH matrix instantiated succesfully! \n")
    
    def get_h_caption(self, captions):
        # Tokenize the input string
        batched_inputs = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt", max_length=self.caption_size)
        batched_inputs = batched_inputs.to(self.device)
        
        # Get hidden states
        with torch.no_grad():
            outputs = self.model(**batched_inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # List of hidden states from each layer
            
        last_hidden_states = hidden_states[-1].detach()  # shape: (batch_size, seq_len, hidden_dim)
        
        mask = batched_inputs.attention_mask.unsqueeze(-1).to(torch.bfloat16)
        
        # Apply mask
        masked_hidden_state = last_hidden_states * mask
        
        return batched_inputs.input_ids, masked_hidden_state

    def get_caption_HD_vec(self, hidden_caption):
        
        caption_HD = torch.sign(torch.matmul(hidden_caption, self.LM_LSH_matrix))
        
        caption_HD = caption_HD.to(torch.int8)
        
        return caption_HD
        
    def generate_with_prompt(self, text, prompt="Paraphrase this:", max_length=10000, device='cuda'):
        # Combine prompt + text
        input_text = f"{prompt} {text}"
        # prepare the model input
        messages = [
            {"role": "user", "content": input_text}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768,
            use_cache=True
        )

        index = 0
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return content