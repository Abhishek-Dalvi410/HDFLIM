import torch
from typing import List

# ---------------- Sampling with CLIP Similarity ---------------- #

class CLIPSemanticSampler:
    """
    Minimal CLIP-guided sampler with:
    - Repetition penalty (exact word repetition)
    - CLIP guidance for image-text alignment
    """
    
    def __init__(self, lm_tokenizer, clip_model, clip_tokenizer):
        self.lm_tokenizer = lm_tokenizer
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
    
    def sample_next_token(
        self,
        logits: torch.Tensor,
        generated_tokens: List[int],
        clip_image_features: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 20,
        top_p: float = 0.9,
        repetition_penalty: float = 1.3,
        clip_weight: float = 0.5,
        min_candidates: int = 3
    ) -> int:
        """
        Sample next token with CLIP guidance, repetition penalty, and semantic penalty.
        
        Args:
            logits: Model logits
            generated_tokens: List of already generated token IDs
            clip_image_features: CLIP image embeddings
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for exact token repetition
            clip_weight: Weight for CLIP score (0-1)
            min_candidates: Minimum number of candidates to consider
        """
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        
        # Apply repetition penalty (exact token repetition)
        if repetition_penalty != 1.0 and len(generated_tokens) > 0:
            recent = generated_tokens[-30:] if len(generated_tokens) > 30 else generated_tokens
            unique_recent = list(set(recent))
            for token_id in unique_recent:
                count = recent.count(token_id)
                penalty = repetition_penalty ** min(count, 3)
                if logits[token_id] > 0:
                    logits[token_id] /= penalty
                else:
                    logits[token_id] *= penalty
        
        # Get top-k candidates
        k = min(top_k, logits.shape[-1])
        topk_logits, topk_indices = torch.topk(logits, k)
        
        # Apply softmax
        topk_probs = torch.softmax(topk_logits, dim=-1)
        
        # Apply top-p (nucleus) sampling
        sorted_probs, sorted_indices = torch.sort(topk_probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum <= top_p
        
        # Ensure minimum candidates
        for i in range(min(min_candidates, len(mask))):
            mask[i] = True
        
        if not mask.any():
            mask[0] = True
        
        kept_indices = sorted_indices[mask]
        kept_probs = sorted_probs[mask]
        
        # Get candidate tokens
        candidate_tokens = topk_indices[kept_indices]
        
        # Score with CLIP if we have multiple candidates
        if len(candidate_tokens) > 1 and clip_image_features is not None:
            # Generate candidate texts
            candidate_texts = []
            for token_id in candidate_tokens:
                new_tokens = generated_tokens + [token_id.item()]
                tokens_for_clip = new_tokens[:]
                text = self.lm_tokenizer.decode(tokens_for_clip, skip_special_tokens=True)
                candidate_texts.append(text)
            
            # Get CLIP scores
            with torch.no_grad():
                text_inputs = self.clip_tokenizer(candidate_texts).to(clip_image_features.device)
                text_features = self.clip_model.encode_text(text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                clip_scores = (clip_image_features @ text_features.T).squeeze()
                
                # Normalize scores
                clip_scores = torch.softmax(clip_scores * 2.0, dim=-1)
                lm_scores = kept_probs / kept_probs.sum()
                
                # Combine scores
                combined = clip_weight * clip_scores + (1 - clip_weight) * lm_scores
                best_idx = combined.argmax().item()
        else:
            best_idx = 0
        
        selected_token = candidate_tokens[best_idx].item()
        
        return selected_token