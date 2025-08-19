import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv2Encoder(nn.Module):
    """
    DINOv2-based encoder for feature extraction
    """
    
    def __init__(self, model_name: str = "dinov2_vits14", finetune: bool = True):
        """
        Initialize DINOv2 encoder
        
        Args:
            model_name: DINOv2 model variant
            finetune: Whether to allow fine-tuning
        """
        super().__init__()
        self.model_name = model_name
        self.finetune = finetune
        
        self.model_mapping = {
            "dinov2_vits14": "facebook/dinov2-small",
            "dinov2_vitb14": "facebook/dinov2-base",
            "dinov2_vitl14": "facebook/dinov2-large", 
            "dinov2_vitg14": "facebook/dinov2-giant"
        }
        
        if model_name not in self.model_mapping:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.encoder = AutoModel.from_pretrained(self.model_mapping[model_name])
        
        self.feature_dim = self.encoder.config.hidden_size
        
        self.set_finetune(finetune)
        
    def set_finetune(self, finetune: bool):
        self.finetune = finetune
        for param in self.encoder.parameters():
            param.requires_grad = finetune
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [B*S, 3, H, W] input tensor
            
        Returns:
            features: [B*S, N, D] feature tensor where N is number of patches
        """
        # DINOv2 expects RGB input, so we need to handle 3-channel input
        outputs = self.encoder(pixel_values=x)
        
        # Get patch embeddings (excluding CLS token)
        features = outputs.last_hidden_state[:, 1:]  # [B*S, N, D]
        
        return features

