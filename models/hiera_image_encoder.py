import torch
import torch.nn as nn


class HieraImageEncoder(nn.Module):
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
            "hiera_tiny_224": "facebook/hiera_tiny_224.mae_in1k",
            "hiera_small_224": "facebook/hiera_small_224.mae_in1k",
            "hiera_base_224": "facebook/hiera_base_224.mae_in1k",
            "hiera_base_plus_224": "facebook/hiera_base_plus_224.mae_in1k",
            "hiera_large_224": "facebook/hiera_large_224.mae_in1k",
            "hiera_huge_224": "facebook/hiera_huge_224.mae_in1k"
        }
        
        if model_name not in self.model_mapping:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # self.encoder = Hiera.from_pretrained(self.model_mapping[model_name])  
        self.model = torch.hub.load("facebookresearch/hiera", model=model_name, pretrained=True, checkpoint=self.model_mapping[model_name].split(".")[-1])
        
        self.set_finetune(finetune)
        
    def set_finetune(self, finetune: bool):
        self.finetune = finetune
        for param in self.parameters():
            param.requires_grad = finetune
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [B*S, 3, H, W] input tensor
            
        Returns:
            outputs:
                [B*S, 1000]  # ignore
            intermediates:
                [B*S, 56, 56, 96]  # shallow
                [B*S, 28, 28, 192]
                [B*S, 14, 14, 384]
                [B*S, 7, 7, 768]  # deep
        """
        outputs, intermediates = self.model(x, return_intermediates=True)
        return outputs, intermediates
    