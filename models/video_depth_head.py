import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import os
from typing import Tuple


class VideoDepthAnythingHead(nn.Module):
    """
    Video Depth Anything spatiotemporal head
    """
    
    def __init__(self, 
                 input_dim: int,
                 model_size: str = "small",
                 sequence_length: int = 8,
                 img_size: Tuple[int, int] = (384, 384),
                 finetune: bool = True):
        """
        Initialize Video Depth Anything head
        
        Args:
            input_dim: Input feature dimension from encoder
            model_size: "small" or "large"
            sequence_length: Number of frames in sequence
            img_size: (H, W) image size
            finetune: Whether to allow fine-tuning
        """
        super().__init__()
        self.input_dim = input_dim
        self.model_size = model_size
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.finetune = finetune
        
        # Calculate patch dimensions (assuming patch size 14x14)
        self.patch_size = 14
        self.num_patches_h = img_size[0] // self.patch_size
        self.num_patches_w = img_size[1] // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Build model components
        self._build_temporal_attention()
        self._build_spatial_temporal_fusion()
        self._build_decoder()
        self._build_prediction_head()
        
        # Initialize weights
        self._initialize_weights()
        
        self.set_finetune(finetune)
        
    def _build_temporal_attention(self):
        """Build temporal attention module"""
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.input_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def _build_spatial_temporal_fusion(self):
        """Build spatial-temporal fusion module"""
        self.st_fusion = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.input_dim, self.input_dim)
        )
        
    def _build_decoder(self):
        """Build decoder layers"""
        decoder_dim = 256
        self.decoder = nn.ModuleList([
            # Upsampling layers
            nn.Sequential(
                nn.ConvTranspose2d(self.input_dim, decoder_dim * 4, 4, 2, 1),
                nn.BatchNorm2d(decoder_dim * 4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(decoder_dim * 4, decoder_dim * 2, 4, 2, 1),
                nn.BatchNorm2d(decoder_dim * 2),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(decoder_dim * 2, decoder_dim, 4, 2, 1),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(decoder_dim, decoder_dim // 2, 4, 2, 1),
                nn.BatchNorm2d(decoder_dim // 2),
                nn.ReLU(inplace=True)
            )
        ])
        
    def _build_prediction_head(self):
        """Build final prediction head"""
        # Final prediction head - outputs 3 channels (depth, dx, dy)
        self.prediction_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),  # decoder_dim // 2 = 128
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1)  # 3 channels: depth, dx, dy
        )
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def set_finetune(self, finetune: bool):
        """Set fine-tuning mode"""
        self.finetune = finetune
        for param in self.parameters():
            param.requires_grad = finetune
            
    def load_pretrained_weights(self, weights_path: str):
        """Load pretrained weights from Video Depth Anything"""
        if weights_path.startswith('http'):
            # Download weights if URL provided
            os.makedirs('./pretrained', exist_ok=True)
            local_path = f'./pretrained/video_depth_anything_{self.model_size}.pth'
            
            if not os.path.exists(local_path):
                print(f"Downloading weights from {weights_path}")
                try:
                    response = requests.get(weights_path, timeout=30)
                    response.raise_for_status()
                    with open(local_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Weights saved to {local_path}")
                except Exception as e:
                    print(f"Failed to download weights: {e}")
                    return
            
            weights_path = local_path
        
        # Load weights (partial loading - only compatible layers)
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
                
            # Filter compatible weights
            model_dict = self.state_dict()
            compatible_dict = {}
            
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    compatible_dict[k] = v
                    
            model_dict.update(compatible_dict)
            self.load_state_dict(model_dict, strict=False)
            print(f"Loaded {len(compatible_dict)} compatible layers from pretrained weights")
            
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Continuing with random initialization")
    
    def _apply_temporal_attention(self, features: torch.Tensor) -> torch.Tensor:
        """Apply temporal attention to features"""
        B, S, N, D = features.shape
        
        # Reshape to [B*N, S, D] for attention
        features_for_attention = features.permute(0, 2, 1, 3).contiguous().reshape(B * N, S, D)
        
        # Temporal attention
        attended_features, _ = self.temporal_attention(
            features_for_attention, features_for_attention, features_for_attention
        )  # [B*N, S, D]
        
        # Reshape back: [B, N, S, D] -> [B, S, N, D]
        attended_features = attended_features.reshape(B, N, S, D).permute(0, 2, 1, 3)
        
        return attended_features
    
    def _apply_spatial_temporal_fusion(self, features: torch.Tensor) -> torch.Tensor:
        """Apply spatial-temporal fusion"""
        fused_features = self.st_fusion(features)
        return features + fused_features  # Residual connection
    
    def _decode_features(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features to output"""
        B_times_S, N, D = features.shape
        
        # Reshape to spatial feature maps: [B*S, D, H_patch, W_patch]
        features = features.permute(0, 2, 1).reshape(
            B_times_S, D, self.num_patches_h, self.num_patches_w
        )
        
        # Apply decoder layers
        x = features
        for decoder_layer in self.decoder:
            x = decoder_layer(x)
        
        # Final prediction
        output = self.prediction_head(x)  # [B*S, 3, H_out, W_out]
        
        return output
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            features: [B*S, N, D] feature tensor from encoder
            
        Returns:
            output: [B, S, 3, H, W] predicted depth, dx, dy
        """
        B_times_S, N, D = features.shape
        B = B_times_S // self.sequence_length
        S = self.sequence_length
        
        # Reshape for temporal processing: [B, S, N, D]
        features = features.reshape(B, S, N, D)
        
        # Apply temporal attention across frames
        attended_features = self._apply_temporal_attention(features)
        
        # Spatial-temporal fusion
        fused_features = self._apply_spatial_temporal_fusion(attended_features)
        
        # Reshape for spatial decoding: [B*S, N, D]
        features_for_decode = fused_features.reshape(B * S, N, D)
        
        # Decode features
        output = self._decode_features(features_for_decode)
        
        # Get actual output dimensions
        _, _, H_out, W_out = output.shape
        
        # Resize to target size if needed
        if (H_out, W_out) != self.img_size:
            output = F.interpolate(
                output, 
                size=self.img_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        # Reshape to [B, S, 3, H, W]
        output = output.reshape(B, S, 3, self.img_size[0], self.img_size[1])
        
        return output
    
    def get_supported_sizes(self) -> list:
        """Get list of supported model sizes"""
        return ["small", "large"]
