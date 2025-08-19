import torch
import torch.nn as nn
from einops import rearrange
from easydict import EasyDict

class TransformerBlock(nn.Module):
    """
    A single block of the Temporal Transformer.
    This module contains a Multi-Head Self-Attention layer followed by a Feed-Forward Network.
    It's the basic building block.
    """
    def __init__(self, in_channels: int, num_heads: int):
        """
        Initialize the Temporal Transformer Block.

        Args:
            in_channels (int): Input feature dimension (C). Must be divisible by num_heads.
            num_heads (int): Number of parallel attention heads.
        """
        super().__init__()
        
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Linear(in_channels * 4, in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, C).

        Returns:
            torch.Tensor: Output tensor of the same shape (B, S, C).
        """
        attn_input = self.norm1(x)
        attn_output, _ = self.multi_head_attention(query=attn_input, key=attn_input, value=attn_input)
        x = x + attn_output

        ffn_input = self.norm2(x)
        ffn_output = self.ffn(ffn_input)
        out = x + ffn_output
        return out

class SpatialLayer(nn.Module):
    """
    Temporal attention module for video depth estimation.
    This module stacks multiple TemporalTransformerBlock layers.
    """
    def __init__(self, in_channels: int, num_attention_heads: int, num_transformer_block: int, pos_embedding_type: str = 'ape2d'):
        """
        Initialize temporal attention module.

        Args:
            in_channels (int): Input feature dimension (C).
            num_attention_heads (int): Number of parallel attention heads for each block.
            num_transformer_block (int): Number of transformer blocks to stack.
            temporal_max_len (int): Number of frames in sequence (S).
            pos_embedding_type (str): Positional encoding type.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_attention_heads
        self.num_blocks = num_transformer_block
        self.pe = pos_embedding_type  # Todo: Implement positional encoding if needed

        if in_channels % self.num_heads != 0:
            raise ValueError(f"'in_channels' ({in_channels}) must be divisible by 'num_attention_heads' ({self.num_heads}).")

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(in_channels=in_channels, num_heads=self.num_heads)
            for _ in range(self.num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the temporal attention module.
        attention is applied across the sequence dimension (W H).
        (b s) (h w) c

        Args:
            x (torch.Tensor): Input tensor of shape (b s c h w).
                              Positional encoding should be added beforehand.

        Returns:
            torch.Tensor: Output tensor of the same shape (b s c h w).
        """
        B, S, C, H, W = x.shape
        x = rearrange(x, "b s c h w -> (b s) (h w) c")
        for block in self.transformer_blocks:
            x = block(x)
        x = rearrange(x, "(b s) (h w) c -> b s c h w", b = B, s= S, h = H, w = W)
        return x

class TemporalLayer(nn.Module):
    """
    Temporal attention module for video depth estimation.
    This module stacks multiple TemporalTransformerBlock layers.
    """
    def __init__(self, in_channels: int, num_attention_heads: int, num_transformer_block: int, temporal_max_len: int, pos_embedding_type: str = 'ape'):
        """
        Initialize temporal attention module.

        Args:
            in_channels (int): Input feature dimension (C).
            num_attention_heads (int): Number of parallel attention heads for each block.
            num_transformer_block (int): Number of transformer blocks to stack.
            temporal_max_len (int): Number of frames in sequence (S).
            pos_embedding_type (str): Positional encoding type.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_heads = num_attention_heads
        self.num_blocks = num_transformer_block
        self.pe = pos_embedding_type  # Todo: Implement positional encoding if needed

        if in_channels % self.num_heads != 0:
            raise ValueError(f"'in_channels' ({in_channels}) must be divisible by 'num_attention_heads' ({self.num_heads}).")

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(in_channels=in_channels, num_heads=self.num_heads)
            for _ in range(self.num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the temporal attention module.
        attention is applied across the sequence dimension (S).
        (b h w) s c

        Args:
            x (torch.Tensor): Input tensor of shape (b s c h w).
                              Positional encoding should be added beforehand.

        Returns:
            torch.Tensor: Output tensor of the same shape (b s c h w).
        """
        B, S, C, H, W = x.shape
        x = rearrange(x, "b s c h w -> (b h w) s c")
        for block in self.transformer_blocks:
            x = block(x)
        x = rearrange(x, "(b h w) s c -> b s c h w", b = B, h = H, w = W)
        return x
    

class VideoDepthAnythingHeadV2(nn.Module):
    def __init__(self, 
                 sequence_length: int = 8,
                 pe='ape'
                 ):
        """
        Initialize Video Depth Anything head
        
        Args:
            input_dim: Input feature dimension from encoder
            sequence_length: Number of frames in sequence
            img_size: (H, W) image size
            finetune: Whether to allow fine-tuning
        """
        super().__init__()
        
        temporal_layer_kwargs = EasyDict(
            num_attention_heads                = 8,
            num_transformer_block              = 2,
            # num_attention_blocks               = 2,
            temporal_max_len                   = sequence_length,
            pos_embedding_type                 = pe)

        self.feature_channels = [96, 192, 384, 768]
        self.temporal_layers = nn.ModuleList([
            TemporalLayer(in_channels=self.feature_channels[0], **temporal_layer_kwargs),
            TemporalLayer(in_channels=self.feature_channels[1], **temporal_layer_kwargs),
            TemporalLayer(in_channels=self.feature_channels[2], **temporal_layer_kwargs),
            TemporalLayer(in_channels=self.feature_channels[3], **temporal_layer_kwargs)
        ])
        
        spatial_layer_kwargs = EasyDict(
            num_attention_heads                = 8,
            num_transformer_block              = 1,
            pos_embedding_type                 = pe)

        self.spatial_layers = nn.ModuleList([
            SpatialLayer(in_channels=self.feature_channels[0], **spatial_layer_kwargs),
            SpatialLayer(in_channels=self.feature_channels[1], **spatial_layer_kwargs),
            SpatialLayer(in_channels=self.feature_channels[2], **spatial_layer_kwargs),
            SpatialLayer(in_channels=self.feature_channels[3], **spatial_layer_kwargs)
        ])
        

        upscale_layer_kwargs = EasyDict(
            kernel_size = 4,
            stride = 2,
            padding = 1
        )
    
        self.upscale_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(self.feature_channels[1], self.feature_channels[0], **upscale_layer_kwargs),
                nn.BatchNorm2d(self.feature_channels[0]),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(               
                nn.ConvTranspose2d(self.feature_channels[2], self.feature_channels[1], **upscale_layer_kwargs),
                nn.BatchNorm2d(self.feature_channels[1]),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(self.feature_channels[3], self.feature_channels[2], **upscale_layer_kwargs),
                nn.BatchNorm2d(self.feature_channels[2]),
                nn.ReLU(inplace=True)
            ),
        ])
        self.final_upscale_layer = nn.Sequential(
            nn.ConvTranspose2d(self.feature_channels[0], self.feature_channels[0] // 2, **upscale_layer_kwargs),
            nn.BatchNorm2d(self.feature_channels[0] // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.feature_channels[0] // 2, 3, **upscale_layer_kwargs),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        #     self.feature_channels = [96, 192, 384, 768]
        self.fusion_layer = nn.ModuleList([
            nn.Conv2d(self.feature_channels[1], self.feature_channels[0], kernel_size=3, padding=1),
            nn.Conv2d(self.feature_channels[2], self.feature_channels[1], kernel_size=3, padding=1),
            nn.Conv2d(self.feature_channels[3], self.feature_channels[2], kernel_size=3, padding=1),
        ])
        
        '''
        fusion_layer_kwargs = EasyDict(
            activation=nn.ReLU(False),
            bn=True,
            align_corners=True,
        )
        
        self.fusion_layers = nn.ModuleList([
            FusionLayer(lhs_channels=self.feature_channels[1], out_channels=self.feature_channels[0], **fusion_layer_kwargs),
            FusionLayer(lhs_channels=self.feature_channels[2], out_channels=self.feature_channels[1], **fusion_layer_kwargs),
            FusionLayer(lhs_channels=self.feature_channels[3], out_channels=self.feature_channels[2], **fusion_layer_kwargs),
        ])
        self.final_fusion_layer = FusionLayer(lhs_channels=self.feature_channels[0], out_channels=3, rhs_size=(224, 224), **fusion_layer_kwargs)  # todo : code smell
        '''

    def forward(self, features) -> torch.Tensor:
        '''
        features: [B, S, C, H, W] * 4
        '''
        temporal_feature = self.temporal_layers[3](features[3])  # [B, S, C, H, W]
        spatial_feature = self.spatial_layers[3](temporal_feature)  # [B, S, C, H, W]
        
        # fusion
        B, S, C, H, W = spatial_feature.shape
        upscaled_feature = spatial_feature.view(B * S, C, H, W)
        for i in reversed(range(len(self.upscale_layers))):
            upscaled_feature = self.upscale_layers[i](upscaled_feature)  # [B * S, C, H, W]
            B_, S_, C_, H_, W_ = features[i].shape
            upscaled_feature = self.fusion_layer[i](torch.cat([upscaled_feature, features[i].view(B_ * S_, C_, H_, W_)], dim=1))

        BS_, C_, H_, W_ = upscaled_feature.shape
        output = self.final_upscale_layer(upscaled_feature)
        output = output.view(B, S, 3, H_ * 4, W_ * 4)
        return output

    """
    def forward(self, features) -> torch.Tensor:
        '''
        features: [B, S, C, H, W] * 4
        '''
        # temporal attention
        temporal_features = []
        for i, x in enumerate(features):
            temporal_feature= self.temporal_layers[i](x)  # [B, S, C, H, W]
            temporal_features.append(temporal_feature)
        
        # spatial attention
        spatial_features = []
        for i, x in enumerate(temporal_features):
            spatial_feature= self.spatial_layers[i](x)  # [B, S, C, H, W]
            spatial_features.append(spatial_feature)
        
        # fusion
        lhs_fusion_features = spatial_features[3]
        for i, x in reversed(list(enumerate(self.fusion_layers))):
            fusion_feature = self.fusion_layers[i](lhs_fusion_features, spatial_features[i])  # [B, S, C, H, W]
            lhs_fusion_features = fusion_feature

        complete_fusion_feature = self.final_fusion_layer(lhs_fusion_features)
        
        return complete_fusion_feature
    """