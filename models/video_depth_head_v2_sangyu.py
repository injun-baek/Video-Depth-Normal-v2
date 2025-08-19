import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fusion_block import FusionLayer
from einops import rearrange
from easydict import EasyDict

def get_sinusoid_encoding_table(length: int, dim: int, dtype=torch.float32) -> torch.Tensor:
    """Return [length, dim] sinusoidal embeddings (non‑trainable)."""
    position = torch.arange(length, dtype=dtype).unsqueeze(1)  # [L, 1]
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=dtype) * -(math.log(10000.0) / dim))
    emb = torch.zeros((length, dim), dtype=dtype)
    emb[:, 0::2] = torch.sin(position * div_term)
    emb[:, 1::2] = torch.cos(position * div_term)
    return emb  # [L, D]

class UpSampleAdd(nn.Module):
    """bilinear ↑ + 3×3 Conv  →  skip‑add (1×1 Conv)"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(                       # ①  ↑  & refine
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.skip_proj = nn.Conv2d(skip_ch, out_ch, 1)   # ②  skip 1×1 proj

    def forward(self, x, skip):
        x   = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)        # bilinear ↑
        x   = self.conv(x)
        x   = x + self.skip_proj(skip)                  # **element‑wise add**
        return x

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
                 pe='ape',
                 attention_feature_levels: list = [2, 3]
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

        self.feature_channels = [96, 192, 384, 768]
        # self.feature_channels = [112, 224, 448, 896]  # base+
        # --- positional embedding ----------------------------------------------------------------
        self.pos_embedding_type = pe
        if self.pos_embedding_type == "ape":
            self.pos_embeds = nn.ParameterList([
                nn.Parameter(torch.zeros(sequence_length, self.feature_channels[0])),
                nn.Parameter(torch.zeros(sequence_length, self.feature_channels[1])),
                nn.Parameter(torch.zeros(sequence_length, self.feature_channels[2])),
                nn.Parameter(torch.zeros(sequence_length, self.feature_channels[3])),
            ])
            nn.init.trunc_normal_(self.pos_embeds[0], std=0.02)
            nn.init.trunc_normal_(self.pos_embeds[1], std=0.02)
            nn.init.trunc_normal_(self.pos_embeds[2], std=0.02)
            nn.init.trunc_normal_(self.pos_embeds[3], std=0.02)
        else:  # "none" or "sine"
            self.register_parameter("pos_embeds", None)

        temporal_layer_kwargs = EasyDict(
            num_attention_heads                = 8,
            num_transformer_block              = 4,
            temporal_max_len                   = sequence_length,
            pos_embedding_type                 = pe)

        self.temporal_layers_first = nn.ModuleList([
            TemporalLayer(in_channels=self.feature_channels[0], **temporal_layer_kwargs),
            TemporalLayer(in_channels=self.feature_channels[1], **temporal_layer_kwargs),
            TemporalLayer(in_channels=self.feature_channels[2], **temporal_layer_kwargs),
            TemporalLayer(in_channels=self.feature_channels[3], **temporal_layer_kwargs)
        ])
        
        self.temporal_layers_second = nn.ModuleList([
            TemporalLayer(in_channels=self.feature_channels[0], **temporal_layer_kwargs),
            TemporalLayer(in_channels=self.feature_channels[1], **temporal_layer_kwargs),
            TemporalLayer(in_channels=self.feature_channels[2], **temporal_layer_kwargs),
            TemporalLayer(in_channels=self.feature_channels[3], **temporal_layer_kwargs)
        ])
        
        spatial_layer_kwargs = EasyDict(
            num_attention_heads                = 8,
            num_transformer_block              = 4,
            pos_embedding_type                 = pe)

        self.spatial_layers_first = nn.ModuleList([
            SpatialLayer(in_channels=self.feature_channels[0], **spatial_layer_kwargs),
            SpatialLayer(in_channels=self.feature_channels[1], **spatial_layer_kwargs),
            SpatialLayer(in_channels=self.feature_channels[2], **spatial_layer_kwargs),
            SpatialLayer(in_channels=self.feature_channels[3], **spatial_layer_kwargs)
        ])

        self.spatial_layers_second = nn.ModuleList([
            SpatialLayer(in_channels=self.feature_channels[0], **spatial_layer_kwargs),
            SpatialLayer(in_channels=self.feature_channels[1], **spatial_layer_kwargs),
            SpatialLayer(in_channels=self.feature_channels[2], **spatial_layer_kwargs),
            SpatialLayer(in_channels=self.feature_channels[3], **spatial_layer_kwargs)
        ])
        
        self.attention_feature_levels = attention_feature_levels
        for lvl in self.attention_feature_levels:
            if lvl not in range(len(self.feature_channels)):
                raise ValueError("attention_feature_levels must contain indices between 0 and 3 inclusive")
    
        self.upscale_layers = nn.ModuleList([
            UpSampleAdd(self.feature_channels[3], self.feature_channels[2], self.feature_channels[2]),   # stride 32 → 16
            UpSampleAdd(self.feature_channels[2], self.feature_channels[1], self.feature_channels[1]),   # stride 16 → 8
            UpSampleAdd(self.feature_channels[1],  self.feature_channels[0],  self.feature_channels[0])    # stride 8  → 4
        ])
        self.final_upscale_layer = nn.Sequential(
            nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=True),
            nn.Conv2d(self.feature_channels[0], self.feature_channels[0], 3, 1, 1, bias=False), nn.BatchNorm2d(self.feature_channels[0]), nn.ReLU(True),
            nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=True),
            nn.Conv2d(self.feature_channels[0], self.feature_channels[0], 3, 1, 1, bias=False), nn.BatchNorm2d(self.feature_channels[0]), nn.ReLU(True),
            nn.Conv2d(self.feature_channels[0], 48, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(48,  3, 3, 1, 1)   # depth / dx / dy
        )
        
        self.fusion_layer = nn.ModuleList([
            nn.Conv2d(self.feature_channels[1], self.feature_channels[0], kernel_size=3, padding=1),
            nn.Conv2d(self.feature_channels[2], self.feature_channels[1], kernel_size=3, padding=1),
            nn.Conv2d(self.feature_channels[3], self.feature_channels[2], kernel_size=3, padding=1),
        ])
        
    def _maybe_process(self, lvl: int, feat: torch.Tensor) -> torch.Tensor:
        """Run Temporal→Spatial on *feat* if *lvl* is in ``self.attention_feature_levels``"""
        if lvl in self.attention_feature_levels:
            B, S, C, H, W = feat.shape
            feat = rearrange(feat, "b s c h w -> (b h w) s c")
            if self.pos_embedding_type == "ape" and self.pos_embeds is not None:
                feat = feat + self.pos_embeds[lvl][:S].unsqueeze(0)  # truncate in case S < max_len
            elif self.pos_embedding_type == "sine":
                pe = get_sinusoid_encoding_table(S, C, dtype=feat.dtype).to(feat.device)
                feat = feat + pe.unsqueeze(0)
            feat = rearrange(feat, "(b h w) s c -> b s c h w", b = B, h = H, w = W)
            
            feat = self.temporal_layers_first[lvl](feat)
            feat = self.spatial_layers_first[lvl](feat)
            feat = self.temporal_layers_second[lvl](feat)
            feat = self.spatial_layers_second[lvl](feat)

        return feat

    def forward(self, features) -> torch.Tensor:
        
        if len(features) != 4:
            raise ValueError("Expected 4 levels of encoder features (stride 4→32)")

        # 1) optionally process each level
        processed = [self._maybe_process(i, f) for i, f in enumerate(features)]

        # 2) deepest feature (index 3) is the "top" of decoder hierarchy
        top = processed[3]
        B, S, C, H, W = top.shape
        x = top.reshape(B * S, C, H, W)

        # 3) construct skip list (level 2 → 0) – each already processed if requested
        skip_feats = [processed[2], processed[1], processed[0]]
        for up, skip in zip(self.upscale_layers, skip_feats):
            x = up(x, skip.reshape(B * S, *skip.shape[2:]))

        # 4) final refinement to full resolution & regression to depth/dx/dy
        out = self.final_upscale_layer(x)
        out = out.view(B, S, 3, out.shape[-2], out.shape[-1])
        return out