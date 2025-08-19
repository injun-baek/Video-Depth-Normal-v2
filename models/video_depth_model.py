from utils.normal_utils import sobel_ix_iy, normal_vector

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.dinov2_encoder import DINOv2Encoder
from models.hiera_image_encoder import HieraImageEncoder
from models.video_depth_head import VideoDepthAnythingHead
from models.video_depth_head_v2_sangyu import VideoDepthAnythingHeadV2
# from utils.common_utils import print_model_memory_info

class VideoDepthEstimationModel(nn.Module):
    """
    Complete Video Depth Estimation Model combining DINOv2 encoder and Video Depth Anything head
    """
    
    def __init__(
        self,
        sequence_length,
        attention_feature_levels=[2, 3],
        encoder='hiera_base_224',
        encoder_finetune=False,
        use_residual=False,
        use_final_relu=False,
        use_depth_feature=True,
        use_rgb_feature=True
    ):

        super().__init__()
        
        self.use_residual = use_residual
        self.use_final_relu = use_final_relu
        self.use_depth_feature = use_depth_feature
        self.use_rgb_feature = use_rgb_feature

        self.img_encoder = HieraImageEncoder(
            model_name=encoder,
            finetune=False
        )
        
        self.encoder = HieraImageEncoder(
            model_name=encoder,
            finetune=encoder_finetune
        )
       
        self.head = VideoDepthAnythingHeadV2(
            #input_dim= self.encoder.feature_dim,
            sequence_length=sequence_length,
            attention_feature_levels=attention_feature_levels
        )
        
        #print_model_memory_info(self.encoder, "Hiera Encoder")
        #print_model_memory_info(self.head, "Video Depth Head")
        #print_model_memory_info(self, "Complete Model")
           
    def forward(self, depth: torch.Tensor, img) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [B, S, H, W] input depth, dx, dy sequence
            
        Returns:
            output: [B, S, 3, H, W] predicted depth, dx, dy
        """
        B, S, H, W = depth.shape
        depth_expanded = depth.unsqueeze(2)
        if True:  # TODO: use self.normal
            normal = normal_vector(depth_expanded)
            depth_expanded = torch.cat([depth_expanded, normal[:, :, :2, :, :]], dim=2)
        else:
            depth_expanded = depth_expanded.expand(-1, -1, 3, -1, -1)
            
        depth_reshaped = depth_expanded.view(B * S, 3, H, W)
        _, depth_features = self.encoder(depth_reshaped)
        
        img_reshaped = img.view(B * S, 3, H, W)
        _, img_features = self.img_encoder(img_reshaped)

        features = []
        for depth_feature, img_feature in zip(depth_features, img_features):
            if self.use_depth_feature and self.use_rgb_feature:
                features.append(depth_feature + img_feature)
            elif self.use_depth_feature:
                features.append(depth_feature)
            elif self.use_rgb_feature:
                features.append(img_feature)

        for i, feature in enumerate(features):
            BS, H_, W_, D_ = feature.shape
            features[i] = feature.view(B, S, D_, H_, W_)
        
        output = self.head(features)  # [B, S, C, H, W]
        # torch.Size([B, S, 3, H, W])

        B_, S_, C_, H_, W_ = output.shape
        if (H_, W_) != (H, W):
            output_reshape = F.interpolate(output.view(B_ * S_, C_, H_, W_), size=(H, W), mode="bilinear", align_corners=True)
            output = output_reshape.view(B_, S_, C_, H, W)

        output_depth = output[:,:,0,:,:]
        if self.use_residual:
            output_depth = output_depth + depth
        
        # todo: if normal
        dx = output[:,:,1,:,:]
        dy = output[:,:,2,:,:]
        ones  = torch.ones_like(depth)
        normal = torch.stack([-dx, -dy, ones], dim=2)

        if self.use_final_relu:
            return torch.relu(output_depth), normal  # relu ?
        return output_depth, normal

    def set_finetune_modes(self, encoder_finetune: bool = None, head_finetune: bool = None):
        """Set fine-tuning modes for encoder and head separately"""
        if encoder_finetune is not None:
            self.encoder.set_finetune(encoder_finetune)
        if head_finetune is not None:
            self.head.set_finetune(head_finetune)
