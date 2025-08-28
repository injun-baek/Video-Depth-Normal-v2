# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from utils.normal_utils import sobel_ix_iy, normal_vector

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
from tqdm import tqdm
import numpy as np
import gc

from video_depth_anything.dinov2 import DINOv2
from video_depth_anything.dpt_temporal import DPTHeadTemporal
from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from utils.util import compute_scale_and_shift, get_interpolate_frames

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
INTERP_LEN = 8

class TanhToExp(nn.Module):
    def __init__(self, max_log_scale: float = 0.2):
        super().__init__()
        self.max_log_scale = max_log_scale  # scale ∈ [e^-m, e^m]
    def forward(self, x):
        return torch.exp(torch.tanh(x) * self.max_log_scale)

class TanhScale(nn.Module):
    def __init__(self, out_ch: int, max_scale: float = 1.0, init_log_scale: float = -4.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.full((out_ch,), init_log_scale))
        self.max_scale = max_scale

    def forward(self, x: torch.Tensor):
        scale = torch.exp(self.log_scale).view(1, -1, 1, 1) * self.max_scale
        return torch.tanh(x) * scale

class ZeroConv(nn.Conv2d):
    def __init__(self, in_ch, out_ch, kernel_size=1, padding=0, bias=True):
        super().__init__(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=bias)
        nn.init.zeros_(self.weight)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class GlobalQuantilePool2d(nn.Module):
    def __init__(self, q: float = 0.5):
        super().__init__()
        self.q = q
    def forward(self, x):  # x: (N, C, H, W)
        # PyTorch 1.9+ : quantile은 미분 가능 (선형 보간)
        try:
            return torch.quantile(x, self.q, dim=(-2, -1), keepdim=True)  # (N,C,1,1)
        except TypeError:
            N, C, H, W = x.shape
            return torch.quantile(x.view(N, C, H*W), self.q, dim=-1, keepdim=True).unsqueeze(-1)

class GlobalScaleHead(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, max_log_scale=0.2):
        super().__init__()
        self.feat = nn.Sequential(
            GlobalQuantilePool2d(0.5),
            ZeroConv(in_channels, out_channels, kernel_size=1, padding=0, bias=True)
        )
        self.to_scale = TanhToExp(max_log_scale=max_log_scale)

    def forward(self, x):  # x: (B*S, 1, H, W)
        g = self.feat(x)              # (B*S, 1, 1, 1)
        s = self.to_scale(g)          # (B*S, 1, 1, 1)  ; 양수
        return s

class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        max_depth = 65535,
        pe='ape',
        use_residual = True,
        input_normal = True
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }

        self.max_depth = max_depth
        self.use_residual = use_residual
        self.input_normal = input_normal
        
        self.encoder = encoder
        
        self.pretrained = DINOv2(model_name=encoder)
        self.scale_head = GlobalScaleHead(in_channels=1, out_channels=1, max_log_scale=1.0)
        self.temporal_head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)
        self.shift_head = nn.Sequential(ZeroConv(1, 1))

    def forward(self, input_depth):
        B, S, H, W = input_depth.shape
        input_depth = input_depth / self.max_depth
        
        input_depth_bs1hw = input_depth.unsqueeze(2).flatten(0, 1)
        scale_bs11 = self.scale_head(input_depth_bs1hw).unflatten(0, (B, S)).squeeze(2)
        input_depth = input_depth * scale_bs11
            
        depth_expanded = input_depth.unsqueeze(2)
        if self.input_normal:
            normal = normal_vector(depth_expanded)
            depth_expanded = torch.cat([depth_expanded, normal[:, :, :2, :, :]], dim=2)
        else:
            depth_expanded = depth_expanded.expand(-1, -1, 3, -1, -1)
            
        depth_reshaped = depth_expanded.view(B * S, 3, H, W)
        
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(depth_reshaped, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depth = self.temporal_head(features, patch_h, patch_w, S)[0]
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        output_depth = depth.squeeze(1).unflatten(0, (B, S))
        
        if self.use_residual:
            output_depth = self.shift_head(output_depth.flatten(0, 1).unsqueeze(1))
            output_depth = input_depth + output_depth.squeeze(1).unflatten(0, (B, S))
                        
        return output_depth * self.max_depth
