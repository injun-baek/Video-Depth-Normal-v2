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

#class ZeroConv1x1(nn.Conv2d):
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

def match_seq_to_first_median(input_depth: torch.Tensor, eps: float = 1e-8, scale: bool = True) -> torch.Tensor:
    """
    input_depth: (B, S, H, W)  [이미 65535로 나눈 상태라고 가정]
    s=0 프레임의 median/MAD에 s=1..S-1 프레임을 맞춤.
    scale=False면 median만 맞추는 shift 정렬만 수행.

    MAD = median(|x - median(x)|)
    """
    assert input_depth.ndim == 4, "input must be (B, S, H, W)"
    x = input_depth

    B, S, H, W = x.shape
    x_flat = x.view(B, S, -1)

    # 각 프레임의 median (H,W 축)
    med = x_flat.median(dim=-1).values.view(B, S, 1, 1)

    # 각 프레임의 MAD (H,W 축)
    mad = (x - med).abs().view(B, S, -1).median(dim=-1).values.view(B, S, 1, 1)

    ref_med = med[:, :1, ...]  # (B,1,1,1) s=0
    ref_mad = mad[:, :1, ...]  # (B,1,1,1)

    out = x.clone()

    # s=1..S-1 프레임 현재 통계
    cur_med = med[:, 1:, ...]
    cur_mad = mad[:, 1:, ...]
    denom = torch.where(cur_mad > eps, cur_mad, torch.ones_like(cur_mad))  # 0 분모 방지

    if scale:
        # median + scale(MAD) 맞춤
        out[:, 1:, ...] = (x[:, 1:, ...] - cur_med) / denom * ref_mad + ref_med
    else:
        # median만 맞춤(shift만 적용)
        out[:, 1:, ...] = x[:, 1:, ...] - cur_med + ref_med

    return out

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
        B, S, H_orig, W_orig = input_depth.shape
        input_depth = input_depth / self.max_depth
        
        input_depth_bs1hw = input_depth.unsqueeze(2).flatten(0, 1)
        scale_bs11 = self.scale_head(input_depth_bs1hw).unflatten(0, (B, S)).squeeze(2)
        input_depth = input_depth * scale_bs11
        
        resize_depth = F.interpolate(input_depth, size=(224, 224), mode='bilinear', align_corners=True)
        _, _, H, W = resize_depth.shape
        
        #depth_expanded = input_depth.unsqueeze(2)
        depth_expanded = resize_depth.unsqueeze(2)
        if self.input_normal:
            normal = normal_vector(depth_expanded)
            depth_expanded = torch.cat([depth_expanded, normal[:, :, :2, :, :]], dim=2)
        else:
            depth_expanded = depth_expanded.expand(-1, -1, 3, -1, -1)
            
        depth_reshaped = depth_expanded.view(B * S, 3, H, W)
        
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(depth_reshaped, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depth = self.temporal_head(features, patch_h, patch_w, S)[0]
        depth = F.interpolate(depth, size=(H_orig, W_orig), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        output_depth = depth.squeeze(1).unflatten(0, (B, S))
        
        if self.use_residual:
            output_depth = self.shift_head(output_depth.flatten(0, 1).unsqueeze(1))
            output_depth = input_depth + output_depth.squeeze(1).unflatten(0, (B, S))
                        
        return output_depth * self.max_depth

    def infer_video_depth(self, frames, target_fps, input_size=518, device='cuda', fp32=False):
        frame_height, frame_width = frames[0].shape[:2]   # (8, 384, 384, 3)
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        frame_list = [frames[i] for i in range(frames.shape[0])]
        frame_step = INFER_LEN - OVERLAP
        org_video_len = len(frame_list)
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len

        depth_list = []
        pre_input = None
        for frame_id in tqdm(range(0, org_video_len, frame_step)):
            cur_list = []
            for i in range(INFER_LEN):
                cur_list.append(torch.from_numpy(transform({'image': frame_list[frame_id+i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0))
            cur_input = torch.cat(cur_list, dim=1).to(device)
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            with torch.no_grad():
                # with torch.autocast(device_type=device.split(':')[0], enabled=(not fp32)):
                with torch.autocast(device_type="cuda", enabled=(not fp32)):
                    depth = self.forward(cur_input) # depth shape: [1, T, H, W]

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
            depth_list += [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]

            pre_input = cur_input

        del frame_list
        gc.collect()

        depth_list_aligned = []
        ref_align = []
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]

        for frame_id in range(0, len(depth_list), INFER_LEN):
            if len(depth_list_aligned) == 0:
                depth_list_aligned += depth_list[:INFER_LEN]
                for kf_id in kf_align_list:
                    ref_align.append(depth_list[frame_id+kf_id])
            else:
                curr_align = []
                for i in range(len(kf_align_list)):
                    curr_align.append(depth_list[frame_id+i])
                scale, shift = compute_scale_and_shift(np.concatenate(curr_align),
                                                       np.concatenate(ref_align),
                                                       np.concatenate(np.ones_like(ref_align)==1))

                pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                post_depth_list = depth_list[frame_id+align_len:frame_id+OVERLAP]
                for i in range(len(post_depth_list)):
                    post_depth_list[i] = post_depth_list[i] * scale + shift
                    post_depth_list[i][post_depth_list[i]<0] = 0
                depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

                for i in range(OVERLAP, INFER_LEN):
                    new_depth = depth_list[frame_id+i] * scale + shift
                    new_depth[new_depth<0] = 0
                    depth_list_aligned.append(new_depth)

                ref_align = ref_align[:1]
                for kf_id in kf_align_list[1:]:
                    new_depth = depth_list[frame_id+kf_id] * scale + shift
                    new_depth[new_depth<0] = 0
                    ref_align.append(new_depth)

        depth_list = depth_list_aligned

        return np.stack(depth_list[:org_video_len], axis=0), target_fps

