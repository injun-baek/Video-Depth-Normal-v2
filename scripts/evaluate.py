#!/usr/bin/env python3
"""
Validation script for video depth estimation model
"""

import argparse
import numpy as np
import os
from tqdm import tqdm
import sys
import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Add the project root directory to the Python path
# This allows us to import modules from sibling directories like 'data', 'loss', etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from depth_anything_v2.depth_anything_v2 import DepthAnythingV2


from torch.utils.data import ConcatDataset
from data.dataset_factory import (
    create_sampler,
    create_data_loader,
    create_datasets,
)

INPUT_SIZE = None
RANDOM_SEED = 42


import os
import sys
import yaml
import time
import torch
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from torch.utils.data import DataLoader, ConcatDataset
from eval_depthcrafter.eval import eval_single_by_data, eval_metrics

from data.dataset_factory import create_datasets, create_data_loader

RANDOM_SEED = 42

def preprocess_depth_sequences(depth_batch):
    def batch_wise_min_max(x):
        B = x.shape[0]
        x_flat = x.view(B, -1)
        x_min = x_flat.min(dim=1, keepdim=True)[0]
        x_max = x_flat.max(dim=1, keepdim=True)[0]
        x_scaled = (x_flat - x_min) / (x_max - x_min + 1e-8)
        return x_scaled.view_as(x)
    
    transform_list = [
        #transforms.Resize(INPUT_SIZE, interpolation=InterpolationMode.NEAREST),
        #transforms.CenterCrop(INPUT_SIZE),
        transforms.Lambda(lambda x: x.clamp(min=0))
    ]

    transform_norm = transforms.Compose(transform_list + [
        # transforms.ToTensor(),
        # transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    
    B, S, C, H, W = depth_batch.shape

    reshaped_batch = depth_batch.view(B * S, C, H, W)
    normalized_batch = transform_norm(reshaped_batch)
    normalized_batch = normalized_batch.view(B, S, C, INPUT_SIZE, INPUT_SIZE)

    depths = normalized_batch.squeeze(2)
    depths = 1. / torch.clamp(depths, min=1e-8)
    depths = batch_wise_min_max(depths)
    return depths


def preprocess_rgb_sequences(rgb_batch):
    transform_list = [
        #transforms.Resize(INPUT_SIZE, interpolation=InterpolationMode.BICUBIC),
        #transforms.CenterCrop(INPUT_SIZE),
        transforms.Lambda(lambda x: x.clamp(0, 1))
    ]

    transform_norm = transforms.Compose(transform_list + [
        #transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    
    B, S, C, H, W = rgb_batch.shape

    reshaped_batch = rgb_batch.view(B * S, C, H, W)
    normalized_batch = transform_norm(reshaped_batch)

    return normalized_batch.view(B, S, C, INPUT_SIZE, INPUT_SIZE)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

@torch.no_grad()
def evaluate(args):
    print("Loading validation configuration...")
    device = torch.device(args.device)

    print(f"Using device: {device}")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    model = DepthAnythingV2(**model_configs[args.encoder])
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.to(device).eval()

    dataset_names = [args.eval_data]
    val_datasets = create_datasets(dataset_names, config_file_path=args.config, split='val', random_seed=RANDOM_SEED)
    val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    val_loader = create_data_loader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    max_batch_limit = args.max_eval_count
    
    infer_elapsed_time = 0
    epoch_start_time = time.time()
    
    metric_vals = []
    S = None
    for batch_idx, batch in enumerate(tqdm(val_loader)):
        if batch_idx >= max_batch_limit:
            break
        
        videos = batch['rgb'].to(device)
        depths = batch['depth'].to(device)
        # masks = batch['mask'].to(device).squeeze(2)
        
        videos = preprocess_rgb_sequences(videos)
        depths = 1. / torch.clamp(depths, min=1e-8)
        # depths = preprocess_depth_sequences(depths)
        
        # predictions = batch['depth_anything_v2'].to(device).squeeze(2)  # for depth anything v2
        
        pred_depths = []
        model.clear_memory()
        S = videos.shape[1]
        for t in range(S):
            video_frame = videos[:, t, :, :, :].to(device)
            infer_start_time = time.time()
            pred_depth = model(video_frame)
            infer_elapsed_time += time.time() - infer_start_time
            pred_depths.append(pred_depth)
        predictions = torch.stack(pred_depths, dim=1)

        B, S = depths.shape[0], depths.shape[1]
        for b in range(B):
            metric = eval_single_by_data(
                predictions[b].cpu().numpy(),
                depths[b].squeeze(1).cpu().numpy(),
                #depths[b].cpu().numpy(),
                device=args.device,
                seq_len=S,
                domain="disp",
                dataset_max_depth=70
            )
            metric_vals.append(metric)


    elapsed_time = time.time() - epoch_start_time
    inferred_batch_len = min(max_batch_limit, len(val_loader))
    total_frame_len = inferred_batch_len * B * S
    
    names_to_print = {
        "abs_difference": "L1",
        "abs_relative_difference": "AbsRel",
        "rmse_linear": "L2",
        "delta1_acc": "D1",
        "delta2_acc": "D2",
        "delta3_acc": "D3",
        "temporal_gradient_matching_error": "TGM",
    }
    
    print(f"Total Elapsed time: {elapsed_time:.2f} seconds.")
    print(f"Total frames processed: {total_frame_len}.")
    print(f"Infer Elapsed time: {infer_elapsed_time:.2f} seconds.")
    print("=" * 80)
    print(f"TPF(msec): {infer_elapsed_time * 1000. / total_frame_len:.4f}")
    for i, metric_name in enumerate(eval_metrics):
        mean = np.nanmean(np.array(metric_vals)[:, i])
        print(f"{names_to_print[metric_name]} mean: {mean:.4f}")
    
    print("Done")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate Video Depth Estimation Model')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for validation (e.g., cuda:0, cpu).')
    parser.add_argument('--checkpoint', type=str, default='trained_models/E258_memory_depth_anything_vits_epoch_3.pth', help='Path to validation checkpoint file')
    parser.add_argument('--config', type=str, default='configs/config_eval.yaml', help='Path to validation configuration file')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loading')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'], help='Encoder type to use for the model')
    parser.add_argument('--input_size', type=int, default=518, help='Input size for the model')
    parser.add_argument('--eval_data', type=str, default='Sintel', choices=['Sintel', 'PointOdyssey'], help='Evaluation dataset to use')
    parser.add_argument('--max_eval_count', type=int, default=200, help='Maximum number of evaluation batches to process')

    args = parser.parse_args()
    INPUT_SIZE = args.input_size
    evaluate(args)