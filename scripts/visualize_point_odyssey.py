import argparse
import os
import sys
from typing import Tuple
import torch
from tqdm import tqdm
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Add the project root directory to the Python path
# This allows us to import modules from sibling directories like 'data', 'loss', etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loss.loss import compute_scale_and_shift

from torch.utils.data import ConcatDataset

INPUT_SIZE = 518
RANDOM_SEED = 42

def compute_spatial_mask(depth_batch, lower_threshold=0.01, upper_threshold=5000):
    """ Computes a mask to exclude regions with large spatial changes in depth.
    Args:
        depth_batch (torch.Tensor): The ground truth depth map tensor.
                                    Shape: (B, T, H, W).
        lower_threshold (float): Lower threshold for spatial depth change.
        upper_threshold (float): Upper threshold for spatial depth change.
    Returns:
        torch.Tensor: A boolean mask. The first frame's mask is all True.
                      For subsequent frames, it's True where the spatial
                      change is less than the threshold.
                      Shape: (B, T, H, W).
    """
    spatial_mask_sub = (depth_batch > lower_threshold) & (depth_batch < upper_threshold)
    return spatial_mask_sub


def compute_temporal_mask(depth_batch, threshold):
    """
    Computes a mask to exclude regions with large temporal changes in depth.
    The mask is True for regions where |depth_t - depth_{t-1}| < threshold.

    Args:
        depth_batch (torch.Tensor): The ground truth depth map tensor.
                                    Shape: (B, T, H, W).
        threshold (float): The threshold for the temporal depth change.

    Returns:
        torch.Tensor: A boolean mask. The first frame's mask is all True.
                      For subsequent frames, it's True where the temporal
                      change is less than the threshold.
                      Shape: (B, T, H, W).
    """
    if depth_batch.shape[1] <= 1:
        return torch.ones_like(depth_batch, dtype=torch.bool)

    diff = torch.abs(depth_batch[:, 1:] - depth_batch[:, :-1])
    temporal_mask_sub = diff < threshold
    first_frame_mask = torch.ones_like(depth_batch[:, :1], dtype=torch.bool)
    final_mask = torch.cat([first_frame_mask, temporal_mask_sub], dim=1)
    
    return final_mask

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

def preprocess_rgb_viz_sequences(rgb_batch):
    transform_list = [
        #transforms.Resize(INPUT_SIZE, interpolation=InterpolationMode.BICUBIC),
        #transforms.CenterCrop(INPUT_SIZE),
        transforms.Lambda(lambda x: x.clamp(0, 1))
    ]

    transform_norm = transforms.Compose(transform_list + [
        # transforms.ToTensor(),
        # transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    
    B, S, C, H, W = rgb_batch.shape

    reshaped_batch = rgb_batch.view(B * S, C, H, W)
    normalized_batch = transform_norm(reshaped_batch)

    return normalized_batch.view(B, S, C, INPUT_SIZE, INPUT_SIZE)


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
        transforms.Lambda(lambda x: x.clamp(min=0, max = 70))
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
    depths = 1. / (depths + 1e-8)
    depths = batch_wise_min_max(depths)
    return depths

def prepare_tensors_for_visualization(
    rgb: torch.Tensor,
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare tensors for visualization by extracting depth channels and moving to CPU.
    
    Args:
        rgb_seq: RGB sequence tensor
        input_seq: Input sequence tensor
        output_seq: Output sequence tensor  
        gt_seq: Ground truth sequence tensor
        
    Returns:
        Tuple of prepared tensors (rgb, input, output, gt)
    """
    rgb_cpu = rgb.squeeze().detach().cpu()
    pred_cpu = pred_depth.squeeze().detach().cpu()
    gt_cpu = gt_depth.squeeze().detach().cpu()
    
    # For RGB, permute channels to be last for displaying
    # Shape changes from (S, C, H, W) to (S, H, W, C)
    rgb_display = rgb_cpu.permute(0, 2, 3, 1)

    return rgb_display, pred_cpu, gt_cpu

import matplotlib.pyplot as plt

# TODO: to utils lib
def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

import torch
from torch.utils.data import ConcatDataset

import yaml
from typing import Dict, Any

try:
    from data.SintelDataset import SintelDataset
    from data.VKittiDataset import VKittiDataset
    from data.TartanAirDataset import TartanAirDataset
    from data.PointOdysseyDataset import PointOdysseyDataset

    DATASET_REGISTRY = {
        "SintelDataset": SintelDataset,
        "VKittiDataset": VKittiDataset,
        "TartanAirDataset": TartanAirDataset,
        "PointOdysseyDataset": PointOdysseyDataset,
    }
except ImportError as e:
    print("="*80)
    print("ERROR: Could not import one of the dataset classes.")
    print("Please ensure that SintelDataset.py, VKittiDataset.py, TartanAirDataset.py,")
    print("PointOdysseyDataset.py, and ScanNetPPDataset.py exist within the 'src/data/' directory.")
    print(f"Details: {e}")
    print("="*80)
    raise

def create_datasets(dataset_names, config_file_path, split='train', random_seed=42):
    """
    Creates and configures a single dataset (either training or validation)
    based on a configuration dictionary and a specified split.
    """
    datasets = []
    config = load_config(config_file_path)
    common_params = config['dataset_common']
    
    valid_dataset_keys = {'sequence_length', 'output_size', 'use_random_crop', 'sequence_stride'}
    filtered_common_params = {k: v for k, v in common_params.items() if k in valid_dataset_keys}

    for dataset_name in dataset_names:
        dataset_config = config['datasets'][dataset_name]
        print(f"INFO: Preparing dataset for split '{split}': {dataset_name}")
        print(f"DEBUG: Full config for '{dataset_name}': {dataset_config}")

        class_name = dataset_name + "Dataset"
        if class_name not in DATASET_REGISTRY:
            print(f"ERROR: Dataset class '{class_name}' is not registered in the factory. Skipping.")
            continue
        DatasetClass = DATASET_REGISTRY[class_name]

        params = dataset_config['params']

        for key in ['envs', 'difficulties', 'cameras', 'data_types', 'exclude_scenes']:
            if key in dataset_config:
                params[key] = dataset_config[key]

        init_params = {**filtered_common_params, **params}

        try:
            full_dataset = DatasetClass(**init_params)

            train_split_perc = dataset_config['train_split']
            dataset_size = len(full_dataset)
            train_size = int(train_split_perc * dataset_size)
            val_size = dataset_size - train_size
            
            if (split == 'train' and train_size == 0) or (split == 'val' and val_size == 0):
                print(f"WARNING: Dataset '{dataset_name}' is too small to be split. Skipping.")
                continue
            
            datasets.append(full_dataset)

        except Exception as e:
            print(f"ERROR: Failed to instantiate or split {class_name}.")
            print(f"       This is often due to an incorrect path in config.yaml or a mismatch in the dataset's directory structure.")
            print(f"       Please verify the parameters below.")
            print(f"       Parameters passed to {class_name}: {init_params}")
            print(f"       Details: {e}")
            continue

    if not datasets:
        raise ValueError(f"No enabled and valid datasets were created for the '{split}' split. Please check your configuration.")
        
    print(f"Total samples for '{split}' split: {sum(len(dataset) for dataset in  datasets)}")
    return datasets



def visualize(args):
    """
    Main training function.
    """
    datasets = create_datasets(["PointOdyssey"], config_file_path="./configs/train_dataset_config.yaml", split='train', random_seed=RANDOM_SEED)

    dataset = datasets[0]

    print("Starting...")
    skip_scene = []
    contained_scene = []
    for i, batch in enumerate(dataset):
        if batch['rgb_paths'][0].split("/")[-3] in skip_scene:
            continue
        if batch['rgb_paths'][0].split("/")[-3] in contained_scene:
            continue
        
        valid = batch['depth'] > 0
        vmin = batch['depth'][valid].min()
        vmax = batch['depth'][valid].max()
        
        videos = batch['rgb'].unsqueeze(0)
        depths = batch['depth'].unsqueeze(0)
        masks = batch['mask'].unsqueeze(0).squeeze(2)
        
        def masked_min_max_normalize(depths: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
            """
            마스크가 True인 영역만 기준으로 min-max 정규화하고 전체 depth를 clip(0,1)한다.
            """
            # float32로 강제 변환 (안정성)
            depths = depths.float()
            
            # 마스크된 유효한 값만 추출
            valid_depths = depths[masks]

            # 최소/최대값 계산
            min_val = valid_depths.min()
            max_val = valid_depths.max()

            # 정규화 (마스크 바깥도 포함한 전체에 적용)
            normalized = (depths - min_val) / (max_val - min_val + 1e-8)

            # clip to [0, 1]
            normalized = normalized.clamp(0.0, 1.0)

            return normalized
        
        norm_depths = masked_min_max_normalize(depths.squeeze(2), masks)
        #depths = preprocess_depth_sequences(depths * batch['mask'].unsqueeze(0))
        # videos = preprocess_rgb_sequences(videos)
        
        #masks = depths * masks
        
        curr_scene = batch['rgb_paths'][0].split("/")[-3]
        print(f"path: {batch['rgb_paths'][0]}")
        print(f"scene: {curr_scene}")
        
        import matplotlib.pyplot as plt
        non_masked_depths = depths * ~masks
        plt.imsave("mask.png", masks[0][0].detach().cpu().numpy(), vmin = 0 , vmax=1, cmap="grey")
        plt.imsave("rgb.png", videos[0][0].permute(1,2,0).detach().cpu().numpy(), cmap="grey")
        plt.imsave("depth.png", norm_depths[0][0].detach().cpu().numpy(), vmin = 0 , vmax= 1, cmap="grey")
        #plt.imsave("depth.png", depths[0][0].detach().cpu().numpy(), vmin = 0 , vmax= 1, cmap="grey")
        #plt.imsave("depth.png", masked_depths[0][0].detach().cpu().numpy(), vmin = 0 , vmax=1, cmap="grey")
        #plt.imsave("none_depth.png", non_masked_depths[0][0].detach().cpu().numpy(), vmin = 0 , vmax=1, cmap="grey")

        # skip_scene.append(scene)
        print(f"skip scene: {skip_scene}")
        print(f"contained scene: {contained_scene}")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything Training')

    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for training (e.g., cuda:0, cpu).')
    parser.add_argument('--freeze_encoder', type=bool, default=True, help='Freeze the encoder weights during training.')
    parser.add_argument('--freeze_depth_head', type=bool, default=False, help='Freeze the depth head weights during training.')
    parser.add_argument('--datasets', type=str, default='small', choices=['small', 'large'], help='Dataset to use for training.')

    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'], help='Model encoder type.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation interval (number of epochs).')
    parser.add_argument('--size', type=int, default=518, help='Input size for the model.')

    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    
    parser.add_argument('--checkpoint_dir', type=str, default='./trained_models', help='Directory to save checkpoints.')
    parser.add_argument('--save_interval', type=int, default=1, help='Save checkpoint every N epochs.')

    parser.add_argument('--use_wandb', type=bool, default=False, help='Use Weights & Biases for logging.')
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device

    visualize(args)
