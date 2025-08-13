import os
os.environ["NCCL_P2P_LEVEL"] = "LOC"

import argparse
import sys
from typing import Tuple
import torch
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Add the project root directory to the Python path
# This allows us to import modules from sibling directories like 'data', 'loss', etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from video_depth_anything.video_depth import VideoDepthAnything
from depth_anything_v2.depth_anything_v2 import DepthAnythingV2
from loss.loss import VideoDepthLoss, compute_scale_and_shift
from utils.wandb_manager import WandbManager

from torch.utils.data import ConcatDataset
from data.dataset_factory import (
    create_sampler,
    create_data_loader,
    create_datasets,
)

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


def preprocess_depth_sequences(depth_batch, masks, norm=True):
    def batch_wise_min_max_norm(x, masks):
        if masks == None:
            B = x.shape[0]
            x_flat = x.view(B, -1)
            x_min = x_flat.min(dim=1, keepdim=True)[0]
            x_max = x_flat.max(dim=1, keepdim=True)[0]
            x_scaled = (x_flat - x_min) / torch.clamp(x_max - x_min, min=1e-8)
            return x_scaled.view_as(x)
        else:
            shape = x.shape
            B = shape[0]
            x_flat = x.view(B, -1)
            min_val = torch.where(masks, x, torch.full_like(x, float('inf'))).view(B, -1).min(dim=1, keepdim=True)[0]
            max_val = torch.where(masks, x, torch.full_like(x, float('-inf'))).view(B, -1).max(dim=1, keepdim=True)[0]

            min_val = min_val.view(B, 1, 1, 1)
            max_val = max_val.view(B, 1, 1, 1)

            normalized = (x - min_val) / torch.clamp(max_val - min_val, min=1e-8)
            normalized = normalized.clamp(0.0, 1.0)
            
            is_valid_mask = masks.any(dim=(-1, -2, -3)).view(B, 1, 1, 1)
            normalized = torch.where(is_valid_mask, normalized, torch.zeros_like(normalized))

            return normalized
    
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

    if norm:
        masks = masks.squeeze(2)
        depths = batch_wise_min_max_norm(depths, masks)
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

def create_sample_visualization(
    rgb_image: torch.Tensor,
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    save_path: str,
    sample_idx: int = 0,
    depth_anything_depth: torch.Tensor = None,
    video_depth_anything_depth: torch.Tensor = None
) -> None:
    """
    Create and save visualization comparing input, output (affine aligned), gt, and affine invariant error.
    
    Args:
        rgb_image: RGB image sequence (S, H, W, C)
        output_depth: Output depth sequence (S, H, W)
        gt_depth: Ground truth depth sequence (S, H, W)
        save_path: Path to save the visualization
        sample_idx: Sample index for naming
        depth_anything_depth: Optional depth from DepthAnything for comparison
        video_depth_anything_depth: Optional depth from VideoDepthAnything for comparison
    """

    mask = compute_temporal_mask(gt_depth, 0.05)
    scale, shift = compute_scale_and_shift(pred_depth.unsqueeze(0).flatten(1, 2), gt_depth.unsqueeze(0).flatten(1, 2), mask.unsqueeze(0).flatten(1, 2))
    pred_depth = scale.view(1, 1, 1, 1) * pred_depth + shift.view(1, 1, 1, 1)
    affine_error = torch.abs(pred_depth - gt_depth)
    
    pred_depth, gt_depth, affine_error = pred_depth.squeeze(), gt_depth.squeeze(), affine_error.squeeze()

    S = rgb_image.shape[0]    
    subplot_rows = 4
    figsize_height = 8
    
    depth_visual_cmap = "grey"
    row_titles = ['RGB Image', 'Output (Aligned)', 'Ground Truth', 'Affine Invariant Error']
    data_rows = [rgb_image, pred_depth, gt_depth, affine_error]
    cmaps = [None, depth_visual_cmap, depth_visual_cmap, 'hot']

    if depth_anything_depth is not None:
        da_scale, da_shift = compute_scale_and_shift(depth_anything_depth.unsqueeze(0).flatten(1, 2), gt_depth.unsqueeze(0).flatten(1, 2), mask.unsqueeze(0).flatten(1, 2))
        depth_anything_depth = da_scale.view(1, 1, 1, 1) * depth_anything_depth + da_shift.view(1, 1, 1, 1)
        depth_anything_depth = depth_anything_depth.squeeze()
        # affine_error_da = torch.abs(depth_anything_depth - gt_depth)
        row_titles.append('Depth Anything')
        data_rows.append(depth_anything_depth)
        cmaps.append(depth_visual_cmap)
        subplot_rows += 1
        figsize_height += 2

    if video_depth_anything_depth is not None:
        vda_scale, vda_shift = compute_scale_and_shift(video_depth_anything_depth.unsqueeze(0).flatten(1, 2), gt_depth.unsqueeze(0).flatten(1, 2), mask.unsqueeze(0).flatten(1, 2))
        video_depth_anything_depth = vda_scale.view(1, 1, 1, 1) * video_depth_anything_depth + vda_shift.view(1, 1, 1, 1)
        video_depth_anything_depth = video_depth_anything_depth.squeeze()
        # affine_error_vda = torch.abs(video_depth_anything_depth - gt_depth)
        row_titles.append('Video Depth Anything')
        data_rows.append(video_depth_anything_depth)
        cmaps.append(depth_visual_cmap)
        subplot_rows += 1
        figsize_height += 2

    fig, axes = plt.subplots(subplot_rows, S, figsize=(2 * S, figsize_height))

    for row_idx, (title, data, cmap) in enumerate(zip(row_titles, data_rows, cmaps)):
        for frame_idx in range(S):
            ax = axes[row_idx][frame_idx]
            
            if cmap:
                im = ax.imshow(data[frame_idx], cmap=cmap, vmin=0, vmax=1)
            else: # For RGB image
                im = ax.imshow(data[frame_idx])

            if row_idx == 0:
                ax.set_title(f'Frame {frame_idx}', fontsize=10)

            if frame_idx == 0:
                 ax.set_ylabel(title, fontsize=10, rotation=90, labelpad=15)
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    fig.suptitle(f'Sample {sample_idx} Visualization', fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")

@torch.no_grad()
def create_sample_visualizations(model, dataset, device, sample_indices, video_depth_anything=None, save_dir='./visualizations', save_path_format="sample.png"):
    """Create sample visualizations during training"""

    os.makedirs(save_dir, exist_ok=True)
    
    unwrapped_model = model.module if hasattr(model, 'module') else model
    unwrapped_model.eval()

    visualization_paths = []
    for i, sample_idx in enumerate(sample_indices):
        if sample_idx >= len(dataset):
            print(f"Warning: Sample index {sample_idx} exceeds dataset size {len(dataset)}")
            continue

        try:
            batch = dataset[sample_idx]
            rgb, gt_depth, da_depth = batch["rgb"], batch["depth"], batch["depth_anything_v2"]
            rgb, gt_depth, da_depth = rgb.to(device).unsqueeze(0), gt_depth.to(device).unsqueeze(0), da_depth.to(device).unsqueeze(0)

            vda_depth = None
            if video_depth_anything is not None:
                vda_input = batch['rgb']
                vda_input = vda_input.squeeze(0).permute(0, 2, 3, 1) * 255.0
                vda_input = vda_input.numpy()
                
                vda_model = video_depth_anything.module if hasattr(video_depth_anything, 'module') else video_depth_anything
                vda_depth = vda_model.infer_video_depth(vda_input, -1, input_size=INPUT_SIZE, device=device, fp32=True)
                vda_depth = vda_depth[0]
                vda_depth = torch.from_numpy(vda_depth)
                vda_depth = vda_depth / torch.clamp(vda_depth.max(), min=1e-8)

            rgb_input = preprocess_rgb_sequences(rgb)
            rgb_viz = preprocess_rgb_viz_sequences(rgb)
            mask = batch["mask"].to(device).unsqueeze(0)
            gt_depth = preprocess_depth_sequences(gt_depth, mask, True)
            
            da_depth = da_depth.squeeze().detach().cpu()
            da_depth = da_depth / torch.clamp(da_depth.max(), min=1e-8)

            predictions = []
            unwrapped_model.clear_memory()
            with torch.no_grad():
                S = rgb_input.shape[1]
                for t in range(S):
                    video_frame = rgb_input[:, t, :, :, :].to(device)
                    pred_depth = model(video_frame)
                    predictions.append(pred_depth)
            pred_depth = torch.stack(predictions, dim=1)

            rgb_viz, pred_vis, gt_vis = prepare_tensors_for_visualization(rgb_viz, pred_depth, gt_depth)
            
            save_cache_path = os.path.join(save_dir, save_path_format.format(sample_idx=sample_idx))
            create_sample_visualization(rgb_viz, pred_vis, gt_vis, save_cache_path, sample_idx, da_depth, vda_depth)
            visualization_paths.append(save_cache_path)
            
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")

    print(f"Visualization completed")
    return visualization_paths

@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """
    Perform validation on the model.
    """
    model.eval()
    running_losses = defaultdict(float)
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for i, batch in enumerate(pbar):
            videos = batch['rgb'].to(device)
            depths = batch['depth'].to(device)
            masks = batch['mask'].to(device)
            
            videos = preprocess_rgb_sequences(videos)
            depths = preprocess_depth_sequences(depths, masks, False)

            masks = masks.squeeze(2)
            # spatial_masks = compute_spatial_mask(depths, 0.01, 70)
            # temporal_masks = compute_temporal_mask(depths, 0.05)
            # combined_masks = masks.squeeze(2) & temporal_masks
            
            unwrapped_model = model.module if hasattr(model, 'module') else model
            unwrapped_model.clear_memory()
            predictions = []
            S = videos.shape[1]
            for t in range(S):
                video_frame = videos[:, t, :, :, :].to(device)
                pred_depth = model(video_frame)
                predictions.append(pred_depth)
            predictions = torch.stack(predictions, dim=1)
            
            loss_dict = criterion(predictions, depths, masks)
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    running_losses[key] += value.item()
            
            interested_keys = ['total_loss', 'absRel_loss', 'd1']
            batch_loss_display = {"avg_" + k: f'{running_losses[k] / (i + 1):.4f}' for k in interested_keys}
            pbar.set_postfix(batch_loss_display)
            
            torch.cuda.empty_cache()
            
    avg_losses = {key: value / len(val_loader) for key, value in running_losses.items()}
    loss_summary = ", ".join([f"Avg {key}: {value:.4f}" for key, value in avg_losses.items()])
    print(f"Validation finished. {loss_summary}")
    
    return avg_losses

def save_model_checkpoint(model, save_dir, checkpoint_name):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), checkpoint_path)
    else:
        torch.save(model.state_dict(), checkpoint_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")

def train(args):
    logger = None
    global exp_num
    exp_num = None
    if args.use_wandb:
        logger = WandbManager(project_name="DA MemoryAtt", entity=None)
        run_name = logger.get_next_run_name(base_name="Test")
        logger.init(run_name=run_name, config=vars(args))
        exp_num = logger.get_experiment_number()

    cuda_nums = [int(num) for num in args.device.split(':')[1].split(',')]
    parallel = len(cuda_nums) > 1
    torch.cuda.set_device(cuda_nums[0])
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    model = DepthAnythingV2(**model_configs[args.encoder])
    
    if True:  # todo: add pretrained option
        checkpoint = torch.load(f'./checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu')
        model_dict = model.state_dict()

        pretrained_weights = {}
        for k, v in checkpoint.items():
            if k.startswith("pretrained.") or k.startswith("depth_head."):  # DPT head
                if k in model_dict:
                    pretrained_weights[k] = v
                else:
                    print(f"Skipping weight '{k}' as it does not match model keys.")

        model_dict.update(pretrained_weights)
        model.load_state_dict(model_dict)
        
        print("Freezing the DinoVisionTransformer (pretrained) parameters.")
        if args.freeze_encoder:
            for param in model.pretrained.parameters():
                param.requires_grad = False
        if args.freeze_depth_head:
            for param in model.depth_head.parameters():
                param.requires_grad = False

    model.to(device)
    if parallel:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.DataParallel(model, device_ids=cuda_nums)
    
    video_depth_anything = None
    if True:  # this is for evaluation
        video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
        video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
        video_depth_anything = video_depth_anything.to(device).eval()
        if parallel:
            video_depth_anything = torch.nn.DataParallel(video_depth_anything, device_ids=cuda_nums)
    
    using_datasets = []
    if args.datasets.lower() == 'small':
        using_datasets = ["Sintel"]
        train_viz_index = [0, 4, 5, 8, 10, 14, 15, 25, 29, 32, 50, 52, 64, 69, 72, 74, 77, 82, 85, 88, 113]
        val_viz_index = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
    elif args.datasets.lower() == 'middle':
        using_datasets = ["PointOdyssey"]
        train_viz_index = [14, 30, 59, 142, 154, 198, 225, 305, 311, 328, 336, 430, 439, 470, 536, 542, 548, 568, 577, 607, 610, 613, 617, 628, 631, 635, 654, 661, 678, 742, 751, 760, 786, 894, 911, 925, 955, 993, 994] #, 1031, 1038, 1121, 1170, 1181, 1194, 1240, 1253, 1254, 1267, 1307, 1308, 1320, 1369, 1387, 1398]
        val_viz_index = [15, 96, 127, 131, 151, 208, 238, 253, 310, 395, 401, 409, 473, 506, 518, 520, 543, 564, 565, 569, 577, 626, 631, 646, 649, 668, 669, 702, 779, 800, 825, 835, 871, 873, 902, 903, 924, 929, 930, 979, 983] #, 1044, 1047, 1056, 1106, 1112, 1143, 1178, 1184, 1213, 1289, 1321, 1330, 1338, 1366]
    elif args.datasets.lower() == 'middle2':
        using_datasets = ["TartanAir"]
        train_viz_index = [0, 4, 5, 8, 10, 14, 15, 25, 29, 32, 50, 52, 64, 69, 72, 74, 77, 82, 85, 88, 113]
        val_viz_index = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
    elif args.datasets.lower() == 'large':
        using_datasets = ["VKitti", "TartanAir", "PointOdyssey"]
        train_viz_index = [32, 93, 101, 149, 169, 193, 241, 248, 286, 411, 548, 703, 780, 812, 933, 1087, 1100, 1189, 1202, 1225, 1258, 1365, 1370, 1373, 1480, 1504, 1508, 1510, 1554, 1567, 1593, 1634, 1647, 1659, 1710, 1778, 1796, 1827, 1839, 1859, 1915, 1973, 2002, 2045, 2174, 2238, 2373, 2408, 2514, 2516, 2566, 2594, 2630, 2649, 2721, 2754, 2837, 2889, 2905, 2911]
        val_viz_index = [42, 63, 513, 567, 717, 770, 840, 874, 899, 920, 947, 976, 1002, 1009, 1018, 1073, 1140, 1146, 1201, 1234, 1246, 1281, 1305, 1313, 1348, 1379, 1380, 1395, 1400, 1402, 1446, 1449, 1524, 1526, 1605, 1609, 1629, 1649, 1770, 1904, 1950, 2011, 2079, 2127, 2199, 2222, 2225, 2295, 2320, 2500, 2539, 2594, 2745, 2757, 2814, 2865, 2898, 2913, 2925, 2955]
    else:
        raise ValueError(f"Unknown dataset option: {args.datasets}. Choose 'small' or 'large'.")

    train_datasets = create_datasets(using_datasets, config_file_path="./configs/train_dataset_config.yaml", split='train', random_seed=RANDOM_SEED)
    val_datasets = create_datasets(using_datasets, config_file_path="./configs/train_dataset_config.yaml", split='val', random_seed=RANDOM_SEED)

    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

    # todo : this is code smell
    if args.datasets.lower() == 'middle2':
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset.dataset, train_dataset.indices[:int(len(train_dataset) / 20)])
        val_dataset = Subset(val_dataset.dataset, val_dataset.indices[:int(len(val_dataset) / 20)])

    sampler_enabled = True
    train_sampler = None
    if sampler_enabled:
        train_sampler = create_sampler(train_dataset)

    train_loader = create_data_loader(train_dataset, args.batch_size, num_workers=args.num_workers, sampler=train_sampler)
    val_loader = create_data_loader(val_dataset, args.batch_size, num_workers=args.num_workers)

    optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10_000, T_mult=2, eta_min=args.final_lr)
    criterion = VideoDepthLoss(alpha=args.alpha, stable_scale=args.stable_scale, ssim_loss_scale=args.ssim_loss_scale).to(device)

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_running_loss_dict = defaultdict(float)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        
        for i, batch in enumerate(pbar):
            videos = batch['rgb'].to(device)
            depths = batch['depth'].to(device)
            masks = batch['mask'].to(device)

            videos = preprocess_rgb_sequences(videos)
            depths = preprocess_depth_sequences(depths, masks, False)
            
            masks = masks.squeeze(2)
            # spatial_masks = compute_spatial_mask(depths, 0.01, 70)
            # temporal_masks = compute_temporal_mask(depths, 0.05)
            # combined_masks = masks.squeeze(2) & temporal_masks

            optimizer.zero_grad()
            unwrapped_model = model.module if hasattr(model, 'module') else model
            unwrapped_model.clear_memory()

            predictions = []
            S = videos.shape[1]
            for t in range(S):
                video_frame = videos[:, t, :, :, :].to(device)
                pred_depth = model(video_frame)
                predictions.append(pred_depth)
            predictions = torch.stack(predictions, dim=1)

            loss_dict = criterion(predictions, depths, masks)
            total_loss = loss_dict['total_loss']
            total_loss.backward()
            
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    train_running_loss_dict[key] += value.item()
            
            optimizer.step()
            scheduler.step()

            running_loss += total_loss.item()
            interested_keys = ['total_loss', 'absRel_loss', 'd1']
            batch_loss_display = {"avg_" + k: f'{train_running_loss_dict[k] / (i + 1):.4f}' for k in interested_keys}
            pbar.set_postfix(batch_loss_display)

        num_batches = len(train_loader)
        train_avg_losses = {key: value / num_batches for key, value in train_running_loss_dict.items()}
        print(f"Epoch {epoch + 1} finished. Average Training Loss: {train_avg_losses['total_loss']:.4f}")
        train_vis_paths = create_sample_visualizations(model, train_dataset, device, train_viz_index, video_depth_anything, save_path_format = f"{exp_num}" + "_train_sample_{sample_idx:03d}.png")
        val_vis_paths = create_sample_visualizations(model, val_dataset, device, val_viz_index, video_depth_anything, save_path_format = f"{exp_num}" + "_val_sample_{sample_idx:03d}.png")
        
        if logger:
            metrics_to_log = {f"training/{key}": value for key, value in train_avg_losses.items()}
            metrics_to_log["epoch"] = epoch + 1
            logger.add_metrics(metrics_to_log)

            if (epoch + 1) % args.val_interval == 0:
                val_avg_losses = validate(model, val_loader, criterion, device)
                metrics_to_log = {f"validation/{key}": value for key, value in val_avg_losses.items()}
                metrics_to_log["epoch"] = epoch + 1
                logger.add_metrics(metrics_to_log)

            logger.add_images({
                "training/visualizations": train_vis_paths,
                "validation/visualizations": val_vis_paths
            })
            logger.send_log(step=epoch + 1)

        if args.datasets != "small" and (epoch + 1) % args.save_interval == 0:
            save_model_checkpoint(model, args.checkpoint_dir, f'{exp_num}_memory_depth_anything_{args.encoder}_epoch_{epoch+1}.pth')
        elif args.datasets == "small" and (epoch + 1) % (args.save_interval * 10) == 0:
            save_model_checkpoint(model, args.checkpoint_dir, f'{exp_num}_memory_depth_anything_{args.encoder}_epoch_{epoch+1}.pth')

        torch.cuda.empty_cache()

    save_model_checkpoint(model, args.checkpoint_dir, f'{exp_num}_memory_depth_anything_{args.encoder}_epoch_{epoch+1}.pth')
    print("Finished Training")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything Training')

    parser.add_argument('--device', type=str, default='cuda:3', help='Device to use for training (e.g., cuda:0, cpu).')
    parser.add_argument('--freeze_encoder', type=bool, default=True, help='Freeze the encoder weights during training.')
    parser.add_argument('--freeze_depth_head', type=bool, default=True, help='Freeze the depth head weights during training.')
    parser.add_argument('--datasets', type=str, default='middle2', choices=['small', 'middle', 'middle2', 'large'], help='Dataset to use for training.')

    parser.add_argument('--initial_lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--final_lr', type=float, default=1e-7, help='Final learning rate.')
    parser.add_argument('--alpha', type=float, default=0.0, help='Alpha value for the gradient loss function.')
    parser.add_argument('--stable_scale', type=float, default=0.0, help='Scale for the temporal loss function.')
    parser.add_argument('--ssim_loss_scale', type=float, default=5.0, help='Scale for the SSIM loss function.')

    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'], help='Model encoder type.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation interval (number of epochs).')
    parser.add_argument('--size', type=int, default=518, help='Input size for the model.')

    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    
    parser.add_argument('--checkpoint_dir', type=str, default='./trained_models', help='Directory to save checkpoints.')
    parser.add_argument('--save_interval', type=int, default=1, help='Save checkpoint every N epochs.')

    parser.add_argument('--use_wandb', type=bool, default=True, help='Use Weights & Biases for logging.')
    args = parser.parse_args()

    # todo smell
    global DEVICE
    DEVICE = args.device
    global INPUT_SIZE
    INPUT_SIZE = args.size

    train(args)
