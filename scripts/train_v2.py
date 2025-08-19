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

# This allows us to import modules from sibling directories like 'data', 'loss', etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from video_depth_anything.video_depth import VideoDepthAnything
from models.video_depth_model_v2 import VideoDepthAnything as TrainingModel
from models.video_depth_model import VideoDepthEstimationModel
from loss.loss import VideoDepthLoss, VideoNormalLoss, compute_scale_and_shift
from utils.normal_utils import normal_vector
from utils.wandb_manager import WandbManager

from torch.utils.data import ConcatDataset
from data.dataset_factory import (
    create_sampler,
    create_data_loader,
    create_datasets,
)

RANDOM_SEED = 42

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
        # x: (B, S, H, W), masks: (B, S, H, W) [bool]
        if masks == None:
            B = x.shape[0]
            x_flat = x.view(B, -1)
            x_min = x_flat.min(dim=1, keepdim=True)[0]
            x_max = x_flat.max(dim=1, keepdim=True)[0]
            x_scaled = (x_flat - x_min) / torch.clamp(x_max - x_min, min=1e-8)
            return x_scaled.view_as(x)
        else:
            B = x.shape[0]
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
    # depths = 1. / torch.clamp(depths, min=1e-8)

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

    scale, shift = compute_scale_and_shift(pred_depth.unsqueeze(0).flatten(1, 2), gt_depth.unsqueeze(0).flatten(1, 2), torch.ones_like(gt_depth).unsqueeze(0).flatten(1, 2))
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
        da_scale, da_shift = compute_scale_and_shift(depth_anything_depth.unsqueeze(0).flatten(1, 2), gt_depth.unsqueeze(0).flatten(1, 2), torch.ones_like(gt_depth).unsqueeze(0).flatten(1, 2))
        depth_anything_depth = da_scale.view(1, 1, 1, 1) * depth_anything_depth + da_shift.view(1, 1, 1, 1)
        depth_anything_depth = depth_anything_depth.squeeze()
        # affine_error_da = torch.abs(depth_anything_depth - gt_depth)
        row_titles.append('Depth Anything')
        data_rows.append(depth_anything_depth)
        cmaps.append(depth_visual_cmap)
        subplot_rows += 1
        figsize_height += 2

    if video_depth_anything_depth is not None:
        vda_scale, vda_shift = compute_scale_and_shift(video_depth_anything_depth.unsqueeze(0).flatten(1, 2), gt_depth.unsqueeze(0).flatten(1, 2), torch.ones_like(gt_depth).unsqueeze(0).flatten(1, 2))
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

@torch.no_grad()
def create_sample_visualizations(model, dataset, device, sample_indices, video_depth_anything=None, save_dir='./visualizations', save_path_format="sample.png"):
    """Create sample visualizations during training"""

    os.makedirs(save_dir, exist_ok=True)
    
    unwrapped_model = model.module if hasattr(model, 'module') else model
    unwrapped_model.eval()

    visualization_paths = []
    pbar = tqdm(sample_indices, desc="Creating Visualizations")
    for i, sample_idx in enumerate(pbar):
        if sample_idx >= len(dataset):
            print(f"Warning: Sample index {sample_idx} exceeds dataset size {len(dataset)}")
            continue

        batch = dataset[sample_idx]
        rgb = batch["rgb"].to(device).unsqueeze(0)
        rgb_input = preprocess_rgb_sequences(rgb)
        
        gt_depth = batch["depth"].to(device).unsqueeze(0)
        da_depth = batch["depth_anything_v2"].to(device).unsqueeze(0)
    
        rgb_viz = preprocess_rgb_viz_sequences(rgb)
        mask = batch["mask"].to(device).unsqueeze(0)
        input_depth = preprocess_depth_sequences(da_depth, mask, False)

        gt_depth = 1. / torch.clamp(gt_depth, min=1e-8)
        gt_depth = preprocess_depth_sequences(gt_depth, mask, True)
        # gt_normals = normal_vector(gt_depth)
        gt_depth = gt_depth.squeeze(2)

        #pred_depth, pred_normal = model(input_depth, rgb_input)
        pred_depth = model(input_depth)

        pred_depth = (pred_depth - pred_depth.min()) / torch.clamp(pred_depth.max() - pred_depth.min(), min=1e-8)
        pred_depth = pred_depth.clamp(0.0, 1.0)
        
        da_depth = (da_depth - da_depth.min()) / torch.clamp(da_depth.max() - da_depth.min(), min=1e-8)
        da_depth = da_depth.clamp(0.0, 1.0)
        da_depth = da_depth.squeeze().cpu()

        rgb_viz, pred_vis, gt_vis = prepare_tensors_for_visualization(rgb_viz, pred_depth, gt_depth)
        
        save_cache_path = os.path.join(save_dir, save_path_format.format(sample_idx=sample_idx))
        create_sample_visualization(rgb_viz, pred_vis, gt_vis, save_cache_path, sample_idx, da_depth, None)
        visualization_paths.append(save_cache_path)
        pbar.set_postfix({"Sample": sample_idx, "Path": save_cache_path})
        torch.cuda.empty_cache()

    return visualization_paths

@torch.no_grad()
def validate(model, val_loader, depth_criterion, normal_criterion, device):
    """
    Perform validation on the model.
    """
    model.eval()
    running_losses = defaultdict(float)
    pbar = tqdm(val_loader, desc="Validating")
    for i, batch in enumerate(pbar):
        rgbs = batch['rgb'].to(device)
        rgbs = preprocess_rgb_sequences(rgbs)
        
        input_depths = batch['depth_anything_v2'].to(device)
        gt_depths = batch['depth'].to(device)
        masks = batch['mask'].to(device)

        input_depths = preprocess_depth_sequences(input_depths, masks, False)
        
        gt_depths = 1. / torch.clamp(gt_depths, min=1e-8)
        #gt_depths = preprocess_depth_sequences(gt_depths, masks, True)
        gt_normals = normal_vector(gt_depths) # .unsqueeze(2)
        gt_depths = gt_depths.squeeze(2)
        
        #pred_depths, pred_normals = model(input_depths, rgbs)
        pred_depths = model(input_depths)
        
        depth_loss_dict = depth_criterion(pred_depths, gt_depths, masks.squeeze(2))
        # normal_loss_dict = normal_criterion(pred_normals, gt_normals, masks.squeeze(2))
        #for key, value in (depth_loss_dict | normal_loss_dict).items():
        for key, value in depth_loss_dict.items():
            if torch.is_tensor(value):
                running_losses[key] += value.item()
        
        interested_keys = ['total_loss', 'absRel_loss', 'd1']
        batch_loss_display = {"avg_" + k: f'{running_losses[k] / (i + 1):.4f}' for k in interested_keys}
        pbar.set_postfix(batch_loss_display)
        
        torch.cuda.empty_cache()
            
    avg_losses = {key: value / len(val_loader) for key, value in running_losses.items()}
    
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
        logger = WandbManager(project_name="VideoDepthNormal-v2", entity=None)
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

    # vda_model = VideoDepthAnything(**model_configs[args.encoder])
    # vda_model.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    # vda_model = vda_model.to(device)

    model = TrainingModel(
        **model_configs[args.encoder],
        use_residual=args.use_residual,
        input_normal=args.input_normal
    )
    if args.from_my_trained:
        model.load_state_dict(torch.load(args.from_my_trained, map_location='cpu'), strict=False)
    else:        
        model.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=False)
    model = model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.pretrained.parameters():
        param.requires_grad = True
    for param in model.final_res.parameters():
        param.requires_grad = True

    # if parallel:
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #     model = torch.nn.DataParallel(model, device_ids=cuda_nums)
    
    # todo
    #if args.pretrained:
    #    model.load_state_dict(torch.load(args.pretrained, map_location=device))

    using_datasets = []
    if args.datasets.lower() == 'small':
        using_datasets = ["Sintel"]
        train_viz_index = [0]#, 4, 5, 8, 10, 14, 15, 25, 29, 32, 50, 52, 64, 69, 72, 74, 77, 82, 85, 88, 113]
        val_viz_index = [0]#, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
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
    depth_criterion = VideoDepthLoss(alpha=args.alpha, stable_scale=args.stable_scale, ssim_loss_scale=args.ssim_loss_scale).to(device)
    normal_criterion = VideoNormalLoss().to(device)

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_running_loss_dict = defaultdict(float)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        
        for i, batch in enumerate(pbar):
            rgbs = batch['rgb'].to(device)
            rgbs = preprocess_rgb_sequences(rgbs)
            
            input_depths = batch['depth_anything_v2'].to(device)
            gt_depths = batch['depth'].to(device)
            masks = batch['mask'].to(device)

            input_depths = preprocess_depth_sequences(input_depths, masks, False)
            
            gt_depths = 1. / torch.clamp(gt_depths, min=1e-8)
            #gt_depths = preprocess_depth_sequences(gt_depths, masks, True)
            gt_normals = normal_vector(gt_depths)  #.unsqueeze(2)
            gt_depths = gt_depths.squeeze(2)

            optimizer.zero_grad()
            #pred_depths, pred_normals = model(input_depths, rgbs)
            pred_depths = model(input_depths)
            depth_loss_dict = depth_criterion(pred_depths, gt_depths, masks.squeeze(2))
            total_loss = depth_loss_dict['total_loss']
            #normal_loss_dict = normal_criterion(pred_normals, gt_normals, masks.squeeze(2))
            #total_loss = depth_loss_dict['total_loss'] + normal_loss_dict['normal_loss'] * args.normal_loss_scale

            total_loss.backward()
            
            #for key, value in (depth_loss_dict | normal_loss_dict).items():
            for key, value in depth_loss_dict.items():
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
        train_vis_paths = create_sample_visualizations(model, train_dataset, device, train_viz_index, None, save_path_format = f"{exp_num}" + "_train_sample_{sample_idx:03d}.png")
        val_vis_paths = create_sample_visualizations(model, val_dataset, device, val_viz_index, None, save_path_format = f"{exp_num}" + "_val_sample_{sample_idx:03d}.png")
        
        if logger:
            metrics_to_log = {f"training/{key}": value for key, value in train_avg_losses.items()}
            metrics_to_log["epoch"] = epoch + 1
            logger.add_metrics(metrics_to_log)

            if (epoch + 1) % args.val_interval == 0:
                val_avg_losses = validate(model, val_loader, depth_criterion, normal_criterion, device)
                metrics_to_log = {f"validation/{key}": value for key, value in val_avg_losses.items()}
                metrics_to_log["epoch"] = epoch + 1
                logger.add_metrics(metrics_to_log)

            logger.add_images({
                "training/visualizations": train_vis_paths,
                "validation/visualizations": val_vis_paths
            })
            logger.send_log(step=epoch + 1)

        if (epoch + 1) % args.save_interval == 0:
            save_model_checkpoint(model, args.checkpoint_dir, f'{exp_num}_video_depth_normal_epoch_{epoch+1}.pth')

        torch.cuda.empty_cache()

    save_model_checkpoint(model, args.checkpoint_dir, f'{exp_num}_video_depth_normal_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything Training')

    parser.add_argument('--device', type=str, default='cuda:2', help='Device to use for training (e.g., cuda:0, cpu).')
    parser.add_argument('--encoder', type=str, default='vits', help='Encoder architecture to use.')
    parser.add_argument('--freeze_encoder', type=bool, default=False, help='Freeze the encoder weights during training.')
    
    parser.add_argument('--use_residual', type=bool, default=True, help='Use residual connections.')
    parser.add_argument('--input_normal', type=bool, default=True, help='Use input normal.')
    parser.add_argument('--from_my_trained', type=str, default=None, help='from my trained model')
    
    parser.add_argument('--datasets', type=str, default='large', choices=['small', 'middle', 'middle2', 'large'], help='Dataset to use for training.')
    parser.add_argument('--sequence_length', type=int, default=16, help='Length of the input video sequence.')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training.')
    parser.add_argument('--size', type=int, default=224, help='Input size for the model.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')

    parser.add_argument('--initial_lr', type=float, default=1e-5, help='Initial learning rate.')
    parser.add_argument('--final_lr', type=float, default=1e-8, help='Final learning rate.')
    parser.add_argument('--alpha', type=float, default=0.0, help='Alpha value for the gradient loss function.')
    parser.add_argument('--stable_scale', type=float, default=0.0, help='Scale for the temporal loss function.')
    parser.add_argument('--ssim_loss_scale', type=float, default=0.0, help='Scale for the SSIM loss function.')
    parser.add_argument('--normal_loss_scale', type=float, default=0.0, help='Scale for the normal loss function.')

    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation interval (number of epochs).')
    
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
