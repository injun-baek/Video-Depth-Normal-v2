import torch
from torch.utils.data import random_split, ConcatDataset, WeightedRandomSampler, DataLoader

import yaml
from typing import Dict, Any

try:
    from data.DA2KDataset import DA2KDataset
    from data.SintelDataset import SintelDataset
    from data.VKittiDataset import VKittiDataset
    from data.TartanAirDataset import TartanAirDataset
    from data.PointOdysseyDataset import PointOdysseyDataset
    # from data.ScanNetPPDataset import ScanNetPPDataset

    # Create a registry to map class names from the config to the actual classes
    DATASET_REGISTRY = {
        "SintelDataset": SintelDataset,
        "VKittiDataset": VKittiDataset,
        "TartanAirDataset": TartanAirDataset,
        "PointOdysseyDataset": PointOdysseyDataset,
        "DA2KDataset": DA2KDataset,
        # "ScanNetPPDataset": ScanNetPPDataset,
    }
except ImportError as e:
    print("="*80)
    print("ERROR: Could not import one of the dataset classes.")
    print("Please ensure that SintelDataset.py, VKittiDataset.py, TartanAirDataset.py,")
    print("PointOdysseyDataset.py, and ScanNetPPDataset.py exist within the 'src/data/' directory.")
    print(f"Details: {e}")
    print("="*80)
    raise

# TODO: to utils lib
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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

        for key in ['envs', 'difficulties', 'cameras', 'data_types', 'exclude_scenes', 'sequence_length', 'sequence_stride']:
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
            
            generator = torch.Generator().manual_seed(random_seed)
            train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
            
            if split == 'train':
                datasets.append(train_subset)
            else: # split == 'val'
                datasets.append(val_subset)

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

def create_sampler(train_dataset):
    """
    Creates a WeightedRandomSampler to handle imbalanced datasets.
    """
    
    if not isinstance(train_dataset, ConcatDataset):
        print("WARNING: WeightedRandomSampler is only supported for ConcatDataset. Skipping.")
        return None

    print("INFO: Creating WeightedRandomSampler for handling imbalanced datasets.")
    
    weights = []
    for dataset in train_dataset.datasets:
        dataset_size = len(dataset)
        if dataset_size > 0:
            weight_per_sample = 1.0 / dataset_size
            weights.extend([weight_per_sample] * dataset_size)

    if not weights:
        print("WARNING: Could not compute weights for sampler. Disabling it.")
        return None
        
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler

def create_data_loader(dataset, batch_size, num_workers=1, sampler=None, shuffle=False):
    """Create datasets and data loaders"""
    print("Creating datasets...")

    if sampler is not None:
        shuffle = False
        print("Using sampler for data loading. set shuffle=False")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Dataset: {len(dataset)} samples")

    return loader
