import os
import torch
import math
import numpy as np
from typing import Optional, List, Dict, Tuple
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import logging
import bisect

# Set up a logger for dataset-related messages
logger = logging.getLogger(__name__)

class BonnDataset(Dataset):
    """
    Bonn Dataset for video depth estimation, with timestamp-based frame matching.

    This dataset scans a root directory for scene subdirectories. Each scene
    is expected to contain 'rgb.txt' and 'depth.txt' files. It matches each
    RGB frame to the depth frame with the closest timestamp.

    'depth_anything_v2' paths are derived from RGB paths.
    'mask' tensors are generated dynamically.
    """
    def __init__(self,
                 root_dir: str,
                 data_types: List[str] = ['rgb', 'depth'],
                 sequence_length: int = 8,
                 sequence_stride: int = 1,
                 output_size: tuple = (436, 1024),
                 use_random_crop: bool = False,
                 exclude_scenes: list = None):
        """
        Initializes the BonnDataset.

        Args:
            root_dir (str): Root directory containing scene subdirectories.
            data_types (list): List of data types to load (e.g., 'rgb', 'depth', 'mask').
            sequence_length (int): Number of frames in each sequence.
            sequence_stride (int): Stride for sampling sequences.
            output_size (tuple): Desired output size (height, width) for frames.
            use_random_crop (bool): Whether to apply random cropping.
            exclude_scenes (list, optional): List of scene names to exclude from loading. Defaults to None.
        """
        super().__init__()

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        self.root_dir = root_dir
        self.data_types = data_types
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.output_size = output_size
        self.use_random_crop = use_random_crop
        self.exclude_scenes = exclude_scenes if exclude_scenes is not None else []

        if not self.data_types:
            raise ValueError("data_types list cannot be empty.")

        if 'rgb' not in self.data_types:
            raise ValueError("'rgb' must be included in data_types for reference.")
        self.crop_reference_type = 'rgb'

        if not isinstance(sequence_length, int) or sequence_length < 1:
            raise ValueError("sequence_length must be a positive integer.")
        if not isinstance(sequence_stride, int) or sequence_stride <= 0:
            raise ValueError("sequence_stride must be a positive integer.")

        self.samples = self._create_data_pointers()

        if not self.samples:
            raise RuntimeError("No valid data sequences found. Check directory structure and txt files.")

        print(f"BonnDataset initialized. Found {len(self.samples)} sequences.")

    def _parse_txt_with_timestamps(self, file_path: str) -> List[Tuple[float, str]]:
        """
        Parses Bonn-style txt file, extracting (timestamp, path) tuples.
        
        Returns:
            A list of tuples, where each tuple is (timestamp, relative_path).
        """
        entries = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    try:
                        timestamp = float(parts[0])
                        relative_path = parts[1]
                        entries.append((timestamp, relative_path))
                    except ValueError:
                        logger.warning(f"Could not parse timestamp from line in {file_path}: {line}")
        return entries

    def _create_data_pointers(self) -> List[Dict[str, List[str]]]:
        """
        Creates sequences by finding the closest depth frame for each RGB frame
        based on timestamps, then grouping them into sequences.
        """
        all_frames = []
        scene_dirs = sorted([os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir)
                             if os.path.isdir(os.path.join(self.root_dir, d))])

        for scene_dir in scene_dirs:
            scene = scene_dir.split(os.path.sep)[-1]
            if scene in self.exclude_scenes:
                logger.info(f"Skipping excluded scene '{scene}' as specified in config.")
                continue
            rgb_txt_path = os.path.join(scene_dir, 'rgb.txt')
            depth_txt_path = os.path.join(scene_dir, 'depth.txt')

            if not (os.path.isfile(rgb_txt_path) and os.path.isfile(depth_txt_path)):
                logger.warning(f"Skipping scene '{os.path.basename(scene_dir)}' as it's missing rgb.txt or depth.txt.")
                continue

            rgb_entries = self._parse_txt_with_timestamps(rgb_txt_path)
            depth_entries = self._parse_txt_with_timestamps(depth_txt_path)

            if not rgb_entries or not depth_entries:
                logger.warning(f"No valid entries in txt files for scene '{os.path.basename(scene_dir)}'. Skipping.")
                continue

            # Sort depth entries by timestamp for efficient search
            depth_entries.sort(key=lambda x: x[0])
            depth_timestamps = [entry[0] for entry in depth_entries]

            # For each RGB frame, find the best matching depth frame
            for rgb_ts, rgb_rel_path in rgb_entries:
                
                frame_paths = {}
                rgb_abs_path = os.path.join(scene_dir, rgb_rel_path)
                frame_paths['rgb'] = rgb_abs_path

                if 'depth' in self.data_types:
                    # Find the insertion point of the rgb timestamp in the sorted depth timestamps list
                    idx = bisect.bisect_left(depth_timestamps, rgb_ts)

                    # Identify the two best candidates for the closest timestamp
                    best_match_depth_rel_path = None
                    min_diff = float('inf')

                    # Candidate 1: The one at the insertion point (or the last one if out of bounds)
                    if idx < len(depth_timestamps):
                        diff = abs(depth_timestamps[idx] - rgb_ts)
                        if diff < min_diff:
                            min_diff = diff
                            best_match_depth_rel_path = depth_entries[idx][1]

                    # Candidate 2: The one just before the insertion point
                    if idx > 0:
                        diff = abs(depth_timestamps[idx - 1] - rgb_ts)
                        if diff < min_diff:
                            min_diff = diff
                            best_match_depth_rel_path = depth_entries[idx - 1][1]
                    
                    if best_match_depth_rel_path:
                        frame_paths['depth'] = os.path.join(scene_dir, best_match_depth_rel_path)
                    else:
                        # This case should ideally not be reached if depth_entries is not empty.
                        # Skip this rgb frame if no suitable depth match is found.
                        print("No suitable depth match found for RGB frame:", rgb_rel_path)
                        continue

                if 'depth_anything_v2' in self.data_types:
                    da_rel_path = rgb_rel_path.replace('rgb/', 'depth_anything_v2/')
                    frame_paths['depth_anything_v2'] = os.path.join(scene_dir, da_rel_path)

                if 'mask' in self.data_types:
                    frame_paths['mask'] = frame_paths['depth']

                all_frames.append(frame_paths)

        # Create sequences from the flat list of matched frames
        pointers = []
        if len(all_frames) >= self.sequence_length:
            for i in range(0, len(all_frames) - self.sequence_length + 1, self.sequence_stride):
                sequence_slice = all_frames[i : i + self.sequence_length]

                sequence_pointer = {dt: [] for dt in self.data_types}
                for frame_data in sequence_slice:
                    for dt in self.data_types:
                        if dt in frame_data:
                            sequence_pointer[dt].append(frame_data[dt])
                
                pointers.append(sequence_pointer)

        return pointers

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        path_dict = self.samples[idx]
        crop_params = None
        pre_resize_size = None

        ref_path = path_dict[self.crop_reference_type][0]
        first_image = Image.open(ref_path).convert('RGB')
        w, h = first_image.size
        target_h, target_w = self.output_size

        if h < target_h or w < target_w:
            scale = max(target_h / h, target_w / w)
            new_h = math.ceil(h * scale)
            new_w = math.ceil(w * scale)
            pre_resize_size = (new_h, new_w)
            first_image = F.resize(first_image, pre_resize_size, interpolation=Image.Resampling.BILINEAR)

        if self.use_random_crop:
            crop_params = transforms.RandomCrop.get_params(first_image, self.output_size)

        data = {}
        for dt, fps in path_dict.items():
            data[dt] = torch.stack([self._load_and_transform_frame(fp, dt, crop_params, pre_resize_size) for fp in fps], dim=0)
            if dt != 'mask':
                data[f'{dt}_paths'] = fps

        return data

    def _load_and_transform_frame(self, frame_path: str, data_type: str, crop_params: Optional[tuple], resize_size: Optional[tuple]) -> torch.Tensor:
        if data_type == 'rgb':
            img = Image.open(frame_path).convert('RGB')
            img_tensor = F.to_tensor(img)
        elif data_type in ['depth', 'depth_anything_v2']:
            img = Image.open(frame_path)
            img_np = np.array(img, dtype=np.float32)
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            if data_type == 'depth':
                img_tensor = img_tensor / 1000.0  # Convert mm to meters if needed
        elif data_type == 'mask':
            depth_img = Image.open(frame_path)
            depth_np = np.array(depth_img, dtype=np.float32)
            mask_np = (depth_np != 0)
            img_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        current_size = [img_tensor.shape[-2], img_tensor.shape[-1]]
        if resize_size and resize_size != tuple(current_size):
            img_tensor = F.resize(img_tensor, resize_size, interpolation=Image.Resampling.BILINEAR)

        if self.use_random_crop and crop_params:
            img_tensor = F.crop(img_tensor, *crop_params)
        else:
            img_tensor = F.center_crop(img_tensor, list(self.output_size))

        current_size = [img_tensor.shape[-2], img_tensor.shape[-1]]
        if tuple(current_size) != self.output_size:
            img_tensor = F.resize(img_tensor, list(self.output_size), interpolation=Image.Resampling.BILINEAR)
        
        if data_type == 'mask':
            return img_tensor.bool()

        return img_tensor
