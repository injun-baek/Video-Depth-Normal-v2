import os
import cv2
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class TartanAirDataset(Dataset):
    """
    TartanAir Dataset for video depth estimation.

    This dataset loads RGB, depth, and depth_anything_v2 sequences
    from the TartanAir dataset. It supports sequence loading, random cropping,
    and returns file paths along with the data.
    """
    def __init__(self, 
                 root_dir, 
                 envs=['AbandonedSchool'],
                 difficulties=['easy', 'hard'],
                 cameras=['lcam_front'],
                 data_types=['rgb', 'depth'],
                 sequence_length=8, 
                 sequence_stride=1,
                 output_size=(640, 640), 
                 use_random_crop=False):
        """
        Initializes the TartanAirDataset.

        Args:
            root_dir (str): Root directory of the TartanAir dataset.
            envs (list): List of environments to load.
            difficulties (list): List of difficulties to load (e.g., 'easy', 'hard').
            cameras (list): List of camera names to use (e.g., 'lcam_front').
            data_types (list): List of data types to load (e.g., 'rgb', 'depth', 'depth_anything_v2').
            sequence_length (int): Number of frames in each sequence.
            sequence_stride (int): Stride for sampling sequences.
            output_size (tuple): Desired output size (height, width) for frames.
            use_random_crop (bool): Whether to apply random cropping.
        """
        super().__init__()
        
        if not os.path.isdir(root_dir): raise FileNotFoundError(f"Root directory not found: {root_dir}")
        self.root_dir, self.envs, self.difficulties, self.cameras, self.data_types = root_dir, envs, difficulties, cameras, data_types
        self.sequence_length, self.sequence_stride, self.output_size = sequence_length, sequence_stride, output_size
        self.use_random_crop = use_random_crop
        
        self.folder_map = {
            'rgb': 'image',
            'depth': 'depth',
            'depth_anything_v2': 'depth_anything_v2',
            'mask': 'depth',
        }

        if not isinstance(sequence_length, int) or sequence_length < 1: raise ValueError("sequence_length must be a positive integer.")
        if not isinstance(sequence_stride, int) or sequence_stride <= 0: raise ValueError("sequence_stride must be a positive integer.")

        self.data_pointers = self._create_data_pointers()
        
        if not self.data_pointers: raise RuntimeError("No valid data found.")
        print(f"TartanAirDataset initialized. Found {len(self.data_pointers)} sequences.")

    def _create_data_pointers(self) -> list:
        """
        Creates a list of dictionaries, where each dictionary contains
        paths to frames/sequences for each data type.

        Returns:
            list: A list of data pointers.
        """
        pointers = []
        for env in self.envs:
            for difficulty in self.difficulties:
                difficulty_path = os.path.join(self.root_dir, env, f"Data_{difficulty}")
                if not os.path.isdir(difficulty_path): continue
                
                for traj_id in sorted([d for d in os.listdir(difficulty_path) if os.path.isdir(os.path.join(difficulty_path, d))]):
                    for camera in self.cameras:
                        type_files = {}
                        min_frames = float('inf')
                        all_types_valid = True

                        for data_type in self.data_types:
                            folder_prefix = self.folder_map.get(data_type, data_type)
                            data_path = os.path.join(difficulty_path, traj_id, f"{folder_prefix}_{camera}")

                            if not os.path.isdir(data_path): all_types_valid = False; break
                            
                            files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.png')])
                            if not files: all_types_valid = False; break
                            
                            type_files[data_type] = files
                            min_frames = min(min_frames, len(files))

                        if not all_types_valid: continue

                        file_counts = {dt: len(files) for dt, files in type_files.items()}
                        if len(set(file_counts.values())) > 1:
                            raise ValueError(f"Inconsistent file counts for env '{env}', difficulty '{difficulty}', traj '{traj_id}', camera '{camera}' "
                                           f"across data types: {file_counts}")

                        if min_frames >= self.sequence_length:
                            for i in range(0, min_frames - self.sequence_length + 1, self.sequence_stride):
                                pointers.append({dt: files[i:i+self.sequence_length] for dt, files in type_files.items()})
        return pointers

    def __len__(self) -> int:
        """
        Returns the total number of samples (sequences or frames) in the dataset.
        """
        return len(self.data_pointers)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing loaded tensors and their corresponding file paths.
                  Keys for data are 'rgb', 'depth', etc.
                  Keys for paths are 'rgb_path' (for single frame) or 'rgb_paths' (for sequence).
        """
        path_dict = self.data_pointers[idx]
        
        crop_params = None
        pre_crop_resize_size = None

        if self.use_random_crop:
            ref_path = path_dict['rgb'][0]
            first_image = Image.open(ref_path).convert('RGB')
            
            w, h = first_image.size
            target_h, target_w = self.output_size

            if h < target_h or w < target_w:
                scale = max(target_h / h, target_w / w)
                new_h = math.ceil(h * scale)
                new_w = math.ceil(w * scale)
                pre_crop_resize_size = (new_h, new_w)  # (height, width)
                first_image = F.resize(first_image, pre_crop_resize_size, interpolation=Image.Resampling.BILINEAR)

            crop_params = transforms.RandomCrop.get_params(first_image, self.output_size)
        
        data = {}
        for dt, fps in path_dict.items():
            data[dt] = torch.stack([self._load_and_transform_frame(fp, dt, crop_params, pre_crop_resize_size) for fp in fps], dim=0)
            data[f'{dt}_paths'] = fps
        return data

    def _read_decode_tartanair_depth(self, depthpath: str) -> np.ndarray:
        """
        Reads and decodes TartanAir depth images.

        Args:
            depthpath (str): Path to the depth image file.

        Returns:
            np.ndarray: Decoded depth data as a 2D numpy array.
        """
        depth_rgba = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
        if depth_rgba is None:
            raise IOError(f"Could not load depth image: {depthpath}")
        
        # TartanAir depth images are stored as RGBA, where depth is encoded in float32
        depth = depth_rgba.view("<f4")
        return np.squeeze(depth, axis=-1)

    def _load_and_transform_frame(self, frame_path: str, data_type: str, crop_params: Optional[tuple], resize_size: Optional[tuple]) -> torch.Tensor:
        """
        Loads and transforms a single frame based on its data type.

        Args:
            frame_path (str): Path to the frame file.
            data_type (str): Type of data ('rgb', 'depth', 'depth_anything_v2').
            crop_params (tuple): Parameters for cropping (top, left, height, width).
            resize_size (tuple, optional): The size (h, w) to resize to before cropping. Can be None.

        Returns:
            torch.Tensor: Transformed frame tensor.
        """
        # Validate crop_params early if needed
        if self.use_random_crop and crop_params and len(crop_params) != 4:
            raise ValueError(f"crop_params must be a tuple of 4 elements, got {len(crop_params)}")
        
        if data_type == 'rgb':
            img = Image.open(frame_path).convert('RGB')
            img_tensor = F.to_tensor(img)
    
        elif data_type == 'depth':
            depth_np = self._read_decode_tartanair_depth(frame_path)
            img_tensor = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0)
    
        elif data_type == 'depth_anything_v2':
            img_np = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            if img_np.dtype != np.float32:
                img_np = img_np.astype(np.float32)
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            
        elif data_type == 'mask':
            depth_np = self._read_decode_tartanair_depth(frame_path)
            depth_np = np.ones_like(depth_np, dtype=bool)  # this is code smell
            img_tensor = torch.from_numpy(depth_np).unsqueeze(0)
        
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        current_size = [img_tensor.shape[-2], img_tensor.shape[-1]]
        
        if resize_size and resize_size != current_size:
            img_tensor = F.resize(img_tensor, resize_size, interpolation=Image.Resampling.BILINEAR)
        
        if self.use_random_crop and crop_params:
            img_tensor = F.crop(img_tensor, *crop_params)
        
        current_size = [img_tensor.shape[-2], img_tensor.shape[-1]]
        if current_size != list(self.output_size):
            img_tensor = F.resize(img_tensor, self.output_size, interpolation=Image.Resampling.BILINEAR)
        
        return img_tensor

