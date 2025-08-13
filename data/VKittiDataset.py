import os
import cv2
import math
import torch
import numpy as np
from typing import Optional
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import glob

class VKittiDataset(Dataset):
    """
    VKitti Dataset for video depth estimation.

    This dataset loads RGB, depth, and depth_anything_v2 sequences
    from the VKitti dataset. It supports sequence loading, random cropping,
    and returns file paths along with the data.
    """
    def __init__(self, 
                 root_dir, 
                 cameras=['Camera_0'],
                 data_types=['rgb', 'depth'],
                 sequence_length=8, 
                 sequence_stride=1,
                 output_size=(256, 512), 
                 use_random_crop=False):
        """
        Initializes the VKittiDataset.

        Args:
            root_dir (str): Root directory of the VKitti dataset.
            cameras (list): List of camera names to use (e.g., 'Camera_0').
            data_types (list): List of data types to load (e.g., 'rgb', 'depth', 'depth_anything_v2').
            sequence_length (int): Number of frames in each sequence.
            sequence_stride (int): Stride for sampling sequences.
            output_size (tuple): Desired output size (height, width) for frames.
            use_random_crop (bool): Whether to apply random cropping.
        """
        super().__init__()
        
        if not os.path.isdir(root_dir): raise FileNotFoundError(f"Root directory not found: {root_dir}")
        self.root_dir, self.cameras, self.data_types = root_dir, cameras, data_types
        self.sequence_length, self.sequence_stride, self.output_size = sequence_length, sequence_stride, output_size
        self.use_random_crop = use_random_crop
        
        self.folder_map = {
            'rgb': 'rgb',
            'depth': 'depth',
            'depth_anything_v2': 'depth_anything_v2',
            'mask': 'depth',
        }
        self.crop_reference_type = 'rgb'

        if not isinstance(sequence_length, int) or sequence_length < 1: raise ValueError("sequence_length must be a positive integer.")
        if not isinstance(sequence_stride, int) or sequence_stride <= 0: raise ValueError("sequence_stride must be a positive integer.")

        self.data_pointers = self._create_data_pointers()
        
        if not self.data_pointers: raise RuntimeError("No valid data found.")
        print(f"VKittiDataset initialized. Found {len(self.data_pointers)} sequences.")

    def _create_data_pointers(self) -> list:
        """
        Creates a list of dictionaries, where each dictionary contains
        paths to frames/sequences for each data type.

        Returns:
            list: A list of data pointers.
        """
        pointers = []
        scenes = sorted([d for d in os.listdir(self.root_dir) if d.startswith('Scene') and os.path.isdir(os.path.join(self.root_dir, d))])

        for scene in scenes:
            scene_path = os.path.join(self.root_dir, scene)
            variations = sorted([d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d))])
            
            for variation in variations:
                for camera in self.cameras:
                    base_path = os.path.join(scene_path, variation, 'frames')
                    type_files = {}
                    min_frames = float('inf')
                    all_types_valid = True

                    for data_type in self.data_types:
                        folder_name = self.folder_map.get(data_type, data_type)
                        data_path = os.path.join(base_path, folder_name, camera)

                        if not os.path.isdir(data_path): all_types_valid = False; break
                        
                        files = sorted([f for f in os.listdir(data_path) if f.endswith(('.png', '.jpg'))])
                        if not files: all_types_valid = False; break
                        
                        type_files[data_type] = [os.path.join(data_path, f) for f in files]
                        min_frames = min(min_frames, len(files))

                    if not all_types_valid: continue

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
            ref_path = path_dict[self.crop_reference_type][0]
            first_image = Image.open(ref_path).convert('RGB')
            w, h = first_image.size
            target_h, target_w = self.output_size

            if h < target_h or w < target_w:
                scale = max(target_h / h, target_w / w)
                new_h = math.ceil(h * scale)
                new_w = math.ceil(w * scale)
                pre_crop_resize_size = (new_h, new_w)
                first_image = F.resize(first_image, pre_crop_resize_size, interpolation=Image.Resampling.BILINEAR)

            crop_params = transforms.RandomCrop.get_params(first_image, self.output_size)

        data = {}
        for dt, fps in path_dict.items():
            data[dt] = torch.stack([self._load_and_transform_frame(fp, dt, crop_params, pre_crop_resize_size) for fp in fps], dim=0)
            data[f'{dt}_paths'] = fps
        return data

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
        if data_type == 'rgb':
            img = Image.open(frame_path).convert('RGB')
            img_tensor = F.to_tensor(img)
            current_size = [img_tensor.shape[-2], img_tensor.shape[-1]]
            if resize_size and resize_size != current_size:
                img_tensor = F.resize(img_tensor, resize_size, interpolation=Image.Resampling.BILINEAR)
            if self.use_random_crop and crop_params:
                img_tensor = F.crop(img_tensor, *crop_params)
            current_size = [img_tensor.shape[-2], img_tensor.shape[-1]]
            if current_size != list(self.output_size):
                img_tensor = F.resize(img_tensor, self.output_size, interpolation=Image.Resampling.BILINEAR)
            return img_tensor

        elif data_type in ['depth', 'depth_anything_v2']:
            img_np = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            img_np = img_np.astype(np.float32)
        elif data_type == 'mask':
            img_np = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            img_np = np.ones_like(img_np, dtype=bool)
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        current_size = [img_tensor.shape[-2], img_tensor.shape[-1]]
        if resize_size and resize_size != current_size:
            img_tensor = F.resize(img_tensor, resize_size, interpolation=Image.Resampling.BILINEAR)
        if self.use_random_crop and crop_params:
            img_tensor = F.crop(img_tensor, *crop_params)
        current_size = [img_tensor.shape[-2], img_tensor.shape[-1]]
        if current_size != list(self.output_size):
            img_tensor = F.resize(img_tensor, self.output_size, interpolation=Image.Resampling.BILINEAR)
        
        return img_tensor

