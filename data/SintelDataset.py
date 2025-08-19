import os
import cv2
import torch
import math
import numpy as np
from typing import Optional
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import glob
import logging

logger = logging.getLogger(__name__)

class SintelDataset(Dataset):
    """
    Sintel Dataset for video depth estimation.

    This dataset loads RGB, depth, and depth_anything_v2 sequences
    from the Sintel dataset. It supports sequence loading, random cropping,
    and returns file paths along with the data.
    """
    def __init__(self,
                 root_dir,
                 data_types=['rgb', 'depth'],
                 sequence_length=8, 
                 sequence_stride=1,
                 output_size=(436, 1024), 
                 use_random_crop=False):
        """
        Initializes the SintelDataset.

        Args:
            root_dir (str): Root directory of the Sintel dataset.
            data_types (list): List of data types to load (e.g., 'rgb', 'depth', 'depth_anything_v2').
            sequence_length (int): Number of frames in each sequence.
            sequence_stride (int): Stride for sampling sequences.
            output_size (tuple): Desired output size (height, width) for frames.
            use_random_crop (bool): Whether to apply random cropping.
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
        
        self.folder_map = {
            'rgb': 'rgb',
            'depth': 'depth',
            'depth_anything_v2': 'depth_anything_v2',
            'mask': 'depth',
        }
        
        self.paths = {dt: os.path.join(self.root_dir, self.folder_map[dt]) for dt in self.data_types}

        if not self.data_types:
            raise ValueError("data_types list cannot be empty.")
        self.primary_data_type = self.data_types[0]
        
        # Always use 'rgb' as crop reference type since RGB is always loaded
        self.crop_reference_type = 'rgb'
        
        if 'rgb' not in self.data_types:
            raise ValueError("'rgb' must be included in data_types as it's required for cropping reference.")

        if not isinstance(sequence_length, int) or sequence_length < 1:
            raise ValueError("sequence_length must be a positive integer.")
        if not isinstance(sequence_stride, int) or sequence_stride <= 0:
            raise ValueError("sequence_stride must be a positive integer.")

        self.samples = self._create_data_pointers()
        
        if not self.samples:
            raise RuntimeError("No valid data found. Check dataset structure and parameters.")

        print(f"SintelDataset initialized. Found {len(self.samples)} sequences.")

    @staticmethod
    def read_dpt_file(filepath: str) -> np.ndarray:
        """
        Reads a .dpt file (Sintel depth format).

        Args:
            filepath (str): Path to the .dpt file.

        Returns:
            np.ndarray: Depth data as a 2D numpy array.
        """
        try:
            with open(filepath, 'rb') as f:
                tag = np.fromfile(f, dtype=np.float32, count=1)
                if tag != 202021.25:
                    raise IOError(f"Invalid dpt file tag in {filepath}")
                width = np.fromfile(f, dtype=np.int32, count=1)[0]
                height = np.fromfile(f, dtype=np.int32, count=1)[0]
                data = np.fromfile(f, dtype=np.float32, count=-1)
                if data.size != width * height:
                    raise IOError(f"Dpt file size mismatch in {filepath}")
                return data.reshape(height, width)
        except Exception as e:
            logger.error(f"Failed to read DPT file {filepath}: {e}")
            raise

    def _create_data_pointers(self) -> list:
        """
        Creates a list of dictionaries, where each dictionary contains
        paths to frames/sequences for each data type.

        Returns:
            list: A list of data pointers.
        """
        pointers = []
        primary_path = self.paths[self.primary_data_type]
        scenes = sorted([d for d in os.listdir(primary_path) if os.path.isdir(os.path.join(primary_path, d))])

        for scene in scenes:
            type_files = {}
            min_frames = float('inf')
            all_types_valid = True

            for data_type in self.data_types:
                data_path = os.path.join(self.paths[data_type], scene)
                if not os.path.isdir(data_path):
                    all_types_valid = False
                    break
                
                if data_type == 'depth':
                    files = sorted(glob.glob(os.path.join(data_path, '*.dpt')))
                elif data_type == 'mask':
                    files = sorted(glob.glob(os.path.join(data_path, '*.dpt')))
                elif data_type == 'depth_anything_v2':
                    files = sorted(glob.glob(os.path.join(data_path, '*.npy')))
                else:
                    files = sorted(glob.glob(os.path.join(data_path, '*.png')))
                
                if not files:
                    all_types_valid = False
                    break
                
                type_files[data_type] = files
                min_frames = min(min_frames, len(files))

            if not all_types_valid:
                continue

            file_counts = {dt: len(files) for dt, files in type_files.items()}
            if len(set(file_counts.values())) > 1:
                raise ValueError(f"Inconsistent file counts for scene '{scene}' across data types: {file_counts}")

            if min_frames >= self.sequence_length:
                for i in range(0, min_frames - self.sequence_length + 1, self.sequence_stride):
                    pointers.append({dt: files[i:i+self.sequence_length] for dt, files in type_files.items()})
            # else:
            #     pointers.append({dt: files[:min_frames] for dt, files in type_files.items()})
        return pointers

    def __len__(self) -> int:
        return len(self.samples)

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
        path_dict = self.samples[idx]
        
        crop_params = None
        pre_resize_size = None

        ref_path = path_dict[self.crop_reference_type][0]
        if os.path.splitext(ref_path)[1] != '.npy':
            first_image = Image.open(ref_path).convert('RGB')
            w, h = first_image.size
            target_h, target_w = self.output_size

            if h < target_h or w < target_w:
                scale = max(target_h / h, target_w / w)
                new_h = math.ceil(h * scale)
                new_w = math.ceil(w * scale)
                pre_resize_size = (new_h, new_w) 

        if pre_resize_size is not None:
            first_image = F.resize(first_image, pre_resize_size, interpolation=Image.Resampling.BILINEAR)
        
        if self.use_random_crop:
            crop_params = transforms.RandomCrop.get_params(first_image, self.output_size)

        data = {}
        for dt, fps in path_dict.items():
            data[dt] = torch.stack([self._load_and_transform_frame(fp, dt, crop_params, pre_resize_size) for fp in fps], dim=0)
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
            else:
                img_tensor = F.center_crop(img_tensor, self.output_size)
            current_size = [img_tensor.shape[-2], img_tensor.shape[-1]]
            if current_size != self.output_size:
                img_tensor = F.resize(img_tensor, self.output_size, interpolation=Image.Resampling.BILINEAR)
            return img_tensor

        elif data_type == 'depth':
            img_np = self.read_dpt_file(frame_path)
            img_np = img_np.astype(np.float32)
        elif data_type == 'depth_anything_v2':
            img_np = np.load(frame_path)
            img_np = img_np.astype(np.float32)
        elif data_type == 'mask':
            img_np = self.read_dpt_file(frame_path)
            img_np = np.ones_like(img_np, dtype=bool)  
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        
        current_size = [img_tensor.shape[-2], img_tensor.shape[-1]]
        if resize_size and resize_size != current_size:
            img_tensor = F.resize(img_tensor, resize_size, interpolation=Image.Resampling.BILINEAR)
        if self.use_random_crop and crop_params:
            img_tensor = F.crop(img_tensor, *crop_params)
        else:
            img_tensor = F.center_crop(img_tensor, self.output_size)
        current_size = [img_tensor.shape[-2], img_tensor.shape[-1]]
        if current_size != self.output_size:
            img_tensor = F.resize(img_tensor, self.output_size, interpolation=Image.Resampling.BILINEAR)
        return img_tensor
