import os
import cv2
import torch
import math
import numpy as np
from typing import Optional, Tuple, List
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import glob
import logging

# Set up a logger for dataset-related messages
logger = logging.getLogger(__name__)

class NYUv2Dataset(Dataset):
    """
    NYUv2 Dataset for video depth estimation.

    This dataset loads RGB, depth, and other specified data types from the
    NYUv2 dataset structure. It supports loading sequences of frames,
    applying consistent random or center cropping, and resizing.
    """
    def __init__(self,
                 root_dir: str,
                 data_types: List[str] = ['rgb', 'depth', 'depth_anything_v2', 'mask'],
                 sequence_length: int = 1,
                 sequence_stride: int = 1,
                 output_size: Tuple[int, int] = (480, 640),
                 use_random_crop: bool = False):
        """
        Initializes the NYUv2Dataset.

        Args:
            root_dir (str): Root directory of the NYUv2 dataset (e.g., '.../nyu-v2/dataset').
            data_types (list): List of data types to load (e.g., 'rgb', 'depth', 'mask').
            sequence_length (int): Number of frames in each sequence. Defaults to 1 for single-image loading.
            sequence_stride (int): Stride for sampling sequences.
            output_size (tuple): Desired output size (height, width) for frames.
            use_random_crop (bool): Whether to apply random cropping. If False, center cropping is used.
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
        
        self.white_edge_size = 8

        # Map data types to their corresponding folder names within the root directory
        self.folder_map = {
            'rgb': 'rgb',
            'depth': 'depth',
            'depth_anything_v2': 'depth_anything_v2',
            'mask': 'depth',  # Mask is derived from depth, so it uses the same folder
        }

        self.paths = {dt: os.path.join(self.root_dir, self.folder_map[dt]) for dt in self.data_types}

        if not self.data_types:
            raise ValueError("The 'data_types' list cannot be empty.")

        # 'rgb' is required as it serves as the reference for transformations like cropping
        if 'rgb' not in self.data_types:
            raise ValueError("'rgb' must be included in data_types as it's required for the cropping reference.")
        self.primary_data_type = 'rgb' # Use RGB as the primary reference
        self.crop_reference_type = 'rgb'

        if not isinstance(sequence_length, int) or sequence_length < 1:
            raise ValueError("sequence_length must be a positive integer.")
        if not isinstance(sequence_stride, int) or sequence_stride <= 0:
            raise ValueError("sequence_stride must be a positive integer.")

        # Create pointers to all valid sequences in the dataset
        self.samples = self._create_data_pointers()

        if not self.samples:
            raise RuntimeError("No valid data found. Check the dataset structure and parameters.")

        print(f"NYUv2Dataset initialized. Found {len(self.samples)} samples/sequences.")

    def _create_data_pointers(self) -> list:
        """
        Creates a list of dictionaries, where each dictionary contains
        paths to frames for a valid sequence for each requested data type.
        This method is robust to missing files by checking for corresponding
        files based on the primary data type's filenames.

        Returns:
            list: A list of sample pointers, where each pointer is a dictionary
                  mapping data_type to a list of file paths for a sequence.
        """
        pointers = []
        primary_path = self.paths[self.primary_data_type]
        scenes = sorted([d for d in os.listdir(primary_path) if os.path.isdir(os.path.join(primary_path, d))])

        # Define file extensions for each data type
        file_extension_map = {
            'rgb': '.png',
            'depth_anything_v2': '.png',
            'depth': '.npy',
            'mask': '.npy', # Mask is derived from depth .npy files
        }

        for scene in scenes:
            scene_samples = []
            # Find all files for the primary data type (e.g., all RGB images)
            primary_files = sorted(glob.glob(os.path.join(primary_path, scene, f'*{file_extension_map[self.primary_data_type]}')))

            for primary_file_path in primary_files:
                # Get the base filename without extension (e.g., 'basement_0001_0489')
                base_name = os.path.splitext(os.path.basename(primary_file_path))[0]
                
                sample_paths = {self.primary_data_type: primary_file_path}
                all_files_exist = True

                # Check for the existence of corresponding files for other data types
                for dt in self.data_types:
                    if dt == self.primary_data_type:
                        continue
                    
                    ext = file_extension_map.get(dt)
                    if ext is None:
                        logger.warning(f"No file extension defined for data type '{dt}'. Skipping.")
                        all_files_exist = False
                        break

                    # Construct the expected path for the corresponding file
                    expected_path = os.path.join(self.paths[dt], scene, base_name + ext)

                    if os.path.exists(expected_path):
                        sample_paths[dt] = expected_path
                    else:
                        all_files_exist = False
                        # This warning is helpful for debugging dataset issues
                        # logger.warning(f"Corresponding file not found for {primary_file_path}: Missing {expected_path}")
                        break
                
                if all_files_exist:
                    scene_samples.append(sample_paths)

            # Create sequences from the list of valid individual samples in the scene
            if len(scene_samples) >= self.sequence_length:
                for i in range(0, len(scene_samples) - self.sequence_length + 1, self.sequence_stride):
                    sequence_group = scene_samples[i : i + self.sequence_length]
                    
                    # Reformat to the desired structure: {data_type: [path1, path2, ...]}
                    sequence_pointer = {dt: [] for dt in self.data_types}
                    for sample in sequence_group:
                        for dt, path in sample.items():
                            sequence_pointer[dt].append(path)
                    pointers.append(sequence_pointer)
            elif len(scene_samples) > 0: # this is for evaluation!
                sequence_group = scene_samples
                sequence_pointer = {dt: [] for dt in self.data_types}
                for sample in sequence_group:
                    for dt, path in sample.items():
                        sequence_pointer[dt].append(path)
                pointers.append(sequence_pointer)
        return pointers


    def __len__(self) -> int:
        """Returns the total number of samples/sequences in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a sample (or sequence) from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing loaded tensors and their corresponding file paths.
                  Keys for data are like 'rgb', 'depth'.
                  Keys for paths are like 'rgb_paths', 'depth_paths'.
        """
        path_dict = self.samples[idx]

        # Determine crop and resize parameters based on the first RGB frame
        # to ensure all frames in the sequence are transformed consistently.
        crop_params = None
        pre_resize_size = None

        ref_path = path_dict[self.crop_reference_type][0]
        first_image = Image.open(ref_path).convert('RGB')
        w, h = first_image.size
        target_h, target_w = self.output_size

        # If the original image is smaller than the target output, we need to upscale it first
        if h < target_h or w < target_w:
            if self.white_edge_size > 0:
                h, w = first_image.size
                top, left = self.white_edge_size, self.white_edge_size
                height, width = h - 2*self.white_edge_size, w - 2*self.white_edge_size
                first_image = F.crop(first_image, top, left, height, width)
                h, w = height, width
                        
            scale = max(target_h / h, target_w / w)
            new_h = math.ceil(h * scale)
            new_w = math.ceil(w * scale)
            pre_resize_size = (new_h, new_w)
            first_image = F.resize(first_image, pre_resize_size, interpolation=Image.Resampling.BILINEAR)

        if self.use_random_crop:
            crop_params = transforms.RandomCrop.get_params(first_image, self.output_size)
        
        # Load and transform all frames in the sequence
        data = {}
        for dt, fps in path_dict.items():
            # Stack frames along a new dimension to create a sequence tensor
            data[dt] = torch.stack([self._load_and_transform_frame(fp, dt, crop_params, pre_resize_size) for fp in fps], dim=0)
            data[f'{dt}_paths'] = fps
        return data

    def _load_and_transform_frame(self, frame_path: str, data_type: str, crop_params: Optional[tuple], resize_size: Optional[tuple]) -> torch.Tensor:
        """
        Loads and transforms a single frame based on its data type.

        Args:
            frame_path (str): Path to the frame file.
            data_type (str): The type of data to load ('rgb', 'depth', 'mask').
            crop_params (tuple, optional): Parameters for cropping (top, left, height, width).
            resize_size (tuple, optional): The size (h, w) to resize to before cropping.

        Returns:
            torch.Tensor: The transformed frame as a tensor.
        """
        # --- Load data based on type ---
        if data_type == 'rgb':
            img = Image.open(frame_path).convert('RGB')
            img_tensor = F.to_tensor(img) # Converts to [C, H, W] and scales to [0, 1]
        elif data_type == 'depth_anything_v2':
            img = Image.open(frame_path)
            img_np = np.array(img, dtype=np.uint16)
            img_np = img_np.astype(np.float32)
            img_tensor = torch.from_numpy(img_np).unsqueeze(0) # Add channel dim: [1, H, W]
        elif data_type == 'depth':
            # Load depth data from .npy file
            depth_np = np.load(frame_path)
            # Convert to float tensor without scaling
            depth_np = depth_np.astype(np.float32)
            img_tensor = torch.from_numpy(depth_np).unsqueeze(0) # Add channel dim: [1, H, W]
            img_tensor = img_tensor.transpose(-2, -1)
        elif data_type == 'mask':
            # Load depth data from .npy file to create a mask
            depth_np = np.load(frame_path)
            # Valid pixels are where depth > 0
            mask_np = (depth_np > 0)
            img_tensor = torch.from_numpy(mask_np).unsqueeze(0) # Add channel dim: [1, H, W]
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        if self.white_edge_size > 0:
            h, w = img_tensor.shape[-2], img_tensor.shape[-1]
            top, left = self.white_edge_size, self.white_edge_size
            height, width = h - 2*self.white_edge_size, w - 2*self.white_edge_size
            img_tensor = F.crop(img_tensor, top, left, height, width)

        # --- Apply transformations ---
        # Apply pre-resize if necessary (upscaling)
        current_size = [img_tensor.shape[-2], img_tensor.shape[-1]]
        if resize_size and tuple(resize_size) != tuple(current_size):
            # Use NEAREST for depth/mask to avoid interpolating values
            interpolation_mode = Image.Resampling.NEAREST if data_type != 'rgb' else Image.Resampling.BILINEAR
            img_tensor = F.resize(img_tensor, resize_size, interpolation=interpolation_mode)

        # Apply crop
        if self.use_random_crop and crop_params:
            img_tensor = F.crop(img_tensor, *crop_params)
        else:
            img_tensor = F.center_crop(img_tensor, self.output_size)

        # Final resize to ensure output size is exact (can happen due to rounding)
        current_size = [img_tensor.shape[-2], img_tensor.shape[-1]]
        if tuple(current_size) != tuple(self.output_size):
            interpolation_mode = Image.Resampling.NEAREST if data_type != 'rgb' else Image.Resampling.BILINEAR
            img_tensor = F.resize(img_tensor, self.output_size, interpolation=interpolation_mode)

        return img_tensor
