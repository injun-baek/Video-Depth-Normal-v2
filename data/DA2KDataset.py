import os
import torch
import math
import numpy as np
from typing import Optional, List, Dict, Tuple
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import glob
import logging

# Set up a logger for dataset-related messages
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DA2KDataset(Dataset):
    """
    DA-2K Dataset for single-image depth estimation.

    This dataset loads individual RGB images (.jpg) and their corresponding 
    depth maps (.png) from the DA-2K dataset structure. It does not handle
    sequences of frames. It supports resizing, random or center cropping,
    and returns file paths along with the tensor data.
    """
    def __init__(self,
                 root_dir: str,
                 data_types: List[str] = ['rgb', 'depth'],
                 sequence_length=1, 
                 sequence_stride=1,
                 output_size: Tuple[int, int] = (518, 518), 
                 use_random_crop: bool = False):
        """
        Initializes the DA2KDataset.

        Args:
            root_dir (str): Root directory of the DA-2K dataset, which should 
                            contain 'images' and 'depth' subdirectories.
            data_types (list): List of data types to load (e.g., 'rgb', 'depth', 'mask').
            output_size (tuple): Desired output size (height, width) for frames.
            use_random_crop (bool): Whether to apply random cropping. If False,
                                    center cropping is used.
        """
        if sequence_length != 1 or sequence_stride != 1:
            raise ValueError("DA2KDataset does not support sequences. Set sequence_length and sequence_stride to 1.")
        
        super().__init__()
        
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")
        self.root_dir = root_dir
        self.data_types = data_types
        self.output_size = output_size
        self.use_random_crop = use_random_crop
        
        self.folder_map = {
            'rgb': 'images',
            'depth': 'depth_anything_v2',
            'mask': 'depth_anything_v2',
            'depth_anything_v2': 'depth_anything_v2'
        }
        
        self.paths = {dt: os.path.join(self.root_dir, self.folder_map[dt]) for dt in self.data_types}

        if not self.data_types:
            raise ValueError("data_types list cannot be empty.")
        
        self.crop_reference_type = 'rgb'
        if self.crop_reference_type not in self.data_types:
             raise ValueError("A suitable image type ('rgb') was not found for crop reference.")

        self.samples = self._create_data_pointers()
        
        if not self.samples:
            raise RuntimeError("No valid data found. Check dataset structure and parameters.")

        logger.info(f"DA2KDataset initialized. Found {len(self.samples)} samples.")

    def _create_data_pointers(self) -> List[Dict[str, str]]:
        """
        Creates a list of dictionaries, where each dictionary contains paths
        to the files for each requested data type.

        Returns:
            list: A list of sample pointers. 
                  Example: [{'rgb': 'path/rgb.jpg', 'depth': 'path/depth.png', 'mask': 'path/depth.png'}, ...]
        """
        pointers = []
        rgb_base_path = self.paths['rgb']
        
        rgb_files = glob.glob(os.path.join(rgb_base_path, '**', '*.jpg'), recursive=True)
        
        depth_dependent_types = ['depth', 'mask', 'depth_anything_v2']
        is_depth_required = any(dt in depth_dependent_types for dt in self.data_types)
        
        for rgb_path in rgb_files:
            current_sample_paths = {}

            if is_depth_required:
                relative_path = os.path.relpath(rgb_path, rgb_base_path)
                depth_relative_path = os.path.splitext(relative_path)[0] + '.png'
                ref_path = os.path.join(self.paths['depth'], depth_relative_path)

                if not os.path.exists(ref_path):
                    raise Exception(f"Required reference file not found for {rgb_path}. Expected at {ref_path}. Skipping sample.")

                for dt in self.data_types:
                    if dt in depth_dependent_types:
                        current_sample_paths[dt] = ref_path
            
            if 'rgb' in self.data_types:
                current_sample_paths['rgb'] = rgb_path
            
            if len(current_sample_paths) != len(self.data_types):
                raise Exception("Not all required data types are present.")
            pointers.append(current_sample_paths)
        return pointers

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing loaded tensors and their corresponding file paths.
                  Keys for data are 'rgb', 'depth', 'mask', etc.
                  Keys for paths are 'rgb_path', 'depth_path', 'mask_path', etc.
        """
        path_dict = self.samples[idx]
        
        crop_params = None
        pre_resize_size = None

        # Determine resize and crop parameters based on the reference image
        ref_path = path_dict[self.crop_reference_type]
        try:
            first_image = Image.open(ref_path).convert('RGB')
        except FileNotFoundError:
            logger.error(f"Reference image not found at {ref_path}")
            # Return an empty dict or handle error appropriately
            return {}
            
        w, h = first_image.size
        target_h, target_w = self.output_size

        # If the image is smaller than the target size, scale it up before cropping
        if h < target_h or w < target_w:
            scale = max(target_h / h, target_w / w)
            new_h = math.ceil(h * scale)
            new_w = math.ceil(w * scale)
            pre_resize_size = (new_h, new_w) 

        if self.use_random_crop:
            # Apply pre-resizing if needed
            if pre_resize_size:
                first_image = F.resize(first_image, pre_resize_size, interpolation=Image.Resampling.BILINEAR)
            crop_params = transforms.RandomCrop.get_params(first_image, self.output_size)

        data = {}
        for dt, fp in path_dict.items():
            data[dt] = self._load_and_transform_frame(fp, dt, crop_params, pre_resize_size)
            data[f'{dt}_path'] = fp
            
        return data

    def _load_and_transform_frame(self, frame_path: str, data_type: str, 
                                  crop_params: Optional[Tuple[int, int, int, int]], 
                                  resize_size: Optional[Tuple[int, int]]) -> torch.Tensor:
        """
        Loads and transforms a single frame based on its data type.

        Args:
            frame_path (str): Path to the frame file.
            data_type (str): Type of data ('rgb', 'depth', 'mask').
            crop_params (tuple, optional): Parameters for cropping (top, left, height, width).
            resize_size (tuple, optional): The size (h, w) to resize to before cropping. Can be None.

        Returns:
            torch.Tensor: Transformed frame tensor.
        """
        try:
            if data_type == 'rgb':
                # Load RGB image
                img = Image.open(frame_path).convert('RGB')
                img_tensor = F.to_tensor(img)
            elif data_type in ['depth', 'depth_anything_v2']:
                # Load Depth image (as single-channel grayscale)
                img = Image.open(frame_path).convert('L')
                img_np = np.array(img).astype(np.float32)
                img_tensor = torch.from_numpy(img_np).unsqueeze(0) # Add channel dimension
            elif data_type == 'mask':
                ref_img = Image.open(frame_path).convert('L')
                ref_np = np.array(ref_img)
                mask_np = np.ones_like(ref_np, dtype=bool)
                img_tensor = torch.from_numpy(mask_np).unsqueeze(0) # Add channel dimension
            else:
                raise ValueError(f"Unsupported data_type: {data_type}")
        except FileNotFoundError:
            logger.error(f"File not found during loading: {frame_path}")
            # Return a tensor of zeros as a fallback
            if data_type == 'rgb':
                return torch.zeros(3, *self.output_size)
            else:
                return torch.zeros(1, *self.output_size)

        # Apply initial resize if necessary (to make it large enough for cropping)
        if resize_size:
            img_tensor = F.resize(img_tensor, resize_size, interpolation=Image.Resampling.BILINEAR)
        
        # Apply crop
        if self.use_random_crop and crop_params:
            img_tensor = F.crop(img_tensor, *crop_params)
        else:
            # Use center crop if not using random crop
            img_tensor = F.center_crop(img_tensor, self.output_size)
            
        # Final resize to ensure output size is exact, handling any minor discrepancies
        current_size = [img_tensor.shape[-2], img_tensor.shape[-1]]
        if current_size != list(self.output_size):
            img_tensor = F.resize(img_tensor, self.output_size, interpolation=Image.Resampling.BILINEAR)
            
        return img_tensor
