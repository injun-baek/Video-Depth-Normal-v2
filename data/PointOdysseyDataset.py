import os
import cv2
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import logging

logger = logging.getLogger(__name__)

class PointOdysseyDataset(Dataset):
    """
    Point Odyssey Dataset for video depth estimation.

    This dataset loads RGB, depth, and depth_anything_v2 sequences
    from the Point Odyssey dataset. It supports sequence loading, random cropping,
    and returns file paths along with the data.
    """
    def __init__(self, 
                 root_dir, 
                 data_types=['rgb', 'depth'],
                 sequence_length=8, 
                 sequence_stride=1,
                 output_size=(480, 640), 
                 use_random_crop=False,
                 exclude_scenes: list = None):
        """
        Initializes the PointOdysseyDataset.

        Args:
            root_dir (str): Root directory of the Point Odyssey dataset.
            data_types (list): List of data types to load (e.g., 'rgb', 'depth', 'depth_anything_v2').
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
        self.data_types, self.sequence_length, self.sequence_stride = data_types, sequence_length, sequence_stride
        self.output_size, self.use_random_crop = output_size, use_random_crop
        self.exclude_scenes = exclude_scenes if exclude_scenes is not None else []
        
        self.folder_map = {
            'rgb': 'rgbs',
            'depth': 'depths',
            'depth_anything_v2': 'depth_anything_v2',
            'mask': 'masks'
        }

        if not isinstance(sequence_length, int) or sequence_length < 1:
            raise ValueError("sequence_length must be a positive integer.")
        if not isinstance(sequence_stride, int) or sequence_stride <= 0:
            raise ValueError("sequence_stride must be a positive integer.")

        self.data_pointers = self._create_data_pointers()
        
        if not self.data_pointers:
            raise RuntimeError("No valid data found.")
        print(f"PointOdysseyDataset initialized. Found {len(self.data_pointers)} sequences.")

    def _create_data_pointers(self) -> list:
        """
        Creates a list of dictionaries, where each dictionary contains
        paths to frames/sequences for each data type.
        It now iterates through 'train' and 'test' subdirectories within root_dir
        to collect all available data.

        Returns:
            list: A list of data pointers.
        """
        pointers = []
        train_test_dirs = ['train', 'test']
        
        for sub_dir in train_test_dirs:
            current_path = os.path.join(self.root_dir, sub_dir)
            if not os.path.isdir(current_path):
                logger.warning(f"Skipping directory '{current_path}' as it does not exist.")
                continue

            scenes = sorted([d for d in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, d))])

            for scene in scenes:
                if scene in self.exclude_scenes:
                    logger.info(f"Skipping excluded scene '{scene}' in '{sub_dir}' as specified in config.")
                    continue

                scene_path = os.path.join(current_path, scene)
                type_files = {}
                min_frames = float('inf')
                all_types_valid = True

                for data_type in self.data_types:
                    folder_name = self.folder_map.get(data_type, data_type)
                    data_path = os.path.join(scene_path, folder_name)

                    if not os.path.isdir(data_path):
                        all_types_valid = False
                        break
                    
                    if data_type == 'rgb':
                        files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.jpg')])
                    else:
                        files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.png')])
                    
                    if not files:
                        all_types_valid = False
                        break
                    
                    type_files[data_type] = files
                    min_frames = min(min_frames, len(files))

                if not all_types_valid:
                    continue

                file_counts = {dt: len(files) for dt, files in type_files.items()}
                if len(set(file_counts.values())) > 1:
                    raise ValueError(f"Inconsistent file counts for scene '{scene}' in '{sub_dir}' across data types: {file_counts}")

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
            
            # PIL Image.size is (width, height), but output_size is (height, width)
            w, h = first_image.size  # width, height
            target_h, target_w = self.output_size  # height, width
            
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
        elif data_type == 'depth':
            img_np = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            img_np = img_np.astype(np.float32)
            # for visualizing custom handling
            max_val = img_np.max()
            zero_mask = (img_np == 0)
            img_np[zero_mask] = max_val + 1
        elif data_type == 'depth_anything_v2':
            img_np = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            img_np = img_np.astype(np.float32)
        elif data_type == 'mask':
            # due to code simplify. allow below code
            depth_path = frame_path.replace('/masks/', '/depths/').replace('mask_', 'depth_')
            depth_np = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            custom_mask_np = (depth_np != 0)
            
            img_np = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            img_np = np.any(img_np != 0, axis=-1)
            img_np = np.logical_and(img_np, custom_mask_np)
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        current_size = [img_tensor.shape[-2], img_tensor.shape[-1]]
        if resize_size and resize_size != current_size:
            img_tensor = F.resize(img_tensor, resize_size, interpolation=Image.Resampling.NEAREST)
        if self.use_random_crop and crop_params:
            img_tensor = F.crop(img_tensor, *crop_params)
        current_size = [img_tensor.shape[-2], img_tensor.shape[-1]]
        if current_size != list(self.output_size):
            img_tensor = F.resize(img_tensor, self.output_size, interpolation=Image.Resampling.NEAREST)
        return img_tensor

    def find_scenes_with_zero_in_depth(self) -> list:
        """
        Returns a list of scene names that contain at least one depth image with a value of 0.
        """
        scenes_with_zero = set()
        train_test_dirs = ['train', 'test']

        for sub_dir in train_test_dirs:
            current_path = os.path.join(self.root_dir, sub_dir)
            if not os.path.isdir(current_path):
                continue

            scenes = sorted([d for d in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, d))])

            for scene in scenes:
                if scene in self.exclude_scenes:
                    continue

                scene_path = os.path.join(current_path, scene)
                depth_dir = os.path.join(scene_path, self.folder_map.get('depth', 'depths'))
                if not os.path.isdir(depth_dir):
                    continue

                depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.png')])
                for depth_file in depth_files:
                    img_np = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                    if img_np is not None and np.any(img_np == 0):
                        scenes_with_zero.add(scene)
                        break

        return sorted(list(scenes_with_zero))

    def zero_ratio_per_scene_in_depth(self) -> list:
        """
        Returns a list of (scene, zero_ratio) pairs, where zero_ratio is the proportion of zero pixels
        across all depth images in the scene.
        """
        results = []
        train_test_dirs = ['train', 'test']

        for sub_dir in train_test_dirs:
            current_path = os.path.join(self.root_dir, sub_dir)
            if not os.path.isdir(current_path):
                continue

            scenes = sorted([d for d in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, d))])

            for scene in scenes:
                if scene in self.exclude_scenes:
                    continue

                scene_path = os.path.join(current_path, scene)
                depth_dir = os.path.join(scene_path, self.folder_map.get('depth', 'depths'))
                if not os.path.isdir(depth_dir):
                    continue

                depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.png')])
                zero_count = 0
                total_count = 0
                for depth_file in depth_files:
                    img_np = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                    if img_np is not None:
                        zero_count += np.count_nonzero(img_np == 0)
                        total_count += img_np.size
                if total_count > 0:
                    zero_ratio = zero_count / total_count
                    results.append((scene, zero_ratio))
                    
        '''
        ani: 0.170584
        ani11_new_: 0.129775
        ani11_new_f: 0.165095
        ani12_new_: 0.209935
        ani13_new_f: 0.165322
        ani15_new_: 0.262229
        ani15_new_f: 0.271109
        ani16_new_f: 0.249213
        ani18_new_f: 0.260281
        ani2_s: 0.357026
        ani4_new_: 0.117779
        ani5_new_: 0.127060
        ani5_new_f: 0.107990
        ani6_new_: 0.104441
        ani7_new_: 0.196788
        ani8_new_f: 0.146252
        ani9_new_: 0.241857
        ani9_new_f: 0.162431
        ani_new_: 0.076558
        ani_new_2: 0.112611
        ani_s: 0.227330
        animal: 0.226707
        animal2_: 0.338657
        animal3: 0.205693
        animal3_s: 0.257273
        animal4_s: 0.249670
        animal5: 0.147687
        animal6: 0.188444
        animal8: 0.257362
        animal_s: 0.194031
        cab_e1_3rd: 0.239486
        cab_e_ego2: 0.070735
        cab_h_bench_3rd: 0.000000
        cab_h_bench_ego1: 0.000000
        cab_h_bench_ego2: 0.376243
        character0: 0.242910
        character2: 0.066113
        character2_: 0.174059
        character2_f: 0.334862
        character3_f: 0.266724
        character4: 0.139601
        character4_: 0.161629
        character4_f: 0.253556
        character5: 0.034788
        character5_: 0.135841
        character5_f: 0.148638
        character6_f: 0.076987
        cnb_dlab_0215_3rd: 0.000477
        cnb_dlab_0215_ego1: 0.000227
        dancing: 0.003013
        dancingroom0_3rd: 0.000475
        footlab_ego1: 0.001197
        gso_out_big: 0.252318
        human_in_scene: 0.014965
        kg_ego1: 0.003189
        kg_ego2: 0.056593
        kitchen_gfloor: 0.033378
        kitchen_gfloor_ego2: 0.035117
        r0_new: 0.219491
        r0_new_: 0.157382
        r0_new_f: 0.157382
        r0_new_f_: 0.405271
        r2_new_f: 0.202068
        r3_new_f: 0.263552
        r5_new_: 0.159801
        r5_new_f: 0.214316
        r6_new_f: 0.210591
        r7_new_: 0.069601
        r7_new_f: 0.200641
        scene_carb_h_tables_ego1: 0.001517
        scene_carb_h_tables_ego2: 0.001201
        scene_d78_0318_3rd: 0.190022
        scene_j716_3rd: 0.001167
        scene_recording_20210910_S05_S06_0_ego2: 0.002192
        seminar_h52_3rd: 0.005155
        ani12_new_f: 0.253935
        ani14_new_: 0.202030
        ani2: 0.256841
        anima5_s: 0.296529
        cab_e_3rd: 0.001199
        dancingroom1_3rd1: 0.550642
        dancingroom1_3rd2: 0.189152
        dancingroom3_3rd: 0.045695
        egobody_egocentric: 0.002286
        seminar_g110_0315_3rd: 0.000043
        seminar_g110_0315_ego1: 0.000011
        seminar_g110_0315_ego2: 0.157461
        '''
        return results
    
    def find_all_empty_mask_paths(self) -> list:
        """
        데이터셋 전체를 순회하며, 모든 값이 0인 mask 파일의 경로를 리스트로 반환합니다.
        """
        empty_mask_paths = []
        train_test_dirs = ['train', 'test']

        for sub_dir in train_test_dirs:
            current_path = os.path.join(self.root_dir, sub_dir)
            if not os.path.isdir(current_path):
                continue

            scenes = sorted([d for d in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, d))])

            for scene in scenes:
                if scene in self.exclude_scenes:
                    continue

                scene_path = os.path.join(current_path, scene)
                mask_dir = os.path.join(scene_path, self.folder_map.get('mask', 'masks'))
                if not os.path.isdir(mask_dir):
                    continue

                mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])
                for mask_file in mask_files:
                    img_np = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
                    if img_np is not None and np.all(img_np == 0):
                        empty_mask_paths.append(mask_file)

        return empty_mask_paths