import copy
import os
import random

import torch
random.seed(0)
from typing import List
import cv2

import numpy as np

from torch.utils.data import Dataset
from lightestimation.data.metadata import SingleRenderMetadata
from lightestimation.data.dataset_utils import build_filepath, get_face_image_directory, get_metadata, read_image, read_envmap, get_label_directory, get_metadata_names

import logging
logger = logging.getLogger(__name__)
NR_TRAIN_IMAGES = 2828

class EnvironmentToEnvironmentDataset(Dataset):
    data: List[np.ndarray]
    lightprobe_filepaths: List[str]
    metadatas: List[SingleRenderMetadata]
    dataset_dir: str
    visualized: bool = False
    
    def __init__(self, dataset_dir: str, transform = None, target_transform = None, random_transform = None, visualized: bool = False, subset_size = None, envmap_subset = None, lazy_load = False, use_random_lightprobe = True, name_subset = None, *args, **kwargs):
        self.visualized = visualized
        self.transform = transform
        self.target_transform = target_transform
        self.random_transform = random_transform
        self.dataset_dir = dataset_dir
        self.subset_size = subset_size
        self.metadatas, _ =get_metadata(dataset_dir, subset_size=self.subset_size, envmap_subset=envmap_subset, remove_dark_images=False, name_subset=name_subset)
        self.metadata_names = get_metadata_names(self.metadatas)
        self.data_folder = get_label_directory(dataset_dir)
        self.lightprobe_filepaths = [build_filepath(metadata, self.data_folder, ".hdr") for metadata in self.metadatas]
        self.lazy_load = lazy_load
        if not self.lazy_load:
            self._load_data()
        self.use_random_lightprobe = use_random_lightprobe
        if not self.lazy_load and self.use_random_lightprobe:
            raise ValueError("Non Lazy load and random lightprobe is activated. Not supported")
        self.len:int|None = None
        self.output_metadata = kwargs.get("output_metadata", False)
        
    def _load_data(self) -> None:
        logger.debug(f"Loading data")
        self.data = []
        for hdr_file in self.lightprobe_filepaths:
            self.data.append(read_envmap(hdr_file))
        for data in self.data:
            if data is None:
                raise RuntimeError("Data must not be None")
            
    def __len__(self) -> int:
        if self.len:
            return self.len
        first_metadata_name = self.metadatas[0].metahuman_name
        sum = 0
        for metadata in self.metadatas:
            if metadata.metahuman_name == first_metadata_name:
                sum += 1
        self.len = sum
        return sum
    
    def random_lightprobe(self, idx: int) -> tuple[str, int]:
        random_name = random.choice(self.metadata_names)
        metadatas = [metadata for metadata in self.metadatas if metadata.metahuman_name == random_name]
        try:
            random_metadata = metadatas[idx]
        except:
            logger.error(f"Could not find metadata for index{idx}. Length of self is {len(self)}. Number of metadatas for name {random_name}: {len(metadatas)}")
            exit()
        filepath = build_filepath(random_metadata, self.data_folder, ".hdr")
        logger.debug(f"Random lightprobe: {filepath}")
        return filepath, self.metadatas.index(random_metadata)
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]|tuple[torch.Tensor, torch.Tensor, int]:
        logger.debug(f"Getting item {idx}, from lightprobe {self.lightprobe_filepaths[idx]}")
        if idx >= len(self):
            raise IndexError("Index out of bounds")
        if idx < 0:
            raise IndexError("Index must not be negative")
        if not self.lazy_load:
            data_np = self.data[idx]
        else:
            if self.use_random_lightprobe:
                hdr_file, metadata_index = self.random_lightprobe(idx)
            else:
                raise NotImplementedError("Not implemented")
                hdr_file = self.lightprobe_filepaths[idx]
            data_np = read_envmap(hdr_file)
        label_np = copy.deepcopy(data_np)
        if self.target_transform:
            label = self.target_transform(label_np)
        else:
            label = torch.from_numpy(label_np)
        if self.transform:
            data = self.transform(data_np)
        else:
            data = torch.from_numpy(data_np)
            
        if self.random_transform:
            both_images = [data, label]
            transformed_images = self.random_transform(both_images)
            data = transformed_images[0]
            label = transformed_images[1]
        if self.visualized:
            label_img = label.numpy().transpose(1,2,0)#[..., ::-1]
            data_img = data.numpy().transpose(1,2,0)#[..., ::-1]
            cv2.imshow("label", label_img)
            cv2.imshow("data", data_img)
            cv2.waitKey(0)
        if data is None:
            raise RuntimeError("Data must not be None")
        if label is None:
            raise RuntimeError("Label must not be None")
        if self.output_metadata:
            return data, label, metadata_index
        return data, label

class FaceToEnvironmentDataset(Dataset):
    """Dataset for light estimation."""
    
    data: List[np.ndarray]
    labels: List[np.ndarray]
    filepaths: List[str]
    lightprobe_filepaths: List[str]
    metadatas: List[SingleRenderMetadata]
    dataset_directory: str
    label_directory: str
    visualized: bool = False
    
    def __init__(self, dataset_directory: str, transform = None, target_transform = None, random_transform = None, visualized: bool = False, subset_size = None, only_faces = False, envmap_subset = None, name_subset = None, lazy_load = False, *args, **kwargs):
        self.load_random = True
        self.len:int|None = None
        self.visualized = visualized     
        self.transform = transform
        self.target_transform = target_transform
        self.random_transform = random_transform
        self.only_faces = only_faces
        logger.debug(f"Loading dataset from {dataset_directory}")
        self.dataset_directory = dataset_directory
        self.label_directory = get_label_directory(self.dataset_directory)
        self.data_directory = get_face_image_directory(self.dataset_directory, only_faces=only_faces)
        self.subset_size = subset_size
        self.metadatas, self.indices =get_metadata(dataset_directory, subset_size=self.subset_size, envmap_subset=envmap_subset, remove_dark_images=True, name_subset=name_subset)
        self.metadata_names = get_metadata_names(self.metadatas)
        self.lazy_load = lazy_load
        self.output_metadata = kwargs.get("output_metadata", False)
        if not self.lazy_load:
            raise NotImplementedError("Not implemented")
            self._load_data()
        
    def _load_data(self) -> None:
        logger.debug(f"Loading data")
        self.data = []
        self.labels = []
        for file, hdr_file in zip(self.filepaths, self.lightprobe_filepaths):
            self.data.append(read_image(file))
            self.labels.append(read_envmap(hdr_file))
                
        for data in self.data:
            if data is None:
                raise RuntimeError("Data must not be None")
                
    def __len__(self) -> int:
        if not self.load_random:
            return len(self.metadatas)
        if self.indices is not None:
            return len(self.indices)
        if self.len:
            return self.len
        lengths = {}
        for metadata in self.metadatas:
            if not metadata.metahuman_name in lengths:
                lengths[metadata.metahuman_name] = 0
            lengths[metadata.metahuman_name] += 1
        self.len = max(lengths.values())
        return self.len
    
    def random_data(self, idx) -> tuple[str, str, int]:
        random_name = random.choice(self.metadata_names)
        metadatas = [metadata for metadata in self.metadatas if metadata.metahuman_name == random_name]
        cubemap_idx = self.indices[idx] if self.indices is not None else idx
        random_metadata = next((metadata for metadata in metadatas if int(metadata.cubemap_index)==cubemap_idx), None)
        if random_metadata is None:
            logger.warning(f"No face with idx {cubemap_idx} for name {random_metadata}. Using other metahuman")
            available = [metadata for metadata in self.metadatas if int(metadata.cubemap_index)==idx]
            
            if len(available) > 0:
                random_metadata = random.choice(available)
            else:
                other_idx = random.randint(0,len(self)-1)
                logger.warning(f"No face with idx {cubemap_idx} for anyone. Trying idx {other_idx}")
                return self.random_data(other_idx)
        
        filepath = build_filepath(random_metadata, self.data_directory, ".png")
        lightprobe_filepath = build_filepath(random_metadata, self.label_directory, ".hdr")
        if not os.path.exists(filepath):
            logger.warning(f"File {filepath} does not exist, because images is too dark. Trying another one")
            return self.random_data(idx)
        logger.debug(f"Random lightprobe: {filepath}")
        index_of_metadata = self.metadatas.index(random_metadata)
        return filepath, lightprobe_filepath, index_of_metadata
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]|tuple[torch.Tensor, torch.Tensor, int]:
        if idx >= len(self):
            raise IndexError("Index out of bounds")
        if idx < 0:
            raise IndexError("Index must not be negative")
        if self.load_random:
            file, hdr_file, metadata_index = self.random_data(idx)
            label_np = read_envmap(hdr_file)
            data_np = read_image(file)
        elif not self.lazy_load:
            raise NotImplementedError("Not implemented")
        else:
            raise NotImplementedError("Not implemented")
        if self.target_transform:
            label = self.target_transform(label_np)
        else:
            label = torch.from_numpy(label_np)
        if self.transform:
            data = self.transform(data_np)
        else:
            data = torch.from_numpy(data_np)
        if self.random_transform:
            both_images = [data, label]
            transformed_images = self.random_transform(both_images)
            data = transformed_images[0]
            label = transformed_images[1]
        if self.visualized:
            cv2.imshow("label", label.numpy().transpose(1,2,0))
            cv2.imshow("data", data.numpy().transpose(1,2,0))
            cv2.waitKey(0)
        if data is None:
            raise RuntimeError("Data must not be None")
        if label is None:
            raise RuntimeError("Label must not be None")
        if self.output_metadata:
            return data, label, metadata_index
        return data, label
