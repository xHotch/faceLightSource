import os
from typing import List
import cv2
import numpy as np
import logging

from lightestimation.data.metadata import SingleRenderMetadata
logger = logging.getLogger(__name__)

def is_laval(metadata: SingleRenderMetadata) -> bool:
    return is_string_laval(metadata.cubemap_path)

def is_string_laval(name: str) -> bool:
    name = str(name)
    if "/calibrated/" in name.lower():
        return True
    if "/laval_outdoor/" in name.lower():
        return True
    if "9C4A" in name and not "-9C4A" in name:
        return True
    if "AG8A" in name and not "-AG8A" in name:
        return True
    return False

def get_val_split() -> list[int]:
    with open("data/train/val_split.txt", "r") as file:
        indices = file.readlines()
    return [int(index) for index in indices]

def get_metadata(directory, subset_size = None, name_subset = None, envmap_subset = None, remove_dark_images = True) -> tuple[List[SingleRenderMetadata], List[int]]:
    metadata_filepath = os.path.join(directory, "metadata.csv")
    if not os.path.exists(metadata_filepath):
        directory = os.path.join(directory,"..")
        metadata_filepath = os.path.join(directory, "_metadata.csv")

    if not os.path.exists(metadata_filepath):
        metadata_filepath = next((os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".csv")), "")
    
    if metadata_filepath == "" and not directory.endswith("merged"):
        directory = os.path.join(directory,"..", "merged")
        metadata_filepath = next((os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".csv")), "")
        
    if metadata_filepath == "":
        raise FileNotFoundError("No metadata file found")
    
    if subset_size is not None:
        if envmap_subset is not None:
            envmap_subset = envmap_subset[:subset_size]
        if name_subset is not None:
            name_subset = name_subset[:subset_size]
            
    logger.info(f"Reading metadata from {metadata_filepath}.")
    metadata = SingleRenderMetadata.list_from_csv(metadata_filepath)
    if name_subset is not None:
        metadata = [m for m in metadata if m.metahuman_name in name_subset]
    if envmap_subset is not None:
        metadata = [m for m in metadata if int(m.cubemap_index) in envmap_subset]
    if remove_dark_images:
        metadata = [m for m in metadata if is_dark_image(m, directory) == False]
        
    metadata = remove_not_existing_files(metadata, directory)
        
    metadata.sort(key=lambda x: (x.metahuman_name, x.cubemap_index))
    
    return metadata, envmap_subset

def remove_not_existing_files(metadatas: List[SingleRenderMetadata], folder: str) -> List[SingleRenderMetadata]:
    logger.info(f"Removing not existing files from {folder}")
    new_metadatas = []
    for metadata in metadatas:
        if not os.path.exists(build_filepath(metadata, get_face_image_directory(folder), ".png")):
            logger.debug(f"File {build_filepath(metadata, folder, '.png')} not found")
            continue
        if not os.path.exists(build_filepath(metadata, get_label_directory(folder), ".hdr")):
            logger.debug(f"File {build_filepath(metadata, folder, '.hdr')} not found")
            continue
        new_metadatas.append(metadata)
    return new_metadatas

def is_dark_image(metadata: SingleRenderMetadata, dataset_dir: str) -> bool:
    filename = get_filename(metadata, ".png")
    folder = get_dark_image_directory(dataset_dir)
    if os.path.exists(os.path.join(folder, filename)):
        return True
    return False

def read_image(filepath: str) -> np.ndarray:
    try:
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            raise FileNotFoundError(f"File {filepath} not found")
        return image
    
    except Exception as e:
        logger.error(f"Error during loading data: {e}")
        if image is None:
            raise FileNotFoundError(f"None when using random image")
        return image
        
def read_envmap(filepath: str) -> np.ndarray:
    try:
        hdr_img = cv2.imread(filepath, flags=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        label = hdr_img
        if label is None:
            raise FileNotFoundError(f"File {filepath} not found")
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        return label
    except Exception as e:
        logger.error(f"Error during loading label: {e}")
        raise e
    
def build_filepath(metadata: SingleRenderMetadata, folder, ending) -> str:
    filename = get_filename(metadata, ending)
    filepath = os.path.join(folder, filename)
    
    return filepath

def get_label_directory(directory: str) -> str:
    return os.path.join(directory, "labels")

def get_face_image_directory(directory: str, only_faces = False) -> str:
    if only_faces:
        return os.path.join(directory, "segmented")
    return os.path.join(directory, "data")

def get_metadata_names(metadatas: List[SingleRenderMetadata]):
    return list(set([metadata.metahuman_name for metadata in metadatas]))

def get_dark_image_directory(directory: str) -> str:
    return os.path.join(directory, "dark_images")

def get_filename(metadata: SingleRenderMetadata, ending = ".png") -> str:
    return metadata.metahuman_name + "_" + str(metadata.cubemap_index) + ending
