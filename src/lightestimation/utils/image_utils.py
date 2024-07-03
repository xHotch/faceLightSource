import logging
import os

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torchvision # type: ignore[import-untyped]


logger = logging.getLogger(__name__)

def _tonemap_drago(hdr: np.ndarray, gamma = 2.2, *args, **kwargs) -> np.ndarray:
    if hdr.shape[2] == 3:
        hdr = cv2.cvtColor(hdr, cv2.COLOR_RGB2BGR)
    tonemap = cv2.createTonemapDrago(gamma)
    scale = 1.0
    ldr = tonemap.process(hdr) * scale
    ldr_rgb = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB)
    ldr_rgb = np.clip(ldr_rgb*255, 0, 255).astype(np.uint8)
    return ldr_rgb

def _tonamap_drago_torch(image: torch.Tensor, gamma = 2.2, *args, **kwargs) -> np.ndarray:
    tonemap = cv2.createTonemapDrago(gamma)
    image_np = image.numpy().transpose(1,2,0)
    immax = np.max(image_np)
    if not immax > 0:
        print("adding eps to image")
        image_np = image_np + 0.00001
    if image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    ldr = tonemap.process(image_np)
    ldr = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB)
    ldr = np.clip(ldr*255, 0, 255).astype(np.uint8)
    return ldr

def _tonemap_clip(hdr: np.ndarray, *args, **kwargs) -> np.ndarray:
    ldr = np.clip(hdr, 0, 1)
    ldr = ldr**(1/2.2)
    ldr = ldr * 255
    return ldr

def _tonemap_clip_torch(image: torch.Tensor, *args, **kwargs) -> np.ndarray:
    image = image.numpy().transpose(1,2,0)
    ldr = np.clip(image, 0, 1)
    ldr = ldr**(1/2.2)
    ldr = np.clip(ldr*255, 0, 255).astype(np.uint8)
    return ldr

def tone_map(image: torch.Tensor|np.ndarray, tonemap: str = "drago", *args, **kwargs) -> np.ndarray:
    if tonemap == "drago":
        if type(image) == torch.Tensor:
            return _tonamap_drago_torch(image, *args, **kwargs)
        elif type(image) == np.ndarray:
            return _tonemap_drago(image, *args, **kwargs)
        raise ValueError(f"Image type {type(image)} not supported.")
    elif tonemap == "clip":
        if type(image) == torch.Tensor:
            return _tonemap_clip_torch(image, *args, **kwargs)
        elif type(image) == np.ndarray:
            return _tonemap_clip(image, *args, **kwargs)#
        raise ValueError(f"Image type {type(image)} not supported.")
    raise ValueError(f"Tonemap {tonemap} not supported.")

def tonemap_images(labels: torch.Tensor, output: torch.Tensor, tonemapped_labels: list[np.ndarray], tonemapped_outputs: list[np.ndarray], max_size: int|None = 64, gamma: float=2.4) -> None:
    if max_size and len(tonemapped_labels) > max_size:
        return
    try:
        for i in range(labels.shape[0]):
            tonemapped_labels.append(tone_map(labels[i], "drago", gamma=gamma))
        for i in range(output.shape[0]):
            tonemapped_outputs.append(tone_map(output[i], "drago", gamma=gamma))
    except Exception as e:
        logger.error("Exception during tonemapping.")
        logger.error(e, exc_info=True)

def create_image_array(tonemapped_outputs: list[np.ndarray], tonemapped_labels: list[np.ndarray]) -> torch.Tensor:
    tonemapped_outputs_np = np.asarray(tonemapped_outputs)
    tonemapped_labels_np = np.asarray(tonemapped_labels)
    
    image_array = np.concatenate([tonemapped_outputs_np, tonemapped_labels_np], axis=-3)
    image_array = torch.from_numpy(image_array).permute(0,3,1,2)
    image_array = torchvision.utils.make_grid(image_array, nrow=8)
    return image_array

def write_single_run_images(folder: str, i: int, tonemapped_output: np.ndarray, tonemapped_label: np.ndarray, label: torch.Tensor, output: torch.Tensor) -> None:
    if not os.path.exists(folder):
        os.makedirs(folder)
    model_output_filepath = os.path.join(folder, f"{i}_tonemapped_output.png")
    toneemapped_label_filepath = os.path.join(folder, f"{i}_tonemapped_label.png")
    label_filepath = os.path.join(folder, f"{i}_label.hdr")
    output_filepath = os.path.join(folder, f"{i}_output.hdr")
    
    cv2.imwrite(model_output_filepath, tonemapped_output)
    cv2.imwrite(toneemapped_label_filepath, tonemapped_label)
    cv2.imwrite(output_filepath, np.float32(output)) # type: ignore
    cv2.imwrite(label_filepath, np.float32(label)) # type: ignore