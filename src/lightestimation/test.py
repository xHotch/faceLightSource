import copy
import logging
import os
import sys

import numpy as np
import wandb
import cv2
import torch
from torch.utils.data import DataLoader
import torchvision # type: ignore
from tqdm import tqdm

from lightestimation.data.metadata import SingleRenderMetadata
from lightestimation.model_configs import get_face2env_test_loader
from lightestimation.utils.config import Config
from lightestimation.utils.model_utils import get_face2env_network, get_face2env_network_from_split_models
from lightestimation.utils.output_utils import write_metrics
from lightestimation.models.brightness_estimation import Env2EnvAutoEncoder
from lightestimation.utils.image_utils import tonemap_images, write_single_run_images
from lightestimation.utils.visualize import visualize_batch
from lightestimation.models.model import ModelWrapper, predict_model

logger = logging.getLogger(__name__)

def main(model_path, second_model_path=None):
    if second_model_path:
        network = get_face2env_network_from_split_models(model_path, second_model_path)
    else:
        network = get_face2env_network(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ModelWrapper(network, loss_fn=None)
    model.update_device(device)
    
    test_loader = get_face2env_test_loader(batch_size=1, visualized=False, subset_size=None, only_faces=False, output_cubemap_id=False)
    
    test_epoch(model, test_loader, save_tonemapped=True, output_path="output/", second_model = None)

def test_epoch(model: ModelWrapper, loader: DataLoader, visualize: bool=False, save_tonemapped:bool=True, output_path:str|None=None, second_model: Env2EnvAutoEncoder|None = None) -> tuple[float, dict[str, float], wandb.Image|None, dict|None]:
    performance_metrics = Config.get_performance_metrics()
    tonemapped_outputs: list[np.ndarray] = []
    tonemapped_labels: list[np.ndarray] = []
    outputs = []
    _labels = []
    metrics: dict[str, float] = {}
    images = None
    all_metadatas: list[SingleRenderMetadata] = []
    total_metrics:dict[str,dict[int, float]] = {}
    gamma = 2.2
    logger.info(f"Testing model")
    for i, batch in enumerate(tqdm(loader)):
        logger.debug(f"Starting batch {i}")
        if len(batch) == 2:
            data, labels = batch
        elif len(batch) == 3:
            data, labels, metadata_indices = batch
            all_metadatas.extend(loader.dataset.metadatas[i] for i in metadata_indices) # type: ignore
        
        output = model.predict(data)
        if second_model:
            labels = labels.to(output.device)
            output = predict_model(second_model.decoder, output, model.device)
        else:
            labels = labels.to(output.device)
        for metric in performance_metrics:
            value: torch.Tensor = metric(output, labels, i)
            if output_path:
                if metric.name not in total_metrics:
                    total_metrics[metric.name] = {}
                total_metrics[metric.name][i] = float(value)
        if visualize:
            visualize_batch(data, labels, output)
        if save_tonemapped:
            labels = labels.to("cpu")
            output = output.to("cpu")
                        
            # convert to numpy For batch size > 1
            output_np = copy.deepcopy(output.numpy())
            label_np =  copy.deepcopy(labels.numpy())
            
            # loop over batch size and transpose
            for i in range(output_np.shape[0]):
                _labels.append(label_np[i,:,:,:].transpose(1,2,0))
                outputs.append(output_np[i,:,:,:].transpose(1,2,0))
            
            tonemap_images(labels, output, tonemapped_labels, tonemapped_outputs, max_size=None, gamma=gamma)
                            
            
    for metric in performance_metrics:
        test_acc = torch.mean(torch.cat(metric.values),0)
        metrics[metric.name] = test_acc.item()  
    
    if save_tonemapped:
        tonemapped_outputs_np = np.asarray(tonemapped_outputs)
        tonemapped_labels_np = np.asarray(tonemapped_labels)
        
        upload_image_array = False
        if upload_image_array:
            image_array = np.concatenate([tonemapped_outputs_np, tonemapped_labels_np], axis=-3)
            image_array = torch.from_numpy(image_array).permute(0,3,1,2)
            image_array = torchvision.utils.make_grid(image_array, nrow=8)
            try:
                images = wandb.Image(image_array, mode="RGB", caption=f"Tonemapped output (top) and tonemapped ground truth (bottom). cv2.createTonemap({gamma})")
            except Exception as e:
                logger.error("Could not create wandb image")
                logger.error(e)
                images = None
        if output_path is not None:
            folder = output_path
            os.makedirs(folder, exist_ok=True)
            if upload_image_array:
                image_array = image_array.permute(1,2,0).numpy()
                cv2.imwrite(output_path+"/array.png", image_array)
            
            write_metrics(metrics, folder)
            
            for i, (tonemapped_output, tonemapped_label, label, output) in enumerate(zip(tonemapped_outputs_np, tonemapped_labels_np, _labels, outputs)):
                if all_metadatas is not None:
                    metadata = all_metadatas[i]
                    i = int(metadata.cubemap_index)
                    with open(os.path.join(folder, f"{i}_metadata.json"), "w") as f:
                        f.write(metadata.to_json())
                write_single_run_images(folder, i, tonemapped_output, tonemapped_label, label, output)
                
                
    output_values = torch.cat(performance_metrics[0].values, dim=0)
    return torch.mean(output_values).item(), metrics, images, total_metrics
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) != 2:
        model_path = r"models//face2envnetwork.pth"
    else:
        model_path = sys.argv[1]
    main(model_path)