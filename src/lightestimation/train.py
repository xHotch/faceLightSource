import logging
import os
import time
import random
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision # type: ignore[import-untyped]
import torch.nn as nn
from tqdm import tqdm
import wandb
import wandb.sdk

from lightestimation.evaluation.accuracy import PerformanceMetric
from lightestimation.test import test_epoch
from lightestimation.utils.config import Config, Config
from lightestimation.utils.image_utils import create_image_array, tonemap_images
from lightestimation.utils.model_utils import get_decoder_state_dict, get_env2env_autoencoder, get_model_filename, save_model
from lightestimation.utils.wandb_utils import add_summary, get_config_and_modelname_from_run, log, log_train
from lightestimation.evaluation.loss import *
from lightestimation.model_configs import get_model_config
from lightestimation.models.model import ModelWrapper, predict_model
from lightestimation.models.brightness_estimation import Env2EnvAutoEncoder
from lightestimation.utils.visualize import visualize_batch

__seed = 0
random.seed(__seed)
torch.manual_seed(__seed)

logger = logging.getLogger(__name__)

def main(config: dict|None) -> None:
    """Runs the training loop for the model.
    """
                
    label_model = None
    project_name = Config.WANDB_PROJECT_PATH.split("/")[-1]
    
    if config is None:
        if Config.DISABLE_WANDB:
            wandb.init(mode="disabled")
        else:
            wandb.init(project=project_name, dir=Config.WANDB_RUN_DIR)
        config = wandb.config
    else:
        if Config.DISABLE_WANDB:
            wandb.init(mode="disabled")
        else:
            wandb.init(project=project_name, dir=Config.WANDB_RUN_DIR, config=config)
    
    decoder_weights = None
    if True:
        if config["model"] == "Face2EnvNetwork":
            if "decoder_run_name" in config:
                run_name = config["decoder_run_name"]
                logger.info(f"Downloading encoder from run {run_name}")
                _, encoder_path = get_config_and_modelname_from_run(run_name)
                decoder_weights = get_decoder_state_dict(encoder_path)
        elif config["model"] == "FaceEncoder":
            if "decoder_run_name" in config:
                run_name = config["decoder_run_name"]
                logger.info(f"Downloading encoder from run {run_name}")
                _, encoder_path = get_config_and_modelname_from_run(run_name)
                label_model = get_env2env_autoencoder(encoder_path)
            else:
                if int(config["bottleneck_size"]) == 64:
                    run_name = "9q0l6ic3"
                else:
                    run_name = "bk2iwg5p"
                _, encoder_path = get_config_and_modelname_from_run(run_name)
                label_model = get_env2env_autoencoder(encoder_path)

            decoder_weights = None     
               
    model, train_dataset, val_dataset, test_dataset = get_model_config(config, visualized=False, lazy_load=True, subset_size=10, decoder_weights=decoder_weights)
    
    if Config.WATCH_MODEL:
        # Log the model to wandb
        log_freq = config["epochs"] // 10
        log_freq = 1000
        wandb.watch(model.net, log="all", log_freq=log_freq)
    

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True, num_workers=1)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True, num_workers=1)
    
    # Check if we can use the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")       
            
    logger.info(f"Using primary device: {device}")
    if type(Config.GPU_ID) == list:
        logger.info(f"Using multiple GPUs: {Config.GPU_ID}")
        model.net = nn.DataParallel(model.net, device_ids=Config.GPU_ID)
        if label_model is not None:
            label_model == nn.DataParallel(label_model, device_ids=Config.GPU_ID)
                        
    model.update_device(device)
    
    performance_metrics = Config.get_performance_metrics()
    
    if label_model is not None:
        label_model.to(model.device)
    try:
        train_loop(model, train_dataloader, val_dataloader, config = config, performance_metrics = performance_metrics, label_model = label_model)
    except Exception as e:
        logger.error(f"Error during training. Saving model and exiting")
        logger.error(e, exc_info=True)
        save_model("error", model, config)

    logger.info("Validating with best model")
    filename = get_model_filename(model)
    
    model.net.load_state_dict(torch.load(filename)["model_state_dict"])
    save_model("best", model, config)
    epoch_acc, metrics, images = validate_epoch(model, "1 epoch with best", val_dataloader, performance_metrics = performance_metrics, visualize=False, save = True, second_model= label_model)
    epoch_acc = epoch_acc.to("cpu")
    log(epoch_acc, metrics, images)
    
    
    # Free memory by deleting the train and val dataloaders
    del train_dataloader
    del val_dataloader
    del train_dataset
    del val_dataset
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True, num_workers=4)
    
    test_acc, test_metrics, test_images, _ = test_epoch(model, test_dataloader, visualize=False, save_tonemapped = True, second_model = label_model)   
    log(test_acc, test_metrics, test_images, split = "test")

def train_loop(model: ModelWrapper, train_dataloader, val_dataloader, config, performance_metrics: list[PerformanceMetric], label_model: Optional[Env2EnvAutoEncoder] = None) -> None:
    best_acc = np.inf
    start_time = time.perf_counter()
    early_stop_count = 0
    for epoch in range(config["epochs"]):
        epoch_loss = train_epoch(model, epoch, train_dataloader, label_model)
        logger.info(f"Epoch {epoch} loss: {epoch_loss}")
        if epoch % Config.IMAGE_LOG_FREQUENCY == 0:
            epoch_acc, metrics, images = validate_epoch(model, epoch, val_dataloader, performance_metrics, save = True, second_model= label_model)
        else:
            epoch_acc, metrics, images = validate_epoch(model, epoch, val_dataloader, performance_metrics, save = False, second_model= label_model)
        model.update_schedule(epoch_acc)
        logger.info(f"Epoch {epoch} accuracy: {epoch_acc}")
        
        
        if check_early_stopping(epoch_acc, best_acc):
            early_stop_count = 0
            save_model(epoch, model, config, only_torch= True)
        else:
            early_stop_count += 1
            if early_stop_count >= config["early_stop"]:
                logger.info(f"Early stopping at epoch {epoch}.")
                break
        log_train(model, epoch_loss, epoch, epoch_acc, metrics, images)
    end_time = time.perf_counter()
    logger.info(f"Training took {end_time - start_time} seconds")
    
def train_epoch(model: ModelWrapper, epoch: int, loader: DataLoader, second_model: Optional[Env2EnvAutoEncoder] = None) -> np.floating:
    logger.info(f"Training epoch {epoch}")
    epoch_loss:list[float] = []
    i = 0
    try:
        for i, batch in enumerate(tqdm(loader)):
            logger.debug(f"Starting batch {i}")
            data, labels = batch
            if second_model:
                labels = predict_model(second_model.encoder, labels, model.device)
            loss = model.train(data, labels) 
            epoch_loss.append(loss)
    except Exception as e:
        logger.error(f"Error during training epoch {epoch}, index {i}")
        logger.error(e, exc_info=True)
    return np.mean(epoch_loss)
        
def validate_epoch(model: ModelWrapper, epoch: int|str, loader: DataLoader, performance_metrics: list[PerformanceMetric], visualize=False, save = False, second_model: Optional[Env2EnvAutoEncoder] = None) -> tuple[torch.Tensor, dict, Optional[wandb.Image]]:
    metrics = {}
    images = None
    gamma = 2.4
    tonemapped_outputs: list[np.ndarray] = []
    tonemapped_labels: list[np.ndarray] = []
    logger.info(f"Validating epoch {epoch}")
    for i, batch in enumerate(tqdm(loader)):
        logger.debug(f"Starting batch {i}")
        data, labels = batch
        output = model.predict(data)
        labels = labels.to(output.device)
        
        if second_model:
            output = predict_model(second_model.decoder, output, model.device)

        for metric in performance_metrics:
            metric(output, labels, i)
        if visualize:
            visualize_batch(data, labels, output)
        if save:
            # Only save the first batch to avoid memory issues
            if tonemapped_outputs:
                ...
            else:
                labels = labels.to("cpu")
                output = output.to("cpu")
                tonemap_images(labels, output, tonemapped_labels, tonemapped_outputs, gamma=gamma)
                            
    if save:
        image_array = create_image_array(tonemapped_outputs, tonemapped_labels)  
        images = wandb.Image(image_array, mode="RGB", caption=f"Tonemapped output (top) and tonemapped ground truth (bottom). cv2.createTonemap({gamma})")
    
    for metric in performance_metrics:
        epoch_acc = torch.mean(torch.cat(metric.values),0)
        metrics[metric.name] = epoch_acc
        
    values = torch.cat(performance_metrics[0].values, dim=0)
    return torch.mean(values), metrics, images
  
def check_early_stopping(epoch_acc, best_acc) -> bool:
    if epoch_acc < best_acc:
          logger.info(f"Saving Model. New best accuracy: {epoch_acc}")
          best_acc = epoch_acc
          add_summary({"best_accuracy": best_acc})
          dir = "models"
          if not os.path.exists(dir):
              os.mkdir(dir)
          return True
    return False
         
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torchvision.disable_beta_transforms_warning()
    
    USE_SWEEP = False
    if USE_SWEEP:
        main(None)
    else:
        run_config =  {
            "epochs": 1,
            "batch_size": 1,
            "initial_learning_rate": 1,
            "model": "Env2EnvAutoEncoderBrightness",
            "loss_fn": "LogL1Loss_solid_angle",
            "early_stop": 200,
            "bottleneck_size" : 512,
            "activation" : "nn.LeakyReLU",
            "relu_output": True
        }
        run_config_rgb =  {
            "epochs": 1,
            "batch_size": 8,
            "initial_learning_rate": 1.0,
            "model": "Env2EnvAutoEncoder",
            "loss_fn": "L1Loss_solid_angle",
            "early_stop": 200,
            "bottleneck_size" : 512,
            "activation" : "nn.LeakyReLU",
            "relu_output": True
        }
        run_config_face2env = {
            "epochs": 1,
            "batch_size": 1,
            "initial_learning_rate": 1.0,
            "model": "Face2EnvNetwork",
            "loss_fn": "L1Loss_solid_angle",
            "early_stop": 45,
            "bottleneck_size" : 512,
            "activation" : "nn.LeakyReLU",
            "relu_output": True,
            "initial_feature_maps": 32,
            #"decoder_run_name": "vlkv8lfs",
            "train_decoder": True,
        }    
        run_config_face2latent = {
            "epochs": 1,
            "batch_size": 1,
            "initial_learning_rate": 1.0,
            "model": "FaceEncoder",
            "loss_fn": "L2Loss",
            "early_stop": 45,
            "bottleneck_size" : 512,
            "activation" : "nn.LeakyReLU",
            "relu_output": True,
            "initial_feature_maps": 128,
            "decoder_run_name": "hc5oghaf",
        }
        
        main(config=run_config)