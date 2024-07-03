from datetime import datetime
import os
from typing import Any, Mapping
import torch
import torch.nn as nn
import logging

import wandb

from lightestimation.models.brightness_estimation import Decoder, Env2EnvAutoEncoder, Face2EnvNetwork, FaceEncoder
from lightestimation.models.model import ModelWrapper
from lightestimation.utils.config import Config
from lightestimation.utils.wandb_utils import upload_model

logger = logging.getLogger(__name__)

def load_state_dict(model: nn.Module, state_dict: Mapping[str, Any]) -> None:
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        logger.debug("Removing 'module' from the keys happens due to wrapping the model in nn.DataParallel")
        # Remove "module." from the keys happens due to wrapping the model in nn.DataParallel
        new_state_dict = { k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        
def rename_network_name(state_dict: Mapping[str, Any], old_name: str, new_name: str) -> Mapping[str, Any]:
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(old_name):
            new_key = key.replace(old_name, new_name)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def get_decoder_state_dict(autoencoder_filepath) -> Mapping[str, Any]:
    model = _load_autoencoder(autoencoder_filepath)
    return model.decoder.state_dict()

def get_decoder(autoencoder_filepath) -> Decoder:
    model = _load_autoencoder(autoencoder_filepath)
    return model.decoder

def get_faceencoder(model_filepath) -> FaceEncoder:
    data = torch.load(model_filepath)
    config = data["config"]
    bottleneck_size =config["bottleneck_size"]
    activation = config["activation"] if type(config["activation"]) != str else eval(config["activation"])
    initial_feature_maps = config["initial_feature_maps"]
    model = FaceEncoder(in_channels=1, bottleneck_size=bottleneck_size, activation=activation, initial_feature_maps=initial_feature_maps)
    load_state_dict(model, data["model_state_dict"])
    return model

def get_face2env_network(model_filepath) -> Face2EnvNetwork:
    data = torch.load(model_filepath)
    config = data["config"]
    bottleneck_size =config["bottleneck_size"]
    activation = config["activation"] if type(config["activation"]) != str else eval(config["activation"])
    initial_feature_maps = config["initial_feature_maps"]
    model = Face2EnvNetwork(in_channels=1, out_channels=1, decoder_state_dict=None, bottleneck_size=bottleneck_size, activation=activation, initial_feature_maps=initial_feature_maps)
    load_state_dict(model, data["model_state_dict"])
    return model

def get_face2env_network_from_split_models(face2latent_model_path, autoencoder_model_path) -> Face2EnvNetwork:
    data = torch.load(face2latent_model_path)
    config = data["config"]
    decoder_state_dict = get_decoder_state_dict(autoencoder_model_path)
    bottleneck_size = config["bottleneck_size"]
    initial_feature_maps = config["initial_feature_maps"]
    activation = config["activation"] if type(config["activation"]) != str else eval(config["activation"])
    full_model_net = Face2EnvNetwork(1, 1, decoder_state_dict=decoder_state_dict, bottleneck_size=bottleneck_size, activation=activation, initial_feature_maps=initial_feature_maps, train_decoder=False)
    load_state_dict(full_model_net.face2latent, data["model_state_dict"])
    return full_model_net

def get_env2env_autoencoder(autoencoder_filepath) -> Env2EnvAutoEncoder:
    model = _load_autoencoder(autoencoder_filepath)
    return model

def get_env2env_autoencoder_state_dict(autoencoder_filepath) -> Mapping[str, Any]:
    model = _load_autoencoder(autoencoder_filepath)
    return model.state_dict()

def _load_autoencoder(autoencoder_filepath) -> Env2EnvAutoEncoder:
    data = torch.load(autoencoder_filepath)
    config = data["config"]
    bottleneck_size =config["bottleneck_size"]
    activation = config["activation"] if type(config["activation"]) != str else eval(config["activation"])
    relu_output = config["relu_output"]
    model = Env2EnvAutoEncoder(in_channels=1, bottleneck_size=bottleneck_size, activation=activation, relu_output=relu_output)
    load_state_dict(model, data["model_state_dict"])
    return model  

#https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/2
def optimizer_to_device(optim: torch.optim.Optimizer, device: torch.device) -> None:
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def save_model(epoch: int|str, model: ModelWrapper, config, only_torch = False) -> None:
    filename = get_model_filename(model)
    config_to_save = config if type(config) == dict else config.as_dict()
    torch.save({'epoch': epoch,
            'model_state_dict': model.net.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'config': config_to_save}, filename
	    )
    if only_torch:
        return
    upload_model(config= config_to_save, filepath=filename)
    
def get_model_filename(model: ModelWrapper) -> str:
    folder = Config.MODEL_OUTPUT_DIR
    run_id = wandb.run.id if wandb.run else datetime.now().strftime("no_wandb_%Y-%m-%d_%H-%M-%S")
    folder = os.path.join(folder, run_id)
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = f'{folder}/{model.net.__class__.__name__}_best.pth'
    return filename


if __name__ == "__main__":
    path = r"D:\Masterarbeit\lightEstimation\models\face2latent\face2latent.pth"
    get_face2env_network(path)