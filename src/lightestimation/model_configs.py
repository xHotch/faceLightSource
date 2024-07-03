
from lightestimation.data.dataset_utils import get_val_split
from lightestimation.utils.config import Config
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 # type: ignore
from lightestimation.data.dataset import NR_TRAIN_IMAGES, FaceToEnvironmentDataset, EnvironmentToEnvironmentDataset
from lightestimation.evaluation.loss import *
from lightestimation.models.brightness_estimation import Face2EnvNetwork, Env2EnvAutoEncoder, FaceEncoder

from lightestimation.models.model import ModelWrapper

def get_face2env_test_loader(batch_size:int = 1, visualized:bool=False, subset_size:int|None=None, only_faces:bool=False, output_cubemap_id:bool=False):
    transform = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1),
        v2.ConvertImageDtype(torch.float32),
    ])
    
    label_transform = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1),
        v2.ConvertImageDtype(torch.float32),
    ])
    
    test_dataset = FaceToEnvironmentDataset(Config.TEST_LIGHTPROBE_PATH, transform=transform, target_transform=label_transform, random_transform=None, visualized=visualized, subset_size=subset_size, only_faces= only_faces, lazy_load=True, output_cubemap_id = output_cubemap_id)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def _get_train_transforms(to_brightness: bool = True):
    transform = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1) if to_brightness else nn.Identity(),
        v2.ConvertImageDtype(torch.float32),
    ])
    
    label_transform = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1) if to_brightness else nn.Identity(),
        v2.ConvertImageDtype(torch.float32),
    ])
    
    random_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
    ])
    
    return transform, label_transform, random_transforms

def get_model_config(config, visualized:bool=False, subset_size:int|None=None, lazy_load:bool=False, *args, **kwargs) -> tuple[ModelWrapper, Dataset, Dataset, Dataset]:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    
    name = config["model"]
    lr = config["initial_learning_rate"]
    activation = config["activation"] if type(config["activation"]) != str else eval(config["activation"])
    loss_fn = config["loss_fn"] if type(config["loss_fn"]) != str else eval(config["loss_fn"])
    relu_output = config.get("relu_output", False)
    initial_feature_maps = config.get("initial_feature_maps", 16)
    ae_bottleneck_size = config.get("bottleneck_size", 256)
    train_decoder = config.get("train_decoder", False)
    only_faces = config.get("only_faces", False)

    val_envmaps = sorted(get_val_split())
    train_envmaps = [i for i in range(NR_TRAIN_IMAGES) if i not in val_envmaps]
          
    test_dataset_path = Config.TEST_LIGHTPROBE_PATH
    dataset_path = Config.TRAIN_LIGHTPROBE_PATH
    
    if name == "Face2EnvNetwork":
        transform, label_transform, random_transforms = _get_train_transforms(to_brightness=True)        
        decoder_weights = kwargs.get("decoder_weights", None)
        model = ModelWrapper(Face2EnvNetwork(1, 1, decoder_weights, bottleneck_size=ae_bottleneck_size, activation=activation, initial_feature_maps=initial_feature_maps, train_decoder=train_decoder), loss_fn=loss_fn, lr = lr, config = config)
        
        train_dataset = FaceToEnvironmentDataset(dataset_directory = dataset_path, transform=transform, target_transform=label_transform, random_transform=random_transforms, visualized=visualized, subset_size=subset_size, only_faces= only_faces, envmap_subset=train_envmaps, lazy_load = lazy_load)        
        val_dataset = FaceToEnvironmentDataset(dataset_path, transform=transform, target_transform=label_transform, random_transform=random_transforms, visualized=visualized, subset_size=subset_size, only_faces= only_faces, envmap_subset=val_envmaps, lazy_load = lazy_load)
        test_dataset = FaceToEnvironmentDataset(test_dataset_path, transform=transform, target_transform=label_transform, random_transform=random_transforms, visualized=visualized, subset_size=subset_size, only_faces= only_faces, lazy_load=lazy_load)
    
    elif name == "FaceEncoder":
        transform, label_transform, random_transforms = _get_train_transforms(to_brightness=True)
        model = ModelWrapper(FaceEncoder(1, bottleneck_size=ae_bottleneck_size, activation=activation, initial_feature_maps=initial_feature_maps), loss_fn=loss_fn, lr = lr, config = config)
        
        train_dataset = FaceToEnvironmentDataset(dataset_directory = dataset_path, transform=transform, target_transform=label_transform, random_transform=random_transforms, visualized=visualized, subset_size=subset_size, only_faces= only_faces, envmap_subset=train_envmaps, lazy_load = lazy_load)        
        val_dataset = FaceToEnvironmentDataset(dataset_path, transform=transform, target_transform=label_transform, random_transform=random_transforms, visualized=visualized, subset_size=subset_size, only_faces= only_faces, envmap_subset=val_envmaps, lazy_load = lazy_load)
        test_dataset = FaceToEnvironmentDataset(test_dataset_path, transform=transform, target_transform=label_transform, random_transform=random_transforms, visualized=visualized, subset_size=subset_size, only_faces= only_faces, lazy_load=lazy_load)

    
    elif "AutoEncoder" in name:
        to_brightness = "Brightness" in name
        input_channels = 1 if to_brightness else 3
        transform, label_transform, random_transforms = _get_train_transforms(to_brightness=to_brightness)
                  
        model = ModelWrapper(Env2EnvAutoEncoder(input_channels, ae_bottleneck_size, activation=activation, relu_output=relu_output), loss_fn=loss_fn, lr = lr, config = config)
    
        train_dataset = EnvironmentToEnvironmentDataset(dataset_path, transform=transform, target_transform=label_transform, random_transform=random_transforms, visualized=visualized, subset_size=subset_size, envmap_subset=train_envmaps, lazy_load = lazy_load)
        val_dataset = EnvironmentToEnvironmentDataset(dataset_path, transform=transform, target_transform=label_transform, random_transform=random_transforms, visualized=visualized, subset_size=subset_size, envmap_subset=val_envmaps, lazy_load = lazy_load)
        test_dataset = EnvironmentToEnvironmentDataset(test_dataset_path, transform=transform, target_transform=label_transform, random_transform=random_transforms, visualized=visualized, subset_size=subset_size, lazy_load = lazy_load)
    
    else:
        raise ValueError(f"Model config {name} not found")
    
    return model, train_dataset, val_dataset, test_dataset