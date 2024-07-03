from typing_extensions import deprecated
import wandb
import os
import logging
from wandb.apis.public import Run
from wandb import Artifact

from lightestimation.models.model import ModelWrapper
from lightestimation.utils.config import Config

logger = logging.getLogger(__name__)

def download_model_from_run(run:Run) -> str:
    """Returns the path to the model downloaded from the run.
    If the model is already downloaded, it will not download it again.
    If multiple models are found in the run, it will return the first one.

    Args:
        run (Run): Wandb run object

    Raises:
        ValueError: If no model is found in the run

    Returns:
        str: Local path to the model
    """
    artefacts = run.logged_artifacts()
    for artefact in artefacts:
        if artefact.type == "model":
            return _download_artefact(artefact, run.id)
    raise ValueError(f"No model found in run {run}")

def get_model_from_model_artefact_path(artefact_path: str) -> str:
    """Returns the path to a model that was logged as an artefact.

    Args:
        artefact_path (str): Wandb artefact path

    Raises:
        ValueError: If the artefact is not logged by any run

    Returns:
        str: Local path to the model
    """
    api = wandb.Api()
    artefact: Artifact = api.artifact(artefact_path)
    run = artefact.logged_by()
    if run is None:
        raise ValueError(f"Artefact {artefact_path} not logged by any run")
    return _download_artefact(artefact, run.id)

def _download_artefact(artefact: Artifact, run_id: str) -> str:
    download_dir = os.path.join(Config.WANDB_RUN_DIR, "models", run_id)
    filename = os.path.join(download_dir, "Model_best.pth")
    if os.path.exists(filename):
        logger.info(f"Model already downloaded to {filename}")
        return filename
    else:
        artefact_dir = artefact.download(download_dir)
        artefact.download(download_dir)
    return os.path.join(artefact_dir, "Model_best.pth")

@deprecated("Directly use download_model_from_run instead of this function")
def get_best_encoder_path(bottleneck_size: int, activation_fn_name: str|None = None) -> str:
    api = wandb.Api()
    if activation_fn_name is not None:
        runs = api.runs(path=Config.WANDB_PROJECT_PATH, filters={"config.bottleneck_size": bottleneck_size, "config.activation": activation_fn_name}, order="+summary_metrics.best_accuracy")
    else:
        runs = api.runs(path=Config.WANDB_PROJECT_PATH, filters={"config.bottleneck_size": bottleneck_size}, order="+summary_metrics.best_accuracy")
    try:
        run = runs[0]
    except IndexError:
        logger.warning(f"No runs found with bottleneck size {bottleneck_size} and activation {activation_fn_name}, trying with random activation")
        runs = api.runs(path=Config.WANDB_PROJECT_PATH, filters={"config.bottleneck_size": bottleneck_size}, order="+summary_metrics.best_accuracy")
        run = runs[0]
    return download_model_from_run(run)

def get_config_and_modelname_from_run(run_name: str)-> tuple[dict, str]:
    """Returns the config and model from a run name

    Args:
        run_name (str): Name of the wandb run

    Returns:
        tuple[dict, str]: Wandb config and local model path
    """
    api = wandb.Api()
    path = Config.WANDB_PROJECT_PATH+run_name
    run = api.run(path)
    config = run.config
    model = download_model_from_run(run)
    return config, model

def log(acc, metrics, images = None, split = "val", existing_dict = None) -> None:
    if existing_dict:
        wandb_dict = existing_dict
    else:
        wandb_dict = {}
    if split not in wandb_dict:
        wandb_dict[split] = {}
    wandb_dict[split]["epoch_accuracy"] = acc
    for key, value in metrics.items():
        wandb_dict[split][key] = value
    if images:
        wandb_dict[split]["output"] = images
    wandb.log(wandb_dict)

def log_train(model: ModelWrapper, epoch_loss, epoch, val_acc, val_metrics, val_images = None) -> None:
    split = "train"
    wandb_dict = {split: {"epoch_loss":epoch_loss, "step": epoch}}
    ds = model.optimizer.param_groups[0]['d']
    dlrs = model.optimizer.param_groups[0]['lr'] * ds
    
    wandb_dict[split]["lr"] = dlrs
    wandb_dict[split]["d"] = ds
    
    log(val_acc, val_metrics, val_images, "val", existing_dict = wandb_dict)

def add_summary(dict: dict) -> None:
    if wandb.run is not None:
        for key, value in dict.items():
            wandb.run.summary[key] = value

def upload_model(config: dict, filepath: str) -> None:
    """Upload a model to wandb

    Args:
        config (dict): configuration file
        filepath (str): local filepath of the model
    """
    metadata = {"name": config["model"]}
    model_artifact = wandb.Artifact(
            config["model"], type="model",
            description="Trained NN model",
            metadata=metadata
            )
    model_artifact.add_file(filepath)
    logger.info(f"Saving model to {filepath}")
    wandb.log_artifact(model_artifact)