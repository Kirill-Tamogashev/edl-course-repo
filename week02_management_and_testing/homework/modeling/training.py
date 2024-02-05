from pathlib import Path

import torch
import torchvision.utils
import torchvision.transforms as T
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import wandb

from omegaconf import OmegaConf, DictConfig

from modeling.diffusion import DiffusionModel
from modeling.unet import UnetModel


def train_step(model: DiffusionModel, inputs: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    loss = model(inputs)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: DiffusionModel, dataloader: DataLoader, optimizer: Optimizer, device: str, logger):
    model.train()

    pbar = tqdm(dataloader, leave=False)
    loss_ema = None

    for x, _ in pbar:
        train_loss = train_step(model, x, optimizer, device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss

        pbar.set_description(f"loss: {loss_ema:.4f}")
        logger.log({"train/loss": loss_ema, "train/lr": optimizer.param_groups[0]["lr"]})


def generate_samples(model: DiffusionModel, device: str, path: str):
    model.eval()
    with torch.no_grad():
        samples = model.sample(8, (3, 32, 32), device=device)
        grid = make_grid(samples, nrow=4)
        save_image(grid, path)


def set_optimizer(cfg: DictConfig, model: DiffusionModel):
    params = model.parameters()
    if cfg.train.optimizer == "adamw":
        return torch.optim.AdamW(params, **cfg.train.optimizer_params)
    elif cfg.train.optimizer == "sgd":
        return torch.optim.SGD(params, **cfg.train.optimizer_params)
    elif cfg.train.optimizer == "adam":
        return torch.optim.Adam(params, **cfg.train.optimizer_params)
    else:
        raise ValueError(f"Optimizer {cfg.train.optimizer} is unknown.")


def train_ddpm(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    unet = UnetModel(
        in_channels=cfg.ddpm.unet.in_channels,
        out_channels=cfg.ddpm.unet.out_channels,
        hidden_size=cfg.ddpm.unet.hidden_size)
    model = DiffusionModel(
        eps_model=unet,
        betas=cfg.ddpm.betas,
        num_timesteps=cfg.ddpm.num_timesteps,
    )
    optimizer = set_optimizer(cfg, model)

    img_transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.CIFAR10(cfg.data.path, train=True, transform=img_transform)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
    )

    with wandb.init(config=OmegaConf.to_container(cfg)) as logger:
        # log data artefact
        data_artifact = wandb.Artifact(name="Cifar10", type="dataset")
        data_artifact.add_dir(local_path="./cifar_data/cifar-10-python.tar.gz")
        logger.log_artifact(data_artifact)

        # log config artefact
        cfg_artifact = wandb.Artifact(name="Experiment Config", type="config")
        cfg_artifact.add_dir(local_path="./config/ddpm_config.yaml")
        logger.log_artifact(cfg_artifact)

        for epoch in range(cfg.train.num_epochs):
            train_epoch(model, train_dataloader, optimizer, cfg.device, logger)

            with torch.no_grad():
                model.eval()
                z, samples = model.sample(cfg.train.num_samples_to_log, (3, 32, 32), device=cfg.device)
                logger.log({
                    "eval/sample inputs": wandb.Image(torchvision.utils.make_grid(z, nrow=4, normalize=True)),
                    "eval/sample images": wandb.Image(torchvision.utils.make_grid(samples, normalize=True, nrow=4),
                                                      caption=f"Generated images, epoch: {epoch}")
                })

            if epoch % cfg.checkpoint.freq == 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }

                torch.save(checkpoint, Path(cfg.checkpoint.path) / "ddpm.pth")
