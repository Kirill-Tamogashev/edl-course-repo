from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, resnet101, ResNet18_Weights, ResNet101_Weights
import hydra


def setup_model(cfg):
    if cfg.model == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif cfg.model == 'resnet101':
        model = resnet101(weights=ResNet101_Weights.DEFAULT)
    else:
        raise ValueError("Unknown model")

    model.fc = nn.Linear(model.fc.in_features, cfg.train.num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
    model.maxpool = nn.Identity()

    if cfg.load_checkpoint is not None:
        model.load_state_dict(torch.load(cfg.load_checkpoint))

    return model.to(cfg.device)


def compute_accuracy(pred_probs, labels):
    pred_labels = torch.argmax(pred_probs, dim=1)
    acc = pred_labels.eq(labels).float().mean()
    return acc


def train_epoch(model, loader, loss_fn, optimizer, epoch, cfg):
    epoch_acc = 0
    pbar = tqdm(loader, leave=False)
    for batch in pbar:
        x, y = (elem.to(cfg.device) for elem in batch)

        optimizer.zero_grad()
        probas = F.softmax(model(x), dim=1)

        loss = loss_fn(probas, y)
        loss.backward()
        optimizer.step()

        acc = compute_accuracy(probas, y)
        epoch_acc += acc
        pbar.set_description(
            desc=f'Train epoch: {epoch + 1}, loss: {loss.item():.4f}, acc: {acc:.3f}'
        )

    return epoch_acc / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, epoch, cfg):
    epoch_acc = 0
    pbar = tqdm(loader, leave=False)
    for batch in pbar:
        x, y = (elem.to(cfg.device) for elem in batch)
        probas = F.softmax(model(x), dim=1)

        acc = compute_accuracy(probas, y)
        epoch_acc += acc

        pbar.set_description(
            desc=f'Val epoch: {epoch + 1}, acc: {acc:.3f}'
        )

    return epoch_acc / len(loader)


@hydra.main(config_path="configs", config_name="finetune", version_base=None)
def finetune(cfg):
    model = setup_model(cfg)
    loss_fn = nn.CrossEntropyLoss()

    train_data = CIFAR10(root="./data", download=True, train=True, transform=T.ToTensor())
    train_loader = DataLoader(train_data, **cfg.loader.train)

    val_data = CIFAR10(root="./data,", download=True, train=False, transform=T.ToTensor())
    val_loader = DataLoader(val_data, **cfg.loader.val)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    acc_prev = 0
    prev_was_good = False
    save_dir = Path(cfg.train.save_dir) / cfg.model
    save_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(cfg.train.num_epochs):
        print(f'Epoch {epoch + 1}/{cfg.train.num_epochs}')
        model.train()
        train_acc = train_epoch(model, train_loader, loss_fn, optimizer, epoch, cfg)
        print(f'Train accuracy {train_acc:.3f}')

        model.eval()
        val_acc = eval_epoch(model, val_loader, epoch, cfg)
        print(f'Val accuracy {val_acc:.3f}\n')

        if torch.abs(val_acc - acc_prev) < 0.01:
            if prev_was_good:
                torch.save(model.state_dict(), save_dir / f"epoch_{epoch}.pth")
                print("Finish training")
                break
        else:
            prev_was_good = False

        if epoch % cfg.train.save_freq == 0:
            torch.save(model.state_dict(), save_dir / f"epoch_{epoch}.pth")


if __name__ == '__main__':
    finetune()
