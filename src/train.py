import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import EmotionDataset
from src.preprocessing import get_train_transforms, get_eval_transforms
from src.model import EmotionCNN


def load_config(path=None):
    if path is None:
        path = os.path.join(ROOT_DIR, 'config.yaml')
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_device(cfg):
    if cfg['inference']['device'] == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(cfg['inference']['device'])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)

    return total_loss / total, correct / total


def train():
    cfg = load_config()
    device = get_device(cfg)

    train_transform = get_train_transforms(
        image_size=cfg['dataset']['image_size'],
        brightness_factor=cfg['augmentation']['brightness_factor'],
        rotation_degrees=cfg['augmentation']['rotation_degrees'],
    )
    eval_transform = get_eval_transforms(image_size=cfg['dataset']['image_size'])

    train_ds = EmotionDataset(os.path.join(ROOT_DIR, cfg['dataset']['train_dir']), transform=train_transform)
    test_ds = EmotionDataset(os.path.join(ROOT_DIR, cfg['dataset']['test_dir']), transform=eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
    )
    val_loader = DataLoader(
        test_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
    )

    model = EmotionCNN(num_classes=cfg['dataset']['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])

    save_path = os.path.join(ROOT_DIR, cfg['training']['model_save_path'])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_acc = 0.0
    patience = cfg['training']['early_stopping_patience']
    no_improve = 0

    for epoch in range(1, cfg['training']['epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:03d} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best val_acc={best_val_acc:.4f}")
                break

    print(f"Training complete. Best val_acc={best_val_acc:.4f}. Model saved to {save_path}")


if __name__ == '__main__':
    train()
