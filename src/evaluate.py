import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import yaml
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from src.dataset import EmotionDataset
from src.preprocessing import get_eval_transforms
from src.model import EmotionCNN


def load_config(path=None):
    if path is None:
        path = os.path.join(ROOT_DIR, 'config.yaml')
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def evaluate():
    cfg = load_config()
    device_str = cfg['inference']['device']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_str == 'auto' else torch.device(device_str)

    transform = get_eval_transforms(image_size=cfg['dataset']['image_size'])
    test_ds = EmotionDataset(os.path.join(ROOT_DIR, cfg['dataset']['test_dir']), transform=transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
    )

    model = EmotionCNN(num_classes=cfg['dataset']['num_classes']).to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT_DIR, cfg['training']['model_save_path']), map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    print(f"Test Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=cfg['dataset']['classes']))

    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)


if __name__ == '__main__':
    evaluate()
