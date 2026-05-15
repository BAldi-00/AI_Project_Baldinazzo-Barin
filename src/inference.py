import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import yaml
import torch
from PIL import Image

from src.preprocessing import get_eval_transforms
from src.model import EmotionCNN


def load_config(path=None):
    if path is None:
        path = os.path.join(ROOT_DIR, 'config.yaml')
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_model(cfg, device):
    model = EmotionCNN(num_classes=cfg['dataset']['num_classes']).to(device)
    model_path = os.path.join(ROOT_DIR, cfg['inference']['model_path'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_image(image, model, transform, classes, device):
    """
    Args:
        image: PIL.Image or numpy array (grayscale or BGR)
    Returns:
        (emotion_label: str, confidence: float, logits: Tensor)
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert('L')

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)

    return classes[pred_idx.item()], confidence.item(), logits


def main():
    import sys
    cfg = load_config()
    device_str = cfg['inference']['device']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_str == 'auto' else torch.device(device_str)
    transform = get_eval_transforms(image_size=cfg['dataset']['image_size'])
    model = load_model(cfg, device)
    classes = cfg['dataset']['classes']

    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if img_path is None:
        print("Usage: python inference.py <image_path>")
        return

    image = Image.open(img_path)
    label, confidence, _ = predict_image(image, model, transform, classes, device)
    print(f"Predicted emotion: {label} ({confidence * 100:.1f}%)")


if __name__ == '__main__':
    main()
