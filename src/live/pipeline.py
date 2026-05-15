import yaml
import torch

from src.inference import load_model
from src.preprocessing import get_eval_transforms
from src.utils.queue_manager import QueueManager
from src.live.camera import CameraThread
from src.live.worker import InferenceWorker


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_pipeline(config_path):
    cfg = load_config(config_path)

    device_str = cfg['inference']['device']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_str == 'auto' else torch.device(device_str)

    transform = get_eval_transforms(image_size=cfg['dataset']['image_size'])
    model = load_model(cfg, device)
    classes = cfg['dataset']['classes']

    qm = QueueManager(maxsize=cfg['live']['queue_maxsize'])

    cam_thread = CameraThread(
        queue_manager=qm,
        camera_index=cfg['live']['camera_index'],
    )
    inference_thread = InferenceWorker(
        queue_manager=qm,
        model=model,
        transform=transform,
        classes=classes,
        device=device,
        face_scale_factor=cfg['live']['face_scale_factor'],
        face_min_neighbors=cfg['live']['face_min_neighbors'],
    )

    return cam_thread, inference_thread, qm, cfg
