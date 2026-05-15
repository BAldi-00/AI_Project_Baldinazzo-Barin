from torchvision import transforms


def get_train_transforms(image_size=48, brightness_factor=0.2, rotation_degrees=10):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(rotation_degrees),
        transforms.ColorJitter(brightness=brightness_factor),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_eval_transforms(image_size=48):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
