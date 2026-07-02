import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir="../data/chest_xray",
                    batch_size=32,
                    image_size=224,
                    num_workers=2):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    phases = ["train", "val", "test"]

    # 評価・テスト用前処理
    eval_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 訓練用前処理（データ拡張含む）
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    data_transforms = {
        'train': train_transforms,
        'val': eval_transforms,
        'test': eval_transforms
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
        for x in phases
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, 
                      shuffle=(x == 'train'), num_workers=num_workers) 
        for x in phases
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in phases}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names, mean, std