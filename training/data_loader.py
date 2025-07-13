import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class AgeGenderDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        
        # Парсим метаданные из имен файлов (формат: age_gender_race_date.jpg)
        self.metadata = []
        for img_name in self.image_files:
            try:
                age, gender, *_ = img_name.split('_')
                self.metadata.append({
                    'age': float(age),
                    'gender': float(gender),  # 0-female, 1-male
                    'image_path': os.path.join(data_dir, img_name)
                })
            except:
                continue
        
        # Автоматическое разделение на train/val если нет готового
        if mode in ['train', 'val']:
            train_meta, val_meta = train_test_split(
                self.metadata, 
                test_size=0.2, 
                random_state=42,
                stratify=[m['gender'] for m in self.metadata]
            )
            self.metadata = train_meta if mode == 'train' else val_meta

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        image = cv2.cvtColor(cv2.imread(meta['image_path']), cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, (meta['age'], meta['gender'])

def create_loaders(data_dir, batch_size=32):
    # Аугментации для тренировочных данных
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Трансформы для валидации
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = AgeGenderDataset(data_dir, transform=train_transform, mode='train')
    val_dataset = AgeGenderDataset(data_dir, transform=val_transform, mode='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader