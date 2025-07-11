import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import os

class UTKFaceDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['filename'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        age = torch.tensor(row['age'], dtype=torch.float32)
        gender = torch.tensor(row['gender'], dtype=torch.long)
        
        return image, {'age': age, 'gender': gender}