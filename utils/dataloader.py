import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        try:
            parts = img_name.split('_')
            
            if len(parts) >= 4:  # Формат: age_gender_race_date.jpg
                age, gender, _, _ = parts
            elif len(parts) == 3:  # Формат: age_gender_race.jpg
                age, gender, _ = parts
            elif len(parts) == 2:  # Формат: age_gender.jpg
                age, gender = parts
            else:
                raise ValueError(f"Неизвестный формат имени файла: {img_name}")
                
            age = int(age)
            gender = int(gender)
            
            # Нормализация пола к [0, 1]
            if gender > 1:
                gender = 1
            elif gender < 0:
                gender = 0
                
        except Exception as e:
            print(f"Ошибка при обработке файла {img_name}: {e}")
            age = 30
            gender = 0
        
        if self.transform:
            img = self.transform(img)
            
        return img, {"age": torch.tensor(age, dtype=torch.float32),
                    "gender": torch.tensor(gender, dtype=torch.float32)}
