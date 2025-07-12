import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm  # Для красивого прогресс-бара

# Конфигурация
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
DATA_PATH = "data"  # Папка с данными
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Архитектура модели
class AgeGenderResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Головы для multi-task
        self.age_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Softplus()  # Возраст > 0
        )
        
        self.gender_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        age = self.age_head(features)
        gender = self.gender_head(features)
        return age, gender

# 2. Датасет с обработкой имен файлов
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') and f.count('_') == 3]
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        
        # Парсинг имени файла
        age, gender, race, _ = self.image_files[idx].split('_')
        return self.transform(img), torch.tensor(float(age)), torch.tensor(float(gender))

# 3. Обучение с валидацией
def train():
    # Инициализация
    model = AgeGenderResNet().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    criterion_age = nn.L1Loss()
    criterion_gender = nn.BCELoss()

    # Создание датасета
    full_dataset = UTKFaceDataset(DATA_PATH)
    
    # Разделение на train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Всего изображений: {len(full_dataset)}")
    print(f"Обучающая выборка: {len(train_dataset)}")
    print(f"Валидационная выборка: {len(val_dataset)}")

    # Цикл обучения
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        # Обучение
        for images, ages, genders in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, ages, genders = images.to(DEVICE), ages.to(DEVICE), genders.to(DEVICE)
            
            optimizer.zero_grad()
            age_pred, gender_pred = model(images)
            
            loss_age = criterion_age(age_pred, ages.view(-1, 1))
            loss_gender = criterion_gender(gender_pred, genders.view(-1, 1))
            loss = loss_age + loss_gender
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Валидация
        model.eval()
        val_loss = 0
        age_mae = 0
        gender_acc = 0
        
        with torch.no_grad():
            for images, ages, genders in val_loader:
                images, ages, genders = images.to(DEVICE), ages.to(DEVICE), genders.to(DEVICE)
                
                age_pred, gender_pred = model(images)
                
                val_loss += criterion_age(age_pred, ages.view(-1, 1)).item()
                val_loss += criterion_gender(gender_pred, genders.view(-1, 1)).item()
                
                age_mae += torch.abs(age_pred - ages.view(-1, 1)).sum().item()
                gender_acc += ((gender_pred > 0.5).float() == genders.view(-1, 1)).sum().item()
        
        # Статистика
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        age_mae /= len(val_dataset)
        gender_acc /= len(val_dataset)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Age MAE: {age_mae:.2f} | Gender Acc: {gender_acc:.2%}")
        
        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Модель сохранена!")

if __name__ == "__main__":
    train()