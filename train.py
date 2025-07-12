import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter

# Конфигурация
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPOCHS = 15
DATA_PATH = "data"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Улучшенная архитектура модели
class AgeGenderRaceResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Общие признаки
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5)
        )
        
        # Головы
        self.age_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Softplus()
        )
        
        self.gender_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.race_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_fc(features)
        
        age = self.age_head(shared)
        gender = self.gender_head(shared)
        race = self.race_head(shared)
        
        return age, gender, race

# 2. Датасет с улучшенной аугментацией
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.image_files = [f for f in os.listdir(root_dir) 
                          if f.endswith('.jpg') and f.count('_') == 3]
        self.root_dir = root_dir
        
        # Разные трансформы для train и val
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(300),
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        
        # Парсинг метаданных
        age, gender, race, _ = self.image_files[idx].split('_')
        return (
            self.transform(img),
            torch.tensor(float(age)),
            torch.tensor(float(gender)),
            torch.tensor(int(race))
        )

# 3. Функция анализа распределения данных
def analyze_dataset(dataset):
    ages = []
    genders = []
    races = []
    
    for _, age, gender, race in dataset:
        ages.append(age.item())
        genders.append(gender.item())
        races.append(race.item())
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.hist(ages, bins=30)
    plt.title('Age Distribution')
    
    plt.subplot(132)
    plt.hist(genders, bins=2)
    plt.title('Gender Distribution')
    
    plt.subplot(133)
    plt.hist(races, bins=5)
    plt.title('Race Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'age_mean': np.mean(ages),
        'gender_ratio': Counter(genders),
        'race_dist': Counter(races)
    }

# 4. Обучение с улучшениями
def train():
    # Инициализация
    full_dataset = UTKFaceDataset(DATA_PATH, mode='train')
    stats = analyze_dataset(full_dataset)
    print(f"Dataset stats:\n{stats}")
    
    # Разделение данных
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Изменяем трансформы для валидации
    val_dataset.dataset.transform = UTKFaceDataset(DATA_PATH, mode='val').transform
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)
    
    # Инициализация модели
    model = AgeGenderRaceResNet().to(DEVICE)
    
    # Взвешенные лосс-функции
    gender_weights = torch.tensor([1.0, 1.2]).to(DEVICE)  # Подправьте под ваши данные
    race_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).to(DEVICE)  # Подправьте
    
    criterion_age = nn.L1Loss()
    criterion_gender = nn.BCEWithLogitsLoss(pos_weight=gender_weights)
    criterion_race = nn.CrossEntropyLoss(weight=race_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Тренировочный цикл
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        age_mae, gender_acc, race_acc = 0, 0, 0
        
        for images, ages, genders, races in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(DEVICE)
            ages = ages.to(DEVICE).view(-1, 1)
            genders = genders.to(DEVICE).float()  # One-hot
            races = races.to(DEVICE)
            
            optimizer.zero_grad()
            age_pred, gender_pred, race_pred = model(images)
            
            loss_age = criterion_age(age_pred, ages)
            loss_gender = criterion_gender(gender_pred, genders.argmax(1))
            loss_race = criterion_race(race_pred, races)
            
            loss = loss_age + loss_gender + 0.5*loss_race
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            age_mae += torch.abs(age_pred - ages).sum().item()
            gender_acc += (gender_pred.argmax(1) == genders.argmax(1)).sum().item()
            race_acc += (race_pred.argmax(1) == races).sum().item()
        
        # Валидация
        model.eval()
        val_loss = 0
        val_age_mae, val_gender_acc, val_race_acc = 0, 0, 0
        
        with torch.no_grad():
            for images, ages, genders, races in val_loader:
                images = images.to(DEVICE)
                ages = ages.to(DEVICE).view(-1, 1)
                genders = genders.to(DEVICE).view(-1, 1)
                races = races.to(DEVICE)
                
                age_pred, gender_pred, race_pred = model(images)
                
                val_loss += (criterion_age(age_pred, ages) + 
                            criterion_gender(gender_pred, genders) + 
                            0.5*criterion_race(race_pred, races)).item()
                
                val_age_mae += torch.abs(age_pred - ages).sum().item()
                val_gender_acc += ((torch.sigmoid(gender_pred) > 0.5).float() == genders).sum().item()
                val_race_acc += (race_pred.argmax(1) == races).sum().item()
        
        # Статистика
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        age_mae /= len(train_dataset)
        gender_acc /= len(train_dataset)
        race_acc /= len(train_dataset)
        
        val_age_mae /= len(val_dataset)
        val_gender_acc /= len(val_dataset)
        val_race_acc /= len(val_dataset)
        
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Age MAE: Train {age_mae:.2f} | Val {val_age_mae:.2f}")
        print(f"Gender Acc: Train {gender_acc:.2%} | Val {val_gender_acc:.2%}")
        print(f"Race Acc: Train {race_acc:.2%} | Val {val_race_acc:.2%}")
        
        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_improved.pth")
            print("Model saved!")

# 5. Тестирование на своих изображениях
def test_image(image_path, model_path="best_model_improved.pth"):
    model = AgeGenderRaceResNet(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        age, gender, race = model(img_tensor)
    
    race_labels = ['White', 'Black', 'Asian', 'Indian', 'Others']
    
    plt.imshow(img)
    plt.title(f"Age: {age.item():.1f}\nGender: {'Male' if gender.item() > 0.5 else 'Female'}\n"
             f"Race: {race_labels[race.argmax().item()]}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    train()
    
    # После обучения можно тестировать:
    test_image("лицо.jpg")