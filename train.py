import torch
from torch.utils.data import DataLoader
from model import ImprovedAgeGenderModel
from data_loader import UTKFaceDataset
from train_utils import Trainer
from torchvision import transforms
import pandas as pd

def main():
    # Конфигурация
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    learning_rate = 0.001
    epochs = 50
    
    # Аугментации
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Загрузка данных
    train_df = pd.read_csv('data/splits/train.csv')
    val_df = pd.read_csv('data/splits/val.csv')
    
    # Вычисление весов для классов пола
    gender_counts = train_df['gender'].value_counts()
    gender_weights = [1/gender_counts[0], 1/gender_counts[1]]  # Обратные частоты
    
    train_dataset = UTKFaceDataset('data/splits/train.csv', 'data/resized_224', train_transform)
    val_dataset = UTKFaceDataset('data/splits/val.csv', 'data/resized_224', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Инициализация модели
    model = ImprovedAgeGenderModel(gender_class_weights=gender_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Тренировка
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        patience=7
    )
    
    trained_model = trainer.train(epochs)

if __name__ == '__main__':
    main()
