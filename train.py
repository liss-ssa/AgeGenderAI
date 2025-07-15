import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.mobilenetv3 import AgeGenderModel
from utils.dataloader import UTKFaceDataset
from utils.transforms import train_transform, val_transform
import argparse
import os
from datetime import datetime

# --- Конфигурация ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--data_dir', type=str, default='UTKFace/part1')  # Путь к исходным данным
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--save_dir', type=str, default='saved_models')
    parser.add_argument('--val_split', type=float, default=0.2)  # Доля данных для валидации
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Создание директорий ---
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # --- Датасеты ---
    print(f"\nЗагрузка данных из: {args.data_dir}")
    full_dataset = UTKFaceDataset(args.data_dir, transform=train_transform)
    
    # Разделение на train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print("\n--- Проверка первых 5 файлов ---")
    for i in range(min(5, len(full_dataset))):
        img, labels = full_dataset[i]
        print(f"Файл: {full_dataset.image_files[i]} | Возраст: {labels['age'].item()} | Пол: {labels['gender'].item()}")
    
    # Применяем разные трансформации
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # --- Загрузчики ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    # --- Проверка данных ---
    print("\n--- Проверка данных ---")
    print(f"Всего изображений: {len(full_dataset)}")
    print(f"Тренировочные: {len(train_dataset)}")
    print(f"Валидационные: {len(val_dataset)}")
    
    sample_batch = next(iter(train_loader))
    images, labels = sample_batch
    print(f"\nРазмер батча: {images.shape}")  # [batch_size, 3, 224, 224]
    print(f"Пример возраста: {labels['age'][:5].tolist()}")
    print(f"Пример пола: {labels['gender'][:5].tolist()}\n")
    
    # --- Модель ---
    model = AgeGenderModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
    
    # --- Лосс-функции ---
    age_criterion = nn.L1Loss()
    gender_criterion = nn.BCEWithLogitsLoss()
    
    # --- Логирование (TensorBoard) ---
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(os.path.join(args.log_dir, current_time))
    
    # --- Обучение ---
    print("--- Начало обучения ---")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            age_labels = targets['age'].to(device, non_blocking=True)
            gender_labels = targets['gender'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            age_pred, gender_pred = model(images)
            
            age_loss = age_criterion(age_pred.squeeze(), age_labels)
            gender_loss = gender_criterion(gender_pred.squeeze(), gender_labels)
            total_loss = age_loss + gender_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
            # Логирование каждые 50 батчей
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {total_loss.item():.4f}")
        
        # --- Валидация ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                age_labels = targets['age'].to(device, non_blocking=True)
                gender_labels = targets['gender'].to(device, non_blocking=True)
                
                age_pred, gender_pred = model(images)
                age_loss = age_criterion(age_pred.squeeze(), age_labels)
                gender_loss = gender_criterion(gender_pred.squeeze(), gender_labels)
                val_loss += (age_loss + gender_loss).item()
        
        # --- Статистика эпохи ---
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\nEpoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(val_loss)
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Новая лучшая модель сохранена (Loss: {best_val_loss:.4f})")
    
    # --- Финализация ---
    writer.close()
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
    }, os.path.join(args.save_dir, 'last_model.pth'))
    
    print(f"\nОбучение завершено. Лучшая валидационная loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()