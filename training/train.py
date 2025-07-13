import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import yaml
from datetime import datetime
from tqdm import tqdm

# Наши модули
from models.efficient_net import EfficientAgeGender
from training.data_loader import create_loaders
from training.callbacks import EarlyStopping, ModelCheckpoint

def train(config):
    # 1. Настройка устройства и семян
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # 2. Создание папок и логгера
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=os.path.join(config['log_dir'], current_time))
    
    # 3. Загрузка данных
    train_loader, val_loader = create_loaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size']
    )
    
    # 4. Инициализация модели
    model = EfficientAgeGender().to(device)
    
    # 5. Оптимизатор и планировщик
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # 6. Функции потерь
    age_criterion = nn.L1Loss()
    gender_criterion = nn.BCELoss()
    
    # 7. Коллбэки
    early_stopping = EarlyStopping(
        patience=config['patience'],
        delta=0.001,
        verbose=True
    )
    
    checkpoint = ModelCheckpoint(
        save_dir=config['checkpoint_dir'],
        save_freq=config['checkpoint_freq']
    )
    
    # 8. Цикл обучения
    for epoch in range(config['epochs']):
        model.train()
        train_age_loss, train_gender_loss = 0, 0
        correct_gender, total_samples = 0, 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        
        for images, (age_labels, gender_labels) in progress_bar:
            images = images.to(device)
            age_labels = age_labels.float().to(device)
            gender_labels = gender_labels.float().to(device)
            
            # Forward pass
            age_pred, gender_pred = model(images)
            
            # Вычисление потерь
            age_loss = age_criterion(age_pred.squeeze(), age_labels)
            gender_loss = gender_criterion(gender_pred.squeeze(), gender_labels)
            total_loss = age_loss * config['age_weight'] + gender_loss * config['gender_weight']
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Статистика
            train_age_loss += age_loss.item()
            train_gender_loss += gender_loss.item()
            
            # Точность определения пола
            predicted_genders = (gender_pred.squeeze() > 0.5).float()
            correct_gender += (predicted_genders == gender_labels).sum().item()
            total_samples += gender_labels.size(0)
            
            # Обновление progress bar
            progress_bar.set_postfix({
                'Age Loss': f'{age_loss.item():.4f}',
                'Gender Loss': f'{gender_loss.item():.4f}',
                'Gender Acc': f'{correct_gender/total_samples:.2%}'
            })
        
        # 9. Валидация
        val_age_loss, val_gender_loss, val_gender_acc = validate(
            model, val_loader, age_criterion, gender_criterion, device
        )
        
        # 10. Логирование
        avg_train_age_loss = train_age_loss / len(train_loader)
        avg_train_gender_loss = train_gender_loss / len(train_loader)
        train_gender_acc = correct_gender / total_samples
        
        writer.add_scalars('Loss/Age', {
            'train': avg_train_age_loss,
            'val': val_age_loss
        }, epoch)
        
        writer.add_scalars('Loss/Gender', {
            'train': avg_train_gender_loss,
            'val': val_gender_loss
        }, epoch)
        
        writer.add_scalars('Accuracy/Gender', {
            'train': train_gender_acc,
            'val': val_gender_acc
        }, epoch)
        
        # 11. Обновление планировщика и коллбэков
        scheduler.step(val_age_loss + val_gender_loss)
        
        early_stopping(val_age_loss + val_gender_loss, model)
        checkpoint(epoch, model, optimizer, val_age_loss + val_gender_loss)
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    writer.close()
    return model

def validate(model, val_loader, age_criterion, gender_criterion, device):
    model.eval()
    age_loss, gender_loss = 0, 0
    correct_gender, total_samples = 0, 0
    
    with torch.no_grad():
        for images, (age_labels, gender_labels) in val_loader:
            images = images.to(device)
            age_labels = age_labels.float().to(device)
            gender_labels = gender_labels.float().to(device)
            
            age_pred, gender_pred = model(images)
            
            age_loss += age_criterion(age_pred.squeeze(), age_labels).item()
            gender_loss += gender_criterion(gender_pred.squeeze(), gender_labels).item()
            
            predicted_genders = (gender_pred.squeeze() > 0.5).float()
            correct_gender += (predicted_genders == gender_labels).sum().item()
            total_samples += gender_labels.size(0)
    
    return (
        age_loss / len(val_loader),
        gender_loss / len(val_loader),
        correct_gender / total_samples
    )

if __name__ == '__main__':
    # Загрузка конфигурации
    with open('configs/train_config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Запуск обучения
    trained_model = train(config)
    
    # Сохранение финальной модели
    final_model_path = os.path.join(config['checkpoint_dir'], 'final_model.pth')
    torch.save(trained_model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')