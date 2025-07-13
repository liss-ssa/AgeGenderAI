import torch.nn as nn
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, accuracy_score
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, 
                 model_dir='saved_models', patience=5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.best_val_loss = float('inf')
        self.counter = 0
        
        # Инициализация лоссов с весами
        self.age_criterion = nn.L1Loss()  # MAE для возраста
        self.gender_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(model.gender_class_weights[1] if model.gender_class_weights else 1.0))
        
        # Создание директории для моделей
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        
        # Инициализация TensorBoard
        log_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.writer = SummaryWriter(log_dir)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        age_loss_total = 0.0
        gender_loss_total = 0.0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            age_target = targets['age'].to(self.device)
            gender_target = targets['gender'].to(self.device).float()
            
            self.optimizer.zero_grad()
            
            age_pred, gender_pred = self.model(images)
            
            age_loss = self.age_criterion(age_pred.squeeze(), age_target)
            gender_loss = self.gender_criterion(gender_pred.squeeze(), gender_target)
            loss = age_loss + gender_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            age_loss_total += age_loss.item()
            gender_loss_total += gender_loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Train Batch: {batch_idx}/{len(self.train_loader)} Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(self.train_loader)
        avg_age_loss = age_loss_total / len(self.train_loader)
        avg_gender_loss = gender_loss_total / len(self.train_loader)
        
        return avg_loss, avg_age_loss, avg_gender_loss
    
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        age_preds = []
        age_targets = []
        gender_preds = []
        gender_targets = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                age_target = targets['age'].to(self.device)
                gender_target = targets['gender'].to(self.device).float()
                
                age_pred, gender_pred = self.model(images)
                
                age_loss = self.age_criterion(age_pred.squeeze(), age_target)
                gender_loss = self.gender_criterion(gender_pred.squeeze(), gender_target)
                loss = age_loss + gender_loss
                
                val_loss += loss.item()
                age_preds.extend(age_pred.cpu().numpy().flatten())
                age_targets.extend(age_target.cpu().numpy())
                gender_preds.extend(torch.sigmoid(gender_pred).cpu().numpy().flatten())
                gender_targets.extend(gender_target.cpu().numpy())
        
        avg_val_loss = val_loss / len(self.val_loader)
        age_mae = mean_absolute_error(age_targets, age_preds)
        gender_acc = accuracy_score(gender_targets, np.round(gender_preds))
        
        return avg_val_loss, age_mae, gender_acc
    
    def train(self, epochs):
        for epoch in range(epochs):
            train_loss, train_age_loss, train_gender_loss = self.train_epoch()
            val_loss, val_age_mae, val_gender_acc = self.validate()
            
            # Логирование в TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Age_MAE/val', val_age_mae, epoch)
            self.writer.add_scalar('Gender_Acc/val', val_gender_acc, epoch)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'Age MAE: {val_age_mae:.2f} | Gender Acc: {val_gender_acc:.2%}')
            
            # Early stopping и сохранение лучшей модели
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                best_model_path = os.path.join(self.model_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), best_model_path)
                print(f'Validation loss improved. Model saved to {best_model_path}')
            else:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
                if self.counter >= self.patience:
                    print('Early stopping triggered!')
                    break
        
        self.writer.close()
        return self.model
