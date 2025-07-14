import torch
import numpy as np
import os
from datetime import datetime

class EarlyStopping:
    def __init__(self, patience=7, delta=0, verbose=True):
        """
        Args:
            patience (int): Количество эпох без улучшения перед остановкой
            delta (float): Минимальное изменение для признания улучшения
            verbose (bool): Вывод сообщений
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, model_path='checkpoints'):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """Сохраняет модель при улучшении"""
        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(path, f'best_model_{timestamp}.pth')
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss
        }, filename)
        
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {filename}')
        self.val_loss_min = val_loss

class ModelCheckpoint:
    """Для сохранения моделей через фиксированные интервалы"""
    def __init__(self, save_dir='checkpoints', save_freq=5):
        self.save_dir = save_dir
        self.save_freq = save_freq
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, epoch, model, optimizer, loss):
        if (epoch + 1) % self.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))