import torch.nn as nn
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, accuracy_score

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    age_criterion = nn.MSELoss()
    gender_criterion = nn.BCELoss()
    
    for images, targets in dataloader:
        images = images.to(device)
        age_target = targets['age'].to(device)
        gender_target = targets['gender'].to(device)
        
        optimizer.zero_grad()
        
        age_pred, gender_pred = model(images)
        
        age_loss = age_criterion(age_pred.squeeze(), age_target)
        gender_loss = gender_criterion(gender_pred.squeeze(), gender_target)
        loss = age_loss + gender_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    age_preds = []
    age_targets = []
    gender_preds = []
    gender_targets = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            
            age_pred, gender_pred = model(images)
            
            age_preds.extend(age_pred.cpu().numpy().flatten())
            age_targets.extend(targets['age'].cpu().numpy())
            gender_preds.extend(gender_pred.cpu().numpy().flatten())
            gender_targets.extend(targets['gender'].cpu().numpy())
    
    age_mae = mean_absolute_error(age_targets, age_preds)
    gender_acc = accuracy_score(gender_targets, np.round(gender_preds))
    
    return age_mae, gender_acc