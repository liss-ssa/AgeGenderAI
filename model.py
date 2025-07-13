import torch.nn as nn
from torchvision.models import resnet18
import torch
import pandas as pd
import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class ImprovedAgeGenderModel(nn.Module):
    def __init__(self, gender_class_weights=None):
        super().__init__()
        self.gender_class_weights = gender_class_weights
        
        # Основная архитектура
        self.backbone = resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Общие слои
        self.shared_fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Голова для возраста
        self.age_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        # Голова для пола
        self.gender_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_fc(features)
        
        age = self.age_head(shared)
        gender = self.gender_head(shared)
        
        return age, gender