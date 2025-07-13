import torch.nn as nn
from torchvision.models import efficientnet_v2_s

class EfficientAgeGender(nn.Module):
    def __init__(self, gender_classes=1, age_classes=1):
        super().__init__()
        self.backbone = efficientnet_v2_s(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Головы для задач
        self.age_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, age_classes)
        )
        
        self.gender_head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.SiLU(),
            nn.Linear(256, gender_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x).flatten(1)
        return self.age_head(x), self.gender_head(x)