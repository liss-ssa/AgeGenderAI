import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class AgeGenderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mobilenet_v3_small(pretrained=True)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Identity()

        # Ветки для возраста и пола
        self.age_head = nn.Linear(in_features, 1)
        self.gender_head = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        age = self.age_head(features)
        gender = self.gender_head(features)
        return age, gender