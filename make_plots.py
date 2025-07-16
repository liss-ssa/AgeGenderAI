import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from model import AgeGenderModel, UTKFaceDataset

# Конфигурация
TEST_CSV = "data/splits/test.csv"
IMG_DIR = "data/resized_224"
MODEL_PATH = "best_model.pth"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
MODEL_NAME = "ResNet18 (Age+Gender)"

# Загрузка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AgeGenderModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Загрузка тестовых данных
test_dataset = UTKFaceDataset(TEST_CSV, IMG_DIR, transform=None)
test_df = pd.read_csv(TEST_CSV)
print(f"Loaded {len(test_df)} test samples")

# Трансформации изображений
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Функция для предсказаний
def evaluate_model(model, dataset):
    ages_true, ages_pred = [], []
    genders_true, genders_pred_probs = [], []
    
    for idx in tqdm(range(len(dataset)), desc="Processing"):
        image, targets = dataset[idx]
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            age_pred, gender_pred = model(image)
        
        ages_true.append(targets['age'].item())
        ages_pred.append(age_pred.item())
        genders_true.append(targets['gender'].item())
        genders_pred_probs.append(gender_pred.item())
    
    return {
        "ages_true": np.array(ages_true),
        "ages_pred": np.array(ages_pred),
        "genders_true": np.array(genders_true),
        "genders_pred_probs": np.array(genders_pred_probs),
    }

# Получение предсказаний
results = evaluate_model(model, test_dataset)

# Вычисление метрик
## Для возраста
results["ages_pred"] = np.clip(results["ages_pred"], 0, 100)
age_mae = mean_absolute_error(results["ages_true"], results["ages_pred"])
age_r2 = r2_score(results["ages_true"], results["ages_pred"])

## Для пола
results["genders_pred"] = (results["genders_pred_probs"] > 0.5).astype(int)
gender_report = classification_report(
    results["genders_true"], 
    results["genders_pred"],
    target_names=["Male", "Female"],
    output_dict=True
)
gender_cm = confusion_matrix(results["genders_true"], results["genders_pred"])

# ROC-кривая
fpr, tpr, _ = roc_curve(results["genders_true"], results["genders_pred_probs"])
roc_auc = auc(fpr, tpr)

# --------------------------------------------------
# Построение графиков
# --------------------------------------------------
sns.set_theme(style="whitegrid")

# 1. График "Истинный vs Предсказанный возраст"
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=results["ages_true"], 
    y=results["ages_pred"],
    alpha=0.6,
    hue=results["genders_true"],
    palette={0: "pink", 1: "blue"},
    style=results["genders_true"],
    markers={0: "o", 1: "s"}
)
plt.plot([0, 100], [0, 100], "--", color="red", linewidth=1)
plt.title(f"{MODEL_NAME}\nAge Prediction\nMAE: {age_mae:.2f} years | R2: {age_r2:.2f}")
plt.xlabel("True Age")
plt.ylabel("Predicted Age")
plt.legend(title="Gender", labels=["Female", "Male"])
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "resnet_age_scatter.png"), dpi=300, bbox_inches="tight")
plt.close()

# 2. Распределение ошибок возраста
age_errors = results["ages_pred"] - results["ages_true"]
plt.figure(figsize=(10, 6))
sns.histplot(age_errors, kde=True, bins=30, color="purple")
plt.title(f"{MODEL_NAME}\nAge Prediction Error Distribution\nMAE: {age_mae:.2f} years")
plt.xlabel("Prediction Error (years)")
plt.ylabel("Count")
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "resnet_age_error_dist.png"), dpi=300)
plt.close()

# 3. Матрица ошибок для пола
plt.figure(figsize=(8, 6))
sns.heatmap(
    gender_cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues",
    xticklabels=["Male", "Female"],
    yticklabels=["Male", "Female"]
)
plt.title(f"{MODEL_NAME}\nGender Classification\nAccuracy: {gender_report['accuracy']:.2f}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(PLOTS_DIR, "resnet_gender_cm.png"), dpi=300)
plt.close()

# 4. ROC-кривая
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "--", color="navy", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"{MODEL_NAME}\nGender ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "resnet_gender_roc.png"), dpi=300)
plt.close()

# 5. Метрики классификации по полу
metrics = ["precision", "recall", "f1-score"]
male_scores = [gender_report["Male"][m] for m in metrics]
female_scores = [gender_report["Female"][m] for m in metrics]

plt.figure(figsize=(10, 5))
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, male_scores, width, label="Male", color="blue")
plt.bar(x + width/2, female_scores, width, label="Female", color="pink")

plt.xlabel("Metric")
plt.ylabel("Score")
plt.title(f"{MODEL_NAME}\nGender Classification Metrics")
plt.xticks(x, metrics)
plt.ylim(0, 1.1)
plt.legend()
plt.grid(True, axis="y")
plt.savefig(os.path.join(PLOTS_DIR, "resnet_gender_metrics.png"), dpi=300)
plt.close()

print(f"Графики успешно сохранены в папку {PLOTS_DIR}/")