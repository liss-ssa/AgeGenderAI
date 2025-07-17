import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_absolute_error, r2_score, 
                            confusion_matrix, classification_report, 
                            roc_curve, auc)
from PIL import Image
from tqdm import tqdm
from models.mobilenetv3 import AgeGenderModel
from utils.transforms import val_transform
from utils.dataloader import UTKFaceDataset
from multiprocessing import freeze_support

def main():
    # Конфигурация
    TEST_DIR = "UTKFace/part1/test"
    MODEL_PATH = "saved_models/best_model.pth"
    PLOTS_DIR = "plots"
    MODEL_NAME = "MobileNetV3 (Age+Gender)"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Загрузка модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AgeGenderModel().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Подготовка тестового датасета
    test_dataset = UTKFaceDataset(TEST_DIR, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0  # Установите num_workers=0 для избежания проблем
    )

    # Функция для вычисления метрик
    def evaluate_model(model, dataloader, device):
        ages_true, ages_pred = [], []
        genders_true, genders_pred_probs = [], []
        
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Evaluating"):
                images = images.to(device)
                age_true = targets['age'].numpy()
                gender_true = targets['gender'].numpy()
                
                age_pred, gender_pred = model(images)
                
                ages_true.extend(age_true)
                ages_pred.extend(age_pred.cpu().numpy().flatten())
                genders_true.extend(gender_true)
                genders_pred_probs.extend(gender_pred.cpu().numpy().flatten())
        
        # Конвертируем в numpy массивы
        ages_true = np.array(ages_true)
        ages_pred = np.array(ages_pred)
        genders_true = np.array(genders_true)
        genders_pred_probs = np.array(genders_pred_probs)
        
        # Бинаризация предсказаний пола
        genders_pred = (genders_pred_probs > 0.5).astype(int)
        
        return {
            'ages_true': ages_true,
            'ages_pred': ages_pred,
            'genders_true': genders_true,
            'genders_pred': genders_pred,
            'genders_pred_probs': genders_pred_probs
        }

    # Вычисление метрик
    results = evaluate_model(model, test_loader, device)

    # 1. Метрики для возраста (регрессия)
    age_mae = mean_absolute_error(results['ages_true'], results['ages_pred'])
    age_r2 = r2_score(results['ages_true'], results['ages_pred'])

    # 2. Метрики для пола (классификация)
    gender_report = classification_report(
        results['genders_true'], 
        results['genders_pred'],
        target_names=['Male', 'Female'],
        output_dict=True
    )
    gender_cm = confusion_matrix(results['genders_true'], results['genders_pred'])

    # ROC-кривая для пола
    fpr, tpr, thresholds = roc_curve(results['genders_true'], results['genders_pred_probs'])
    roc_auc = auc(fpr, tpr)

    # Визуализация результатов

    # 1. График истинного vs предсказанного возраста
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=results['ages_true'], 
        y=results['ages_pred'],
        alpha=0.6,
        edgecolor=None
    )
    plt.plot([0, 100], [0, 100], '--', color='red', linewidth=1)
    plt.title(f'{MODEL_NAME}\nAge Prediction: True vs Predicted\nMAE: {age_mae:.2f}, R2: {age_r2:.2f}')
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'age_prediction_scatter.png'), dpi=300)
    plt.close()

    # 2. Распределение ошибок для возраста
    age_errors = results['ages_pred'] - results['ages_true']
    plt.figure(figsize=(10, 6))
    sns.histplot(age_errors, kde=True, bins=30)
    plt.title(f'{MODEL_NAME}\nAge Prediction Error Distribution\nMAE: {age_mae:.2f}')
    plt.xlabel('Prediction Error (years)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'age_error_distribution.png'), dpi=300)
    plt.close()

    # 3. Матрица ошибок для пола
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        gender_cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Male', 'Female'],
        yticklabels=['Male', 'Female']
    )
    plt.title(f'{MODEL_NAME}\nGender Confusion Matrix\nAccuracy: {gender_report["accuracy"]:.2f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'gender_confusion_matrix.png'), dpi=300)
    plt.close()

    # 4. ROC-кривая для пола
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{MODEL_NAME}\nReceiver Operating Characteristic for Gender')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'gender_roc_curve.png'), dpi=300)
    plt.close()

    # 5. Отчет по метрикам классификации
    plt.figure(figsize=(8, 4))
    metrics = ['precision', 'recall', 'f1-score']
    male_scores = [gender_report['Male'][m] for m in metrics]
    female_scores = [gender_report['Female'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width/2, male_scores, width, label='Male')
    plt.bar(x + width/2, female_scores, width, label='Female')

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'{MODEL_NAME}\nGender Classification Metrics')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'gender_metrics.png'), dpi=300)
    plt.close()

    print("Графики успешно сохранены в папку plots/")

if __name__ == '__main__':
    freeze_support()  # Для поддержки замороженных приложений (например, pyinstaller)
    main()