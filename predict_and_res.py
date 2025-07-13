import random
import pandas as pd
from pathlib import Path
import torch
from model import ImprovedAgeGenderModel
from torchvision import transforms
import cv2
import os
from typing import Tuple, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path: str, device: str = 'auto') -> torch.nn.Module:
    """
    Универсальная загрузка модели с автоматическим определением устройства
    
    Args:
        model_path: Путь к файлу с весами модели (.pth)
        device: 'auto', 'cuda' или 'cpu'
    
    Returns:
        Загруженная модель в режиме eval()
    """
    # Определение устройства
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    logger.info(f"Попытка загрузки модели на устройство: {device}")
    
    try:
        # Инициализация модели
        model = ImprovedAgeGenderModel()
        model.to(device)
        
        # Загрузка весов с обработкой разных случаев:
        # 1. Модель была сохранена на GPU, а загружается на CPU
        # 2. Модель была сохранена как DataParallel
        state_dict = torch.load(model_path, map_location=device)
        
        # Удаление 'module.' из ключей, если модель сохранялась через DataParallel
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.eval()
        
        logger.info(f"Модель успешно загружена на {device}")
        return model
        
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {str(e)}")
        raise

def predict(image_path: str, model: torch.nn.Module, target_size: Tuple[int, int] = (224, 224)) -> Tuple[Optional[float], Optional[float]]:
    """
    Предсказание возраста и пола по изображению
    
    Args:
        image_path: Путь к изображению
        model: Загруженная модель AgeGenderModel
        target_size: Размер изображения для модели
    
    Returns:
        (age, gender_prob) или (None, None) при ошибке
    """
    # Проверка существования файла
    if not os.path.exists(image_path):
        logger.error(f"Файл не найден: {image_path}")
        return None, None
    
    # Трансформы для изображения
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),  # Ресайз до нужного размера
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Не удалось загрузить изображение (возможно, поврежденный файл)")
        
        # Конвертация цветового пространства
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Применение трансформов и добавление batch-размерности
        image_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)
        
        # Предсказание
        with torch.no_grad():
            age, gender = model(image_tensor)
            return age.item(), gender.item()
            
    except Exception as e:
        logger.error(f"Ошибка при обработке {image_path}: {str(e)}")
        return None, None

def analyze_predictions(data_dir: str = 'data/resized_224', num_samples: int = 100) -> pd.DataFrame:
    """
    Анализирует предсказания модели на случайных изображениях из папки
    
    Args:
        data_dir: Путь к папке с изображениями
        num_samples: Количество изображений для анализа
    
    Returns:
        DataFrame с результатами
    """
    # Получаем список всех подходящих файлов
    image_files = [f for f in Path(data_dir).glob('*.jpg') if len(f.stem.split('_')) == 4]
    
    # Выбираем случайные файлы (но не больше, чем есть)
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    results = []
    
    # Загружаем модель
    model = load_model('saved_models/best_model.pth')
    if model is None:
        raise RuntimeError("Не удалось загрузить модель!")
    
    for file_path in selected_files:
        # Парсим метаданные из имени файла
        try:
            age_true, gender_true, race, _ = file_path.stem.split('_')
            age_true = int(age_true)
            gender_true = int(gender_true)  # 0 - женский, 1 - мужской
        except:
            logger.error(f"Неверный формат имени файла: {file_path}")
            continue
        
        # Делаем предсказание
        age_pred, gender_prob = predict(str(file_path), model)
        if age_pred is None or gender_prob is None:
            continue
        
        # Определяем предсказанный пол
        gender_pred = 1 if gender_prob > 0.5 else 0
        gender_correct = gender_pred == gender_true
        
        # Добавляем результат
        results.append({
            'filename': file_path.name,
            'true_age': age_true,
            'predicted_age': round(age_pred, 1),
            'age_error': round(abs(age_pred - age_true), 1),
            'true_gender': 'Female' if gender_true == 0 else 'Male',
            'predicted_gender': 'Female' if gender_pred == 0 else 'Male',
            'gender_prob': round(max(gender_prob, 1-gender_prob), 4),
            'gender_correct': gender_correct,
            'race': race
        })
    
    return pd.DataFrame(results)

if __name__ == '__main__':
    try:
        # Анализируем 100 случайных изображений
        results_df = analyze_predictions(num_samples=100)
        
        # Сохраняем результаты
        output_file = 'prediction_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nРезультаты сохранены в файл: {output_file}")
        
        # Выводим статистику
        accuracy = results_df['gender_correct'].mean()
        mean_age_error = results_df['age_error'].mean()
        
        print(f"\nСтатистика:")
        print(f"- Точность определения пола: {accuracy:.2%}")
        print(f"- Средняя ошибка возраста: {mean_age_error:.1f} лет")
        print(f"- Проанализировано изображений: {len(results_df)}")
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")