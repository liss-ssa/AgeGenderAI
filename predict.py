import torch
from model import AgeGenderModel
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
        model = AgeGenderModel()
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

if __name__ == '__main__':
    try:
        # Автоматическое определение устройства
        device = 'auto'
        
        # Загрузка модели
        model = load_model('best_model.pth', device=device)
        if model is None:
            raise RuntimeError("Не удалось загрузить модель!")
        
        # Тестовое предсказание
        test_image = 'data/25_1_4_20161221193646742.jpg'
        age, gender_prob = predict(test_image, model)
        
        if age is not None and gender_prob is not None:
            print('\nРезультат предсказания:')
            print(f'• Возраст: {age:.1f} лет')
            print(f'• Пол: {"Мужской" if gender_prob > 0.5 else "Женский"} (вероятность: {max(gender_prob, 1-gender_prob):.2%})')
        else:
            print("Не удалось выполнить предсказание для тестового изображения")
            
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")