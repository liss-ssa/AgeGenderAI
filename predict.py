import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.efficient_net import EfficientAgeGender  # Импорт архитектуры модели
import os

# 1. Загрузка модели
def load_model(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = EfficientAgeGender().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Переводим в режим оценки
    return model

# 2. Препроцессинг изображения
def preprocess_image(image_path, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet-нормализация
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Добавляем batch-размер

# 3. Предсказание
def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        age_pred, gender_pred = model(image_tensor)
        age = age_pred.item()  # Возраст (число)
        gender = 'Male' if gender_pred.item() < 0.5 else 'Female'  # Пол (бинарный)
    return age, gender

# 4. Основная функция
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Пути (замените на ваши)
    checkpoint_path = 'checkpoints/final_model.pth'  # Путь к весам модели
    test_image_path = 'test_photo.jpg'     # Путь к тестовому изображению

    # Загрузка модели и изображения
    model = load_model(checkpoint_path, device)
    image_tensor = preprocess_image(test_image_path)

    # Предсказание
    age, gender = predict(model, image_tensor, device)
    print(f'Predicted Age: {age:.1f} years')
    print(f'Predicted Gender: {gender}')