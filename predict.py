import torch
from PIL import Image
from models.mobilenetv3 import AgeGenderModel
from utils.transforms import val_transform
import os
import argparse

def main():
    # Парсинг аргументов
    parser = argparse.ArgumentParser(description='Age and Gender Prediction')
    parser.add_argument('--image', type=str, required=True, 
                      help='Path to input image (e.g., UTKFace/part1/1_0_0_20161219194756275.jpg)')
    parser.add_argument('--model', type=str, default='saved_models/best_model.pth', 
                      help='Path to model weights (default: saved_models/best_model.pth)')
    args = parser.parse_args()

    # Проверка существования файлов
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        print("Please provide a valid image path.")
        return

    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        print("Please:")
        print("1. Train the model first (run train.py)")
        print("2. Or specify the correct model path using --model")
        return

    try:
        # Инициализация модели
        model = AgeGenderModel()
        
        # Загрузка весов модели
        checkpoint = torch.load(args.model, map_location='cpu')
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        # Обработка изображения
        img = Image.open(args.image).convert("RGB")
        img_tensor = val_transform(img).unsqueeze(0)

        # Предсказание
        with torch.no_grad():
            age, gender = model(img_tensor)
            gender_prob = torch.sigmoid(gender).item() if not hasattr(model, 'gender_head') else gender.item()
            gender_label = "Female" if gender_prob > 0.5 else "Male"
            age_label = max(0, int(age.item()))  # Возраст не может быть отрицательным
            confidence = max(gender_prob, 1 - gender_prob)

        # Вывод результатов
        print("\n--- Prediction Results ---")
        print(f"Image: {os.path.basename(args.image)}")
        print(f"Predicted Gender: {gender_label} (confidence: {confidence:.1%})")
        print(f"Predicted Age: {age_label} years")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Possible solutions:")
        print("- Check if the model architecture matches the saved weights")
        print("- Verify the image is a valid JPEG/PNG file")
        print("- Make sure all dependencies are properly installed")

if __name__ == '__main__':
    main()