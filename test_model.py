import os
import pandas as pd
from predict import predict, load_model
import torch

def run_tests(model, test_dir="test_photos"):
    # Проверка существования папки
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Папка {test_dir} не найдена!")
    
    results = []
    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        try:
            age, gender = predict(img_path, model)
            results.append({
                'file': img_name,
                'age': round(age, 1),
                'gender': "Male" if gender > 0.5 else "Female",
                'gender_prob': round(max(gender, 1-gender), 4)
            })
        except Exception as e:
            print(f"Ошибка на {img_name}: {str(e)}")
    
    # Сохранение результатов
    df = pd.DataFrame(results)
    df.to_csv("test_results.csv", index=False)
    print("Результаты сохранены в test_results.csv")
    return df

if __name__ == "__main__":
    print(f"PyTorch использует: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    model = load_model("best_model.pth")
    if model:
        run_tests(model)