import os
import cv2
import pandas as pd
from tqdm import tqdm  # Для прогресс-бара

def resize_and_save_images(input_dir, output_dir, target_size=(224, 224)):
    """
    Ресайзит изображения и сохраняет их в новую директорию.
    
    :param input_dir: Папка с исходными изображениями (например, 'UTKFace')
    :param output_dir: Папка для сохранения ресайзнутых изображений
    :param target_size: Размер (ширина, высота)
    """
    os.makedirs(output_dir, exist_ok=True)
    data = []
    
    for img_name in tqdm(os.listdir(input_dir)):
        try:
            # Парсинг названия файла
            age, gender, _ = img_name.split('_')[:3]
            age, gender = int(age), int(gender)
            
            # Загрузка и ресайз изображения
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Конвертация в RGB
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # Сохранение
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
            
            # Запись метаданных
            data.append({'filename': img_name, 'age': age, 'gender': gender})
        
        except Exception as e:
            print(f"Ошибка при обработке {img_name}: {e}")
    
    # Сохранение метаданных в CSV
    pd.DataFrame(data).to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
    print(f"Изображения сохранены в {output_dir}. Всего: {len(data)}")

# Пример вызова:
if __name__ == "__main__":
    resize_and_save_images(
        input_dir="utkface_aligned_cropped/UTKFace",
        output_dir="data/resized_224"
    )