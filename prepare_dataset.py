import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Конфигурация путей
DATA_DIR = "data"
RESIZED_DIR = os.path.join(DATA_DIR, "resized_224")
SPLIT_DIR = os.path.join(DATA_DIR, "split")
OUTPUT_DIR = "processed_data"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

# Создаем директории
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

def collect_and_split_data():
    # 1. Собираем все изображения из обеих папок
    all_images = []
    
    # Собираем из основной папки data
    for root, _, files in os.walk(DATA_DIR):
        if root in [RESIZED_DIR, SPLIT_DIR]:
            continue  # Пропускаем служебные папки
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append({
                    'path': os.path.join(root, file),
                    'filename': file
                })
    
    # Собираем из resized_224
    for file in os.listdir(RESIZED_DIR):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_images.append({
                'path': os.path.join(RESIZED_DIR, file),
                'filename': file
            })
    
    # 2. Загружаем существующие разметки
    existing_test = set()
    test_csv_path = os.path.join(SPLIT_DIR, "test.csv")
    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path, header=None, names=["filename", "age", "gender"])
        existing_test = set(test_df["filename"])
    
    # 3. Разделяем данные (80/10/10)
    new_images = [img for img in all_images if img['filename'] not in existing_test]
    filenames = [img['filename'] for img in new_images]
    
    # Первое разделение: 80% train, 20% temp (val+test)
    train_filenames, temp_filenames = train_test_split(
        filenames, test_size=0.2, random_state=42
    )
    
    # Второе разделение: 10% val, 10% test
    val_filenames, test_filenames = train_test_split(
        temp_filenames, test_size=0.5, random_state=42
    )
    
    # 4. Объединяем с существующими тестовыми данными
    if existing_test:
        test_filenames.extend(existing_test)
    
    # 5. Копируем файлы в соответствующие директории
    def copy_files(filenames, target_dir):
        for filename in tqdm(filenames, desc=f"Copying to {target_dir}"):
            src_path = next(img['path'] for img in all_images if img['filename'] == filename)
            shutil.copy(src_path, os.path.join(target_dir, filename))
    
    copy_files(train_filenames, TRAIN_DIR)
    copy_files(val_filenames, VAL_DIR)
    copy_files(test_filenames, TEST_DIR)
    
    # 6. Создаем/обновляем CSV файлы
    def create_metadata_csv(filenames, csv_path):
        metadata = []
        for filename in filenames:
            try:
                # Парсим возраст и пол из имени файла (формат: age_gender_*.jpg)
                age_gender = filename.split('_')[:2]
                age = int(age_gender[0])
                gender = int(age_gender[1])
                metadata.append([filename, age, gender])
            except:
                continue
        
        pd.DataFrame(metadata).to_csv(csv_path, header=False, index=False)
    
    create_metadata_csv(train_filenames, os.path.join(SPLIT_DIR, "train.csv"))
    create_metadata_csv(val_filenames, os.path.join(SPLIT_DIR, "val.csv"))
    create_metadata_csv(test_filenames, os.path.join(SPLIT_DIR, "test.csv"))
    
    print(f"\nDataset prepared successfully!\n"
          f"Train: {len(train_filenames)} images\n"
          f"Val: {len(val_filenames)} images\n"
          f"Test: {len(test_filenames)} images")

if __name__ == "__main__":
    collect_and_split_data()