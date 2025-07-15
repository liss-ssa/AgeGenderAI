import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Пути
DATA_DIR = "UTKFace/part1"
OUTPUT_DIR = "UTKFace/part1"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

# Создаем директории
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Собираем метаданные
data = []
for img_name in os.listdir(DATA_DIR):
    try:
        parts = img_name.split("_")
        if len(parts) == 4:
            age, gender, race, _ = parts
        elif len(parts) == 3:
            age, gender, race = parts
        else:
            print(f"Пропущен файл с нестандартным именем: {img_name}")
            continue
            
        data.append({
            "image": img_name,
            "age": int(age),
            "gender": int(gender)  # 0 = Male, 1 = Female
        })
    except Exception as e:
        print(f"Ошибка при обработке файла {img_name}: {e}")

df = pd.DataFrame(data)

# Разделяем данные
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Функция для копирования файлов
def copy_files(df, target_dir):
    for _, row in df.iterrows():
        shutil.copy(
            os.path.join(DATA_DIR, row["image"]),
            os.path.join(target_dir, row["image"])
        )

copy_files(train_df, TRAIN_DIR)
copy_files(val_df, VAL_DIR)
copy_files(test_df, TEST_DIR)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")