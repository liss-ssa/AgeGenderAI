import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(metadata_path, output_dir, val_size=0.1, test_size=0.1, random_state=42):
    """
    Разделяет данные на train/val/test с точными пропорциями 80/10/10.
    Автоматически группирует редкие возрасты для стратификации.
    """
    df = pd.read_csv(metadata_path)
    
    # Группировка возрастов (объединяем редкие категории)
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, np.inf]
    age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '80+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    
    # Проверяем распределение по группам
    print("Распределение по возрастным группам:")
    print(df['age_group'].value_counts().sort_index())
    
    # Сначала отделяем train (80%)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),  # 20% на val + test
        stratify=df['age_group'],  # Стратификация по возрастным группам
        random_state=random_state
    )
    
    # Затем делим оставшиеся 20% на val (10%) и test (10%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size/(val_size + test_size),  # 0.1/0.2 = 0.5
        stratify=temp_df['age_group'],
        random_state=random_state
    )
    
    # Проверка пропорций
    total = len(df)
    print("\nИтоговое разделение:")
    print(f"Total: {total}")
    print(f"Train: {len(train_df)} ({len(train_df)/total:.1%})")
    print(f"Val: {len(val_df)} ({len(val_df)/total:.1%})")
    print(f"Test: {len(test_df)} ({len(test_df)/total:.1%})")
    
    # Удаляем временную колонку перед сохранением
    for df_split in [train_df, val_df, test_df]:
        df_split.drop('age_group', axis=1, inplace=True)
    
    # Сохранение
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

if __name__ == "__main__":
    split_dataset(
        metadata_path="data/resized_224/metadata.csv",
        output_dir="data/splits"
    )