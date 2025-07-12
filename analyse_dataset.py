import os
import pandas as pd
import matplotlib.pyplot as plt

# Собираем метаданные из названий файлов
# Собираем метаданные из названий файлов
data = []
for img_name in os.listdir('utkface_aligned_cropped/UTKFace'):
    age, gender, _ = img_name.split('_')[:3]
    data.append({'age': int(age), 'gender': int(gender)})

df = pd.DataFrame(data)

# Гистограмма возраста
plt.figure(figsize=(12, 5))
plt.hist(df['age'], bins=100, edgecolor='black')
plt.title('Распределение возраста')
plt.xlabel('Возраст')
plt.ylabel('Количество')
plt.savefig('plots/age_distribution.png', dpi=300, bbox_inches='tight')
#plt.show()

# Гистограмма пола (0 — женский, 1 — мужской)
plt.figure(figsize=(6, 4))
plt.bar(['Female', 'Male'], df['gender'].value_counts())
plt.title('Распределение пола')
plt.savefig('plots/gender_distribution.png', dpi=300, bbox_inches='tight')
#plt.show()