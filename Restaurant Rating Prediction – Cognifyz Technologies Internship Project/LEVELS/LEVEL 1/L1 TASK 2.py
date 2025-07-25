import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/Asus/Desktop/SIES/COGNIFYZ/DATASETS/Dataset .csv")

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='Reds')
plt.title("Missing Value Heatmap (Before Cleaning)")
plt.show()

df.isnull().sum().plot(kind='bar', color='blue')
plt.title("Missing Values Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df['Cuisines'] = df['Cuisines'].fillna("Unknown")
df = df.drop_duplicates()
