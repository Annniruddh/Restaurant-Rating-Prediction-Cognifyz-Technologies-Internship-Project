import pandas as pd

file_path = r"C:/Users/Asus/Desktop/SIES/COGNIFYZ/DATASETS/Dataset .csv"
df = pd.read_csv(file_path)
print("Dataset Loaded Successfully")

print("Initial Shape:", df.shape)
print(df.head())

df = df.drop_duplicates()
print("After Dropping Duplicates:", df.shape)

df.to_csv("Cleaned_Restaurant_Data_Task1_19072025.csv", index=False)