import pandas as pd
import numpy as np

# Load dataset
file_path = r"C:/Users/Asus/Desktop/SIES/COGNIFYZ/DATASETS/Dataset .csv"
df = pd.read_csv(file_path)

# Feature Engineering
df['Restaurant_Name_Length'] = df['Restaurant Name'].astype(str).apply(len)
df['Address_Length'] = df['Address'].astype(str).apply(len)
df['Cuisine_Count'] = df['Cuisines'].astype(str).apply(lambda x: len(x.split(',')))
df['Votes_per_Cost'] = df['Votes'] / (df['Average Cost for two'] + 1)
df['Has_Table_Booking'] = df['Has Table booking'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
df['Has_Online_Delivery'] = df['Has Online delivery'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

# Save the dataset
df.to_csv("C:/Users/Asus/Desktop/SIES/COGNIFYZ/DATASETS/Day_4.csv", index=False)
