import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

file_path = r"C:/Users/Asus/Desktop/SIES/COGNIFYZ/DATASETS/Dataset .csv"
df = pd.read_csv(file_path)

plt.figure(figsize=(10, 6))
df['Cuisines'].value_counts().head(10).plot(kind='barh', color='skyblue')
plt.title("Top 10 Cuisines Served Across All Restaurants")
plt.xlabel("Number of Restaurants")
plt.ylabel("Cuisine")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
df['City'].value_counts().head(10).plot(kind='bar', color='orange')
plt.title("Top 10 Cities with Most Restaurants")
plt.xlabel("City")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df = df.dropna(subset=['Latitude', 'Longitude'])

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']))
gdf.plot(markersize=2, color='blue', alpha=0.5)
plt.title("Restaurant Locations (Static Map)")
plt.show()
