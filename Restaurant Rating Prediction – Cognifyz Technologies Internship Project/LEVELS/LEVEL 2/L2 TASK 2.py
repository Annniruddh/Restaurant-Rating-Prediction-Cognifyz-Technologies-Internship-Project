import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

df = pd.read_csv("C:/Users/Asus/Desktop/SIES/COGNIFYZ/DATASETS/Day_4.csv")

features = [
    'Price range', 'Aggregate rating', 'Votes', 
    'Restaurant_Name_Length', 'Address_Length', 
    'Cuisine_Count', 'Votes_per_Cost', 
    'Has_Table_Booking', 'Has_Online_Delivery'
]
X = df[features]
y = df['Average Cost for two']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0),
    'SVR': SVR()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name}: MSE={mse:.2f}, R2={r2:.4f}")

X.to_csv("C:/Users/Asus/Desktop/SIES/COGNIFYZ/DATASETS/final_dataset_day5.csv", index=False)
y.to_csv("C:/Users/Asus/Desktop/SIES/COGNIFYZ/DATASETS/target_day5.csv", index=False)
