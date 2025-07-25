import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

df = pd.read_csv("C:/Users/Asus/Desktop/SIES/COGNIFYZ/DAYS/DAY 5/final_dataset_day5.csv")

X = df.drop("Average Cost for two", axis=1)
y = df["Average Cost for two"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(random_state=42)
param_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
search_rf = RandomizedSearchCV(rf, param_distributions=param_rf, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
search_rf.fit(X_train, y_train)
rf_best = search_rf.best_estimator_

xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
param_xgb = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.7, 0.8, 1],
    "colsample_bytree": [0.7, 0.8, 1]
}
search_xgb = RandomizedSearchCV(xgb, param_distributions=param_xgb, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
search_xgb.fit(X_train, y_train)
xgb_best = search_xgb.best_estimator_

svr = SVR()
param_svr = {
    "kernel": ["rbf", "linear"],
    "C": [0.1, 1, 10],
    "epsilon": [0.1, 0.2, 0.5]
}
search_svr = GridSearchCV(svr, param_grid=param_svr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
search_svr.fit(X_train, y_train)
svr_best = search_svr.best_estimator_

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    return rmse, r2, mae

print("Random Forest:", evaluate(rf_best, X_test, y_test))
print("XGBoost:", evaluate(xgb_best, X_test, y_test))
print("SVR:", evaluate(svr_best, X_test, y_test))
