import joblib
import numpy as np  # make sure this import is present
model = joblib.load("C:/Users/Asus/Desktop/SIES/COGNIFYZ/FINAL FOLDER/LEVELS/LEVEL 3/final_gradient_boosting_model_clean.pkl")
joblib.dump(model, "final_gradient_boosting_model_clean.pkl")
