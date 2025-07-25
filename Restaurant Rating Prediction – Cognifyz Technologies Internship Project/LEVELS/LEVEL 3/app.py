import streamlit as st
import pandas as pd
import joblib

model = joblib.load("C:/Users/Asus/Desktop/SIES/COGNIFYZ/FINAL FOLDER/LEVELS/LEVEL 3/final_gradient_boosting_model_clean.pkl")

reference_df = pd.read_csv("C:/Users/Asus/Desktop/SIES/COGNIFYZ/DAYS/DAY 5/final_dataset_day5.csv")
feature_columns = reference_df.drop("Aggregate rating", axis=1).columns.tolist()

st.title("Restaurant Rating Predictor")

cost = st.number_input("Average Cost for Two", min_value=0)
votes = st.number_input("Votes", min_value=0)
has_table_booking = st.selectbox("Has Table Booking?", ["Yes", "No"])
has_online_delivery = st.selectbox("Has Online Delivery?", ["Yes", "No"])
cuisine_count = st.slider("Cuisine Count", 1, 10, 2)
address_length = st.slider("Address Length", 10, 100, 30)
name_length = st.slider("Restaurant Name Length", 5, 50, 15)
price_range = st.selectbox("Price Range (1 = Low, 4 = High)", [1, 2, 3, 4])

votes_per_cost = votes / cost if cost > 0 else 0
has_table_booking = 1 if has_table_booking == "Yes" else 0
has_online_delivery = 1 if has_online_delivery == "Yes" else 0

raw_input = {
    "Average Cost for two": cost,
    "Votes": votes,
    "Has_Table_Booking": has_table_booking,
    "Has_Online_Delivery": has_online_delivery,
    "Cuisine_Count": cuisine_count,
    "Address_Length": address_length,
    "Restaurant_Name_Length": name_length,
    "Price range": price_range,
    "Votes_per_Cost": votes_per_cost
}

input_df = pd.DataFrame([raw_input]).reindex(columns=feature_columns)

if st.button("Predict Rating"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Aggregate Rating: {prediction:.2f}")


