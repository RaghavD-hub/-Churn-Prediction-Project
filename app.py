import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the saved model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Churn Prediction App")

st.write("Enter the user details to predict churn:")

# Define input fields based on features used in the model
# Assuming features from the notebook: age, hour, day, etc.
# Adjust the input fields as per actual features in the dataset

age = st.number_input("Age", min_value=10, max_value=100, value=30)
hour = st.number_input("Hour of app use", min_value=0, max_value=23, value=12)
day = st.number_input("Day of week", min_value=0, max_value=6, value=3)
numscreens = st.number_input("Number of screens visited", min_value=0, max_value=100, value=10)

minigame = st.selectbox("Played minigame?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
used_premium_feature = st.selectbox("Used premium feature?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
liked = st.selectbox("Liked the app?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

if st.button("Predict"):
    # Prepare the input data as a numpy array or dataframe
    input_data = np.array([[day, hour, age, numscreens, minigame, used_premium_feature, liked]])
    # Predict churn
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success(f"The user is likely to churn. Probability: {prediction_proba[0][1]:.2f}")
    else:
        st.info(f"The user is unlikely to churn. Probability: {prediction_proba[0][0]:.2f}")
