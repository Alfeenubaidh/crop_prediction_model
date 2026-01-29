import streamlit as st
import pandas as pd

from api.predict import predict_yield  # we will define this cleanly

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Crop Yield Prediction System",
    layout="centered"
)

st.title("🌾 Crop Yield Prediction")
st.markdown("Predict crop yield using satellite, weather, and soil data.")

# --------------------------------------------------
# User inputs
# --------------------------------------------------
st.subheader("Input Parameters")

state = st.selectbox(
    "State",
    ["Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh"]
)

district = st.text_input("District", "Ludhiana")

crop = st.selectbox(
    "Crop",
    ["Wheat", "Rice", "Maize"]
)

season = st.selectbox(
    "Season",
    ["Kharif", "Rabi"]
)

year = st.slider("Year", 2015, 2024, 2022)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Yield"):
    with st.spinner("Running prediction..."):
        try:
            input_payload = {
                "state": state,
                "district": district,
                "crop": crop,
                "season": season,
                "year": year,
            }

            prediction = predict_yield(input_payload)

            st.success("Prediction completed")

            st.metric(
                label="Predicted Yield (tons/hectare)",
                value=round(prediction, 2)
            )

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
