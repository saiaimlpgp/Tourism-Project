
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("tourism_project/deployment/model.pkl")

st.title("Wellness Tourism Package Purchase Prediction")

st.write(
    "Enter the customer details below to predict whether the customer "
    "is likely to purchase the Wellness Tourism Package."
)

# Customer details
age = st.number_input("Age", min_value=18.0, max_value=100.0, value=35.0)

type_of_contact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

duration_of_pitch = st.number_input(
    "Duration of Pitch",
    min_value=0.0,
    value=10.0
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business", "Government"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

number_of_person_visiting = st.number_input(
    "Number of Person Visiting",
    min_value=1,
    value=2,
    step=1
)

number_of_followups = st.number_input(
    "Number of Followups",
    min_value=0.0,
    value=3.0
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

preferred_property_star = st.number_input(
    "Preferred Property Star",
    min_value=1.0,
    max_value=5.0,
    value=3.0
)

marital_status = st.selectbox(
    "Marital Status",
    ["Married", "Divorced", "Single"]
)

number_of_trips = st.number_input(
    "Number of Trips",
    min_value=0.0,
    value=3.0
)

passport = st.selectbox(
    "Passport",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

own_car = st.selectbox(
    "Own Car",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

number_of_children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0.0,
    value=0.0
)

designation = st.selectbox(
    "Designation",
    ["Manager", "Executive", "Senior Manager", "AVP", "VP"]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=20000.0
)

# Create a dataframe from the user inputs
input_data = pd.DataFrame({
    "Age": [age],
    "TypeofContact": [type_of_contact],
    "CityTier": [city_tier],
    "DurationOfPitch": [duration_of_pitch],
    "Occupation": [occupation],
    "Gender": [gender],
    "NumberOfPersonVisiting": [number_of_person_visiting],
    "NumberOfFollowups": [number_of_followups],
    "ProductPitched": [product_pitched],
    "PreferredPropertyStar": [preferred_property_star],
    "MaritalStatus": [marital_status],
    "NumberOfTrips": [number_of_trips],
    "Passport": [passport],
    "PitchSatisfactionScore": [pitch_satisfaction_score],
    "OwnCar": [own_car],
    "NumberOfChildrenVisiting": [number_of_children_visiting],
    "Designation": [designation],
    "MonthlyIncome": [monthly_income]
})

# Make prediction
if st.button("Predict Purchase"):

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success(
            "Prediction: The customer is likely to purchase the Wellness Tourism Package."
        )
    else:
        st.info(
            "Prediction: The customer is unlikely to purchase the Wellness Tourism Package."
        )
