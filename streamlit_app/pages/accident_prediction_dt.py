import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime
import sys
sys.path.append('../')
from prediction.supervised.preprocess_data import get_prediction

# Define the options for each feature
light_conditions_options = ["Daylight", "Darkness - lights lit", "Darkness - no lighting", "Darkness - lights unlit", "Darkness - lighting unknown"]
weather_conditions_options = ["Fine no high winds", "Raining no high winds", "Snowing no high winds", "Fine + high winds", "Raining + high winds", "Snowing + high winds", "Fog or mist"]
speed_limit_options = [20, 30, 40, 50, 60, 70]
road_type_options = ["Single carriageway", "Dual carriageway", "Roundabout", "One way street"]
road_surface_conditions_options = ["Dry", "Wet or damp", "Snow", "Frost or ice", "Flood over 3cm. deep"]
urban_or_rural_area_options = ["Urban", "Rural"]
age_band_of_driver_options = ["0 - 5", "6 - 10", "11 - 15", "16 - 20", "21 - 25", "26 - 35", "36 - 45", "46 - 55", "56 - 65", "66 - 75", "Over 75"]
sex_of_driver_options = ["Male", "Female"]
age_of_vehicle_options = list(range(1, 21))  # Assuming vehicle age between 1 and 20 years
vehicle_manoeuvre_options = ["Reversing", "Parked", "Waiting to go - held up", "Slowing or stopping", "Moving off", "U-turn", "Turning left", "Waiting to turn left", "Turning right", "Waiting to turn right", "Changing lane to left", "Changing lane to right", "Overtaking moving vehicle - offside", "Overtaking stationary vehicle - offside", "Overtaking moving vehicle - nearside", "Overtaking stationary vehicle - nearside", "Going ahead left-hand bend", "Going ahead right-hand bend", "Going ahead other"]



st.title("Accident Severity Prediction")

col1, col2 = st.columns(2)

with col1:
    input_scenario = {
        "Time": st.time_input("Time of Accident", datetime.time(8, 45)),
        "Light_Conditions": st.selectbox("Light Conditions", light_conditions_options),
        "Weather_Conditions": st.selectbox("Weather Conditions", weather_conditions_options),
        "Speed_limit": st.selectbox("Speed Limit", speed_limit_options),
        "Road_Type": st.selectbox("Road Type", road_type_options),
        "Road_Surface_Conditions": st.selectbox("Road Surface Conditions", road_surface_conditions_options)
    }

with col2:
    input_scenario.update({
        "Urban_or_Rural_Area": st.selectbox("Urban or Rural Area", urban_or_rural_area_options),
        "Age_Band_of_Driver": st.selectbox("Age Band of Driver", age_band_of_driver_options),
        "Sex_of_Driver": st.selectbox("Sex of Driver", sex_of_driver_options),
        "Age_of_Vehicle": st.selectbox("Age of Vehicle", age_of_vehicle_options),
        "Vehicle_Manoeuvre": st.selectbox("Vehicle Manoeuvre", vehicle_manoeuvre_options)
    })

if st.button("Predict Severity"):
    # Get prediction
    severity = get_prediction(pd.DataFrame([input_scenario]))[0]

    st.write(f"Predicted severity: {severity}")
