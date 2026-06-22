import streamlit as st
st.set_page_config(
    page_title="Traffic Volume Prediction",
    page_icon="🚗",
    layout="centered"
)
import pandas as pd
import joblib

volume_model = joblib.load('traffic_volume_model.pkl')
level_model = joblib.load('traffic_level_model.pkl')
columns = joblib.load('traffic_columns.pkl')

st.title("🚗 Traffic Volume Prediction")

st.markdown(
    "Predict the traffic volume and congestion level using weather and time information."
)

st.markdown("---")
temp = st.number_input("Temperature (K)", value=290.0)

rain_1h = st.number_input("Rain in last 1 hour", value=0.0)

snow_1h = st.number_input("Snow in last 1 hour", value=0.0)

clouds_all = st.slider("Cloud Cover (%)", 0, 100, 50)

time = st.slider("Hour of the Day", 0, 23, 12)

is_weekend = st.selectbox(
    "Is it a weekend?",
    ["No", "Yes"]
)

is_holiday = st.selectbox(
    "Is it a holiday?",
    ["No", "Yes"]
)

weather = st.selectbox(
    "Weather",
    [
        "Clear",
        "Clouds",
        "Drizzle",
        "Fog",
        "Haze",
        "Mist",
        "Rain",
        "Smoke",
        "Snow",
        "Squall",
        "Thunderstorm"
    ]
)
if st.button("Predict"):

    try:

        input_data = {
            'temp': temp,
            'rain_1h': rain_1h,
            'snow_1h': snow_1h,
            'clouds_all': clouds_all,
            'is_weekend': 1 if is_weekend == "Yes" else 0,
            'time': time,
            'is_holiday': 1 if is_holiday == "Yes" else 0,

            'weather_main_Clouds': 0,
            'weather_main_Drizzle': 0,
            'weather_main_Fog': 0,
            'weather_main_Haze': 0,
            'weather_main_Mist': 0,
            'weather_main_Rain': 0,
            'weather_main_Smoke': 0,
            'weather_main_Snow': 0,
            'weather_main_Squall': 0,
            'weather_main_Thunderstorm': 0
        }

        if weather != "Clear":
            input_data[f'weather_main_{weather}'] = 1

        input_df = pd.DataFrame([input_data])

        # Add any missing columns expected by the model
        for col in columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Arrange columns in the same order as training
        input_df = input_df[columns]

        volume_prediction = volume_model.predict(input_df)[0]
        level_prediction = level_model.predict(input_df)[0]

        st.markdown("---")
        st.subheader("Prediction Results")

        st.metric(
            label="Predicted Traffic Volume",
            value=f"{int(volume_prediction)} vehicles"
        )

        st.metric(
            label="Traffic Category",
            value=level_prediction
        )

        if level_prediction == "Low":
            st.success("🟢 Traffic Condition: Low")

        elif level_prediction == "Normal":
            st.warning("🟡 Traffic Condition: Normal")

        else:
            st.error("🔴 Traffic Condition: High")

    except Exception as e:
        st.error(f"ERROR: {e}")