# dashboard_business/dashboard.py
import streamlit as st
from api_client import get_prediction

st.set_page_config(
    page_title="Agri Yield Intelligence",
    layout="wide",
)

st.title("🌾 Agri Yield Intelligence Platform")
st.caption("Scenario-based crop yield forecasting for agri-input companies")

# ============================================================
# Sidebar Inputs
# ============================================================
st.sidebar.header("Forecast Inputs")

# ---- Explainability toggle
explain_flag = st.sidebar.toggle(
    "Enable Explainability (SHAP)",
    value=False,
    help="Turn ON to see key yield drivers (slower)"
)

# ---- Scenario mode
scenario_mode = st.sidebar.radio(
    "Scenario Mode",
    ["Single Scenario", "Compare Baseline vs Stress"]
)

st.sidebar.subheader("🌦 Climate Conditions")
avg_temp = st.sidebar.slider("Average Temperature (°C)", 10.0, 40.0, 26.0)
max_temp = st.sidebar.slider("Max Temperature (°C)", 15.0, 45.0, 35.0)
min_temp = st.sidebar.slider("Min Temperature (°C)", 5.0, 30.0, 20.0)
rainfall = st.sidebar.slider("Total Rainfall (mm)", 0.0, 1200.0, 600.0)
solar_rad = st.sidebar.slider("Solar Radiation (MJ/m²)", 10.0, 30.0, 20.0)
humidity = st.sidebar.slider("Relative Humidity (%)", 20.0, 100.0, 65.0)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 8.0, 2.5)

st.sidebar.subheader("🌱 Soil Conditions")
soc = st.sidebar.slider("Soil Organic Carbon (%)", 0.2, 1.5, 0.7)

st.sidebar.subheader("🛰 Vegetation Status")
ndvi = st.sidebar.slider("Early Season NDVI", 0.1, 0.9, 0.6)

run = st.sidebar.button("Run Forecast")

# ============================================================
# Payload builders
# ============================================================
def build_payload(
    avg_t, max_t, min_t, rain, rad, hum, wind, soc, ndvi
):
    return {
        "location_id": "IN-PB-LDH",
        "agro_climatic_zone": "Trans-Gangetic Plains",
        "crop": "Wheat",
        "season": "Rabi",
        "climate": {
            "avg_temperature": avg_t,
            "max_temperature": max_t,
            "min_temperature": min_t,
            "total_rainfall": rain,
            "solar_radiation": rad,
            "relative_humidity": hum,
            "wind_speed": wind,
        },
        "soil": {
            "soil_organic_carbon": soc,
        },
        "vegetation": {
            "ndvi_early": ndvi,
        },
    }


# ============================================================
# Main Output
# ============================================================
if run:

    # -------- BASELINE SCENARIO
    baseline_payload = build_payload(
        avg_temp, max_temp, min_temp,
        rainfall, solar_rad, humidity,
        wind_speed, soc, ndvi
    )

    with st.spinner("Running baseline forecast..."):
        baseline_result = get_prediction(
            baseline_payload,
            explain=explain_flag
        )

    # -------- STRESS SCENARIO (AUTO-GENERATED)
    stress_payload = build_payload(
        avg_temp + 2.5,          # heat stress
        max_temp + 3.0,
        min_temp + 2.0,
        rainfall * 0.7,          # drought
        solar_rad,
        humidity * 0.9,
        wind_speed + 1.0,
        soc,
        ndvi * 0.85              # vegetation stress
    )

    if scenario_mode == "Compare Baseline vs Stress":
        with st.spinner("Running stress scenario..."):
            stress_result = get_prediction(
                stress_payload,
                explain=False
            )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================
    st.subheader("📈 Yield Outlook")

    if scenario_mode == "Single Scenario":
        pred = baseline_result["prediction"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Yield (t/ha)", pred["expected_yield"])
        col2.metric("Risk Level", pred["risk_level"])
        col3.metric("Confidence", pred["confidence"])

        st.caption(
            f"Expected range: {pred['yield_lower']} – {pred['yield_upper']} t/ha"
        )

    else:
        base = baseline_result["prediction"]
        stress = stress_result["prediction"]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🌱 Baseline")
            st.metric("Yield (t/ha)", base["expected_yield"])
            st.metric("Risk", base["risk_level"])

        with col2:
            st.markdown("### 🔥 Stress Scenario")
            st.metric("Yield (t/ha)", stress["expected_yield"])
            st.metric("Risk", stress["risk_level"])

        delta = round(stress["expected_yield"] - base["expected_yield"], 2)
        st.warning(f"📉 Yield change under stress: {delta} t/ha")

    # ========================================================
    # SHAP EXPLANATION
    # ========================================================
    if explain_flag and baseline_result.get("explanation"):
        st.subheader("🔍 Key Yield Drivers (Explainable AI)")
        st.bar_chart(baseline_result["explanation"])
