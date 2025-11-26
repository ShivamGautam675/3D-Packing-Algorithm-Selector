import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model + scaler
# -----------------------------
model = joblib.load("algorithm_selector_model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("3D Packing Algorithm Selector")
st.write("This app predicts the best packing algorithm for irregular 3D parts based on part features.")

st.header("Enter Part Features")

# Input fields
n_parts = st.number_input("Number of Parts", min_value=1, value=10, step=1)
total_vol = st.number_input("Total Volume of All Parts", min_value=1.0, value=50000.0, step=1000.0)
avg_vol = st.number_input("Average Part Volume", min_value=1.0, value=5000.0, step=100.0)
std_vol = st.number_input("Standard Deviation of Part Volume", min_value=0.0, value=1000.0, step=50.0)

mean_dim_x = st.number_input("Mean Dimension X", min_value=1.0, value=20.0, step=1.0)
mean_dim_y = st.number_input("Mean Dimension Y", min_value=1.0, value=15.0, step=1.0)
mean_dim_z = st.number_input("Mean Dimension Z", min_value=1.0, value=12.0, step=1.0)

# Derived features (MUST match training logic)
vol_per_part = total_vol / n_parts
dim_ratio_xy = mean_dim_x / (mean_dim_y + 1e-9)
dim_ratio_xz = mean_dim_x / (mean_dim_z + 1e-9)
vol_variation = std_vol / (avg_vol + 1e-9)

if st.button("Predict Best Algorithm"):
    # Build input data row
    data = {
        "n_parts": n_parts,
        "total_vol": total_vol,
        "avg_vol": avg_vol,
        "std_vol": std_vol,
        "mean_dim_x": mean_dim_x,
        "mean_dim_y": mean_dim_y,
        "mean_dim_z": mean_dim_z,
        "vol_per_part": vol_per_part,
        "dim_ratio_xy": dim_ratio_xy,
        "dim_ratio_xz": dim_ratio_xz,
        "vol_variation": vol_variation
    }

    df_input = pd.DataFrame([data])

    # Scale using the same scaler from training
    X_scaled = scaler.transform(df_input)

    # Predict
    pred_class = int(model.predict(X_scaled)[0])

    algo_map = {
        0: "Algorithm 1 (Largest-First)",
        1: "Algorithm 2 (Flat-First)",
        2: "Algorithm 3 (Random-Greedy)"
    }

    st.subheader("Prediction Result")
    st.success(f"Best Algorithm: {algo_map.get(pred_class, pred_class)}")