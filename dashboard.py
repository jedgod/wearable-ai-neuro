import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier

# -----------------------------------------------
# TITLE: Wearable AI Dashboard for Neurological Detection
# -----------------------------------------------

st.set_page_config(page_title="Wearable AI Dashboard", layout="wide")
st.title("🧠 AI-Enhanced Wearable Dashboard for Neurological Disorder Detection")

# -----------------------------------------------
# SECTION 1: Simulate synthetic sensor data
# -----------------------------------------------

st.subheader("📊 Simulated Wearable Data Generation")

np.random.seed(42)
n_samples = 1000
time = np.arange(n_samples)
accel_x = np.random.normal(0, 1, n_samples) + np.sin(time / 50)
heart_rate = np.random.normal(80, 10, n_samples)
labels = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # 0 = Healthy, 1 = Disorder

data = pd.DataFrame({
    'time': time,
    'accel_x': accel_x,
    'heart_rate': heart_rate,
    'label': labels
})

# -----------------------------------------------
# SECTION 2: Feature Engineering
# -----------------------------------------------

data['accel_x_smoothed'] = data['accel_x'].rolling(window=10).mean()
data['stride_variability'] = data['accel_x'].diff().abs().rolling(window=20).std()
data = data.dropna()

# -----------------------------------------------
# SECTION 3: Model Training (Gradient Boosting)
# -----------------------------------------------

st.subheader("🧠 Model Training and Prediction")

X = data[['accel_x_smoothed', 'heart_rate', 'stride_variability']]
y = data['label']
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
predictions = model.predict(X)
accuracy = model.score(X, y)

data['prediction'] = predictions

st.success(f"✅ Model trained with accuracy: **{accuracy:.2f}** on simulated data")

# -----------------------------------------------
# SECTION 4: Interactive Visualizations
# -----------------------------------------------

st.header("📈 Patient Monitoring Visualizations")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.line(data, x='time', y='accel_x_smoothed', color=data['prediction'].astype(str),
                   title='Smoothed Accelerometer Signal by Predicted Label',
                   labels={'color': 'Prediction'})
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.line(data, x='time', y='heart_rate', color=data['prediction'].astype(str),
                   title='Heart Rate Over Time by Predicted Label',
                   labels={'color': 'Prediction'})
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------
# SECTION 5: Summary Statistics
# -----------------------------------------------

st.header("📋 Summary Statistics")
st.dataframe(data[['accel_x_smoothed', 'heart_rate', 'stride_variability']].describe().T)

# -----------------------------------------------
# FOOTER
# -----------------------------------------------

st.markdown("---")
st.markdown("Created with ❤️ for neurological disorder detection research using synthetic wearable data.")

