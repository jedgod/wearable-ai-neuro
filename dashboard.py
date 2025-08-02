import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import GradientBoostingClassifier

# Streamlit Dashboard for Wearable AI Analysis
st.title("AI-Enhanced Wearable Dashboard for Neurological Disorder Detection")

# Simulate synthetic data
np.random.seed(42)
n_samples = 1000
time = np.arange(n_samples)
accel_x = np.random.normal(0, 1, n_samples) + np.sin(time / 50)
heart_rate = np.random.normal(80, 10, n_samples)
labels = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

data = pd.DataFrame({
    'time': time,
    'accel_x': accel_x,
    'heart_rate': heart_rate,
    'label': labels
})

# Preprocess data
data['accel_x_smoothed'] = data['accel_x'].rolling(window=10).mean()
data['stride_variability'] = data['accel_x'].diff().abs().rolling(window=20).std()
data = data.dropna()

# Train model
X = data[['accel_x_smoothed', 'heart_rate', 'stride_variability']]
y = data['label']
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Display model predictions
st.header("Model Predictions")
predictions = model.predict(X)
data['prediction'] = predictions
st.write(f"Prediction Accuracy (on training data): {model.score(X, y):.2f}")

# Interactive visualizations
st.header("Patient Data Visualizations")
fig1 = px.line(data, x='time', y='accel_x_smoothed', color='prediction',
               title='Smoothed Accelerometer Data (Predicted Labels)')
st.plotly_chart(fig1)

fig2 = px.line(data, x='time', y='heart_rate', color='prediction',
               title='Heart Rate Over Time (Predicted Labels)')
st.plotly_chart(fig2)

# Display summary statistics
st.header("Summary Statistics")
st.write(data[['accel_x_smoothed', 'heart_rate', 'stride_variability']].describe())
