import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from fpdf import FPDF
import locale

# ✅ First Streamlit command must be right after imports
st.set_page_config(page_title="Credit Risk Analysis", layout="wide", initial_sidebar_state="expanded", page_icon="📊")
st.markdown("""
    <style>
        .main {background-color: #ADD8E6;}
    </style>
""", unsafe_allow_html=True)

# Set up language support
locale.setlocale(locale.LC_ALL, '')
LANGUAGES = {"English": "en", "Spanish": "es", "French": "fr", "German": "de"}
st.sidebar.selectbox("🌍 Select Language", options=list(LANGUAGES.keys()))

# Load the trained model
def load_model():
    model_path = r"C:\Users\SANJANA\Downloads\Nse Projects\LCDataDictionary.pkl"  # Corrected file path
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file '{model_path}' not found! Check if the file exists.")
        return None

# Generate 100 credit risk data samples
def generate_credit_risk_data(n=100):
    np.random.seed(42)
    data = {
        "Age": np.random.randint(18, 80, n),
        "Annual Income": np.random.randint(20000, 200000, n),
        "Total Debt": np.random.randint(0, 150000, n),
        "Credit Score": np.random.randint(300, 850, n),
        "Years of Employment": np.random.randint(0, 40, n),
        "Total Assets": np.random.randint(5000, 1000000, n),
        "Loan Amount": np.random.randint(1000, 500000, n),
        "Payment History": np.random.uniform(0, 1, n),
        "Past Defaults": np.random.randint(0, 5, n),
        "Savings": np.random.randint(0, 500000, n),
        "Expense Ratio": np.random.uniform(0.1, 0.9, n),
        "Investment History": np.random.randint(0, 500000, n),
        "Marital Status": np.random.choice(["Single", "Married", "Divorced", "Widowed"], n),
        "Education Level": np.random.choice(["High School", "Bachelor's", "Master's", "PhD"], n),
        "Employment Type": np.random.choice(["Salaried", "Self-Employed", "Unemployed", "Retired"], n),
        "Home Ownership": np.random.choice(["Own", "Rent", "Mortgage", "Other"], n),
        "Loan Purpose": np.random.choice(["Home", "Car", "Business", "Education", "Medical", "Other"], n),
        "Credit Inquiries": np.random.randint(0, 10, n),
        "Region": np.random.choice(["North", "South", "East", "West", "Central"], n),
        "Employment Stability": np.random.uniform(0, 1, n),
        "Loan Tenure": np.random.randint(1, 30, n),
        "Banking Relationship": np.random.choice(["Good", "Average", "Poor"], n)
    }
    df = pd.DataFrame(data)
    df["DTI Ratio"] = round(df["Total Debt"] / df["Annual Income"], 2)
    return df

# Display credit risk data
credit_risk_data = generate_credit_risk_data()
st.write("### Sample Credit Risk Data (First 10 Rows)")
st.dataframe(credit_risk_data.head(10))

# Save data as CSV for further testing
csv_file = "credit_risk_data.csv"
credit_risk_data.to_csv(csv_file, index=False)
st.download_button(label="Download Credit Risk Data", data=open(csv_file, "rb"), file_name=csv_file, mime="text/csv")

# Data Visualization
st.write("### Data Distribution")
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Credit Score Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(credit_risk_data["Credit Score"], bins=30, kde=True, color='blue', ax=ax)
    ax.set_title("Credit Score Distribution")
    st.pyplot(fig)

with col2:
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 4))
    numeric_data = credit_risk_data.select_dtypes(include=[np.number])  # Keep only numeric columns
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)

st.markdown("---")

# Interactive visualization
st.write("### Loan Purpose Distribution")
fig = px.pie(credit_risk_data, names="Loan Purpose", title="Distribution of Loan Purpose")
st.plotly_chart(fig)

st.write("### Employment Type Distribution")
fig = px.bar(credit_risk_data, x="Employment Type", title="Employment Type Distribution", color="Employment Type")
st.plotly_chart(fig)

st.write("### Credit Score vs Loan Amount Over Time")
fig = px.scatter(credit_risk_data, x="Credit Score", y="Loan Amount", animation_frame=credit_risk_data.index.astype(str), color="Employment Type", title="Credit Score vs Loan Amount Trend", size_max=10)
st.plotly_chart(fig)

# Comparison function
st.write("### Compare Credit Risk Metrics")
st.sidebar.subheader("Comparison Panel")
metric1 = st.sidebar.selectbox("Select First Metric", credit_risk_data.columns)
metric2 = st.sidebar.selectbox("Select Second Metric", credit_risk_data.columns)
fig = px.scatter(credit_risk_data, x=metric1, y=metric2, color="Loan Purpose", title=f"Comparison: {metric1} vs {metric2}")
st.plotly_chart(fig)
