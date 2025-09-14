# app.py
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import plotly.express as px

# ---------------------------
# Load and prepare dataset
# ---------------------------
TOP_FEATURES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "PPE"
]

@st.cache_data
def load_data():
    df = pd.read_csv("parkinsons.csv")
    return df

@st.cache_resource
def train_model(df):
    # Use only top 10 features
    X = df[TOP_FEATURES]
    y = df["status"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    # Train model
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, acc, cm, report, TOP_FEATURES

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(
    page_title="Disease Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("<h1 style = 'text-align: center;'>🧠 Parkinson's Disease Prediction App</h>", unsafe_allow_html=True)
st.markdown("---", unsafe_allow_html=True)

# Load dataset
df = load_data()
st.subheader("Dataset Preview (Top Features Only)")
df.index = range(1, len(df) + 1)
st.dataframe(df[["name"] + TOP_FEATURES + ["status"]].head(10))

# Train model
model, acc, cm, report, feature_names = train_model(df)

st.subheader("Model Performance")
st.write(f"✅ Accuracy: **{acc*100:.2f}%**")

# Show confusion matrix
labels = ["Healthy", "Parkinson's"]

fig = px.imshow(
    cm,
    text_auto=True,   # Annotates each cell with value
    color_continuous_scale="Blues",
    x=labels,
    y=labels
)

fig.update_layout(
    xaxis_title="Predicted",
    yaxis_title="Actual"
)

st.plotly_chart(fig)

st.write("📊 Classification Report")
st.json(report)

# ---------------------------
# Prediction form
# ---------------------------
st.subheader("🔮 Predict Disease")

with st.form("prediction_form"):
    input_data = []
    for col in feature_names:
        val = st.number_input(f"{col}", value=0)
        input_data.append(val)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        proba = model.predict_proba(input_array)[0][prediction]

        if prediction == 1:
            st.error(f"⚠️ The model predicts **Parkinson’s Disease** with probability {proba:.2f}")
        else:
            st.success(f"🎉 The model predicts **Healthy** with probability {proba:.2f}")
