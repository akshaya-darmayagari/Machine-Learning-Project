import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# -------------------- LIGHT ORANGE MAIN BACKGROUND --------------------
st.markdown("""
<style>

[data-testid="stSidebar"] {
    background-color: #E8F4FF;
}

[data-testid="stAppViewContainer"] {
    background-color: #FFF6ED;
}

h1 {
    color: #0A3D62;
    font-weight: 800;
}

.kpi-card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    text-align: center;
}

.prediction-box {
    padding: 30px;
    border-radius: 15px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODELS --------------------
model_files = {
    "Logistic Regression": "models/LogisticRegression.pkl",
    "Decision Tree": "models/DecisionTree.pkl",
    "Random Forest": "models/RandomForest.pkl",
    "KNN": "models/KNN.pkl",
    "SVM": "models/SVM.pkl",
    "Gradient Boosting": "models/GradientBoosting.pkl",
    "AdaBoost": "models/AdaBoost.pkl",
    "XGBoost": "models/XGBoost.pkl"
}

models = {name: joblib.load(path) for name, path in model_files.items()}
scaler = joblib.load("models/scaler.pkl")

# -------------------- LOAD DATA --------------------
data = pd.read_csv("data/staged/processed_data.csv")

# -------------------- SIDEBAR --------------------
st.sidebar.title("👤 Employee Details")

model_choice = st.sidebar.selectbox("Select ML Model", list(models.keys()))

age = st.sidebar.slider("Age", 18, 60, 30)
income = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
years = st.sidebar.slider("Years at Company", 0, 40, 5)
satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)
worklife = st.sidebar.slider("Work Life Balance (1-4)", 1, 4, 3)
overtime = st.sidebar.selectbox("OverTime", ["No", "Yes"])

# -------------------- TITLE --------------------
st.title("💼 Employee Attrition Prediction Dashboard")
st.markdown("### HR Analytics | Machine Learning Powered Insights")
st.markdown("---")

# -------------------- KPI SECTION --------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <h3>Total Employees</h3>
        <h2>{len(data)}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <h3>Attrition Count</h3>
        <h2>{int(data["Attrition"].sum())}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <h3>Attrition Rate (%)</h3>
        <h2>{round(data["Attrition"].mean()*100,2)}%</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# -------------------- PREDICTION --------------------
model = models[model_choice]
overtime_val = 1 if overtime == "Yes" else 0

input_dict = {col: 0 for col in model.feature_names_in_}
input_dict["Age"] = age
input_dict["MonthlyIncome"] = income
input_dict["YearsAtCompany"] = years
input_dict["JobSatisfaction"] = satisfaction
input_dict["WorkLifeBalance"] = worklife
input_dict["OverTime"] = overtime_val

input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

colA, colB = st.columns(2)

with colA:
    st.subheader("🔮 Prediction Result")

    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-box" style="background-color:#FFEBEE; color:#C62828;">
            🔴 High Attrition Risk <br>
            Probability: {probability:.2f}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box" style="background-color:#E8F5E9; color:#2E7D32;">
            🟢 Employee Likely to Stay <br>
            Probability: {probability:.2f}
        </div>
        """, unsafe_allow_html=True)

with colB:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Attrition Probability %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1565C0"},
            'steps': [
                {'range': [0, 40], 'color': "#C8E6C9"},
                {'range': [40, 70], 'color': "#FFF9C4"},
                {'range': [70, 100], 'color': "#FFCDD2"},
            ],
        }
    ))

    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=False)

st.markdown("---")

# -------------------- DISTRIBUTION & MODEL COMPARISON SIDE BY SIDE --------------------
st.subheader("📊 Distribution & Model Comparison")

colX, colY = st.columns(2)

with colX:
    attrition_counts = data["Attrition"].value_counts().reset_index()
    attrition_counts.columns = ["Attrition", "Count"]

    fig_dist = px.pie(
        attrition_counts,
        names="Attrition",
        values="Count",
        color="Attrition",
        color_discrete_map={0: "#4CAF50", 1: "#F44336"},
        hole=0.4
    )

    fig_dist.update_layout(height=350)
    st.plotly_chart(fig_dist, use_container_width=True)

with colY:
    probabilities = []
    for name, model_obj in models.items():
        prob = model_obj.predict_proba(input_scaled)[0][1]
        probabilities.append((name, prob))

    prob_df = pd.DataFrame(probabilities, columns=["Model", "Probability"])

    fig_compare = px.bar(
        prob_df,
        x="Model",
        y="Probability",
        color="Probability",
        color_continuous_scale="Teal"
    )

    fig_compare.update_layout(height=350)
    st.plotly_chart(fig_compare, use_container_width=True)

st.markdown("---")

# -------------------- FEATURE IMPORTANCE --------------------
if hasattr(model, "feature_importances_"):
    st.subheader("📈 Top 10 Important Features")

    feat_df = pd.DataFrame({
        "Feature": model.feature_names_in_,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(10)

    fig_feat = px.bar(
        feat_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig_feat, use_container_width=True)

st.markdown("---")

# -------------------- CORRELATION HEATMAP --------------------
st.subheader("📉 Feature Correlation Heatmap")

corr = data.corr()

fig_heat = px.imshow(
    corr,
    color_continuous_scale="RdBu",
    aspect="auto"
)

st.plotly_chart(fig_heat, use_container_width=True)