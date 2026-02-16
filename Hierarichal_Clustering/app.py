import streamlit as st
import pandas as pd
import joblib
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mall Customer Clustering",
    page_icon="🛍️",
    layout="wide"
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    dt_model = joblib.load("decision_tree_model.pkl")
    scaler = joblib.load("scaler.pkl")
    agglom_model = joblib.load("agglomerative_model.pkl")
    return dt_model, scaler, agglom_model

@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

dt_model, scaler, agglomerative_model = load_models()
df = load_data()

# ---------------- FIX COLUMN NAMES ----------------
# EXACT names used during training
df_model = df.rename(columns={
    'CustomerID': 'Customer ID',
    'Annual Income (k$)': 'Annual Income',
    'Spending Score (1-100)': 'Spending Score'
})

FEATURES = ['Customer ID', 'Annual Income', 'Spending Score']
X = df_model[FEATURES]

# ---------------- SCALE DATA ----------------
X_scaled = scaler.transform(X)

# ---------------- AGGLOMERATIVE CLUSTERING ----------------
agg_model = AgglomerativeClustering(
    n_clusters=agglomerative_model.n_clusters,
    linkage=agglomerative_model.linkage
)

df['Agglomerative_Cluster'] = agg_model.fit_predict(X_scaled)

# ---------------- DECISION TREE PREDICTION ----------------
df['Predicted_Cluster'] = dt_model.predict(X_scaled)

# ---------------- UI ----------------
st.markdown("<h1 style='text-align:center;'>🛍️ Mall Customer Clustering</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Agglomerative Clustering + Decision Tree Prediction</p>",
    unsafe_allow_html=True
)

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Customer Details")

    customer_id = st.number_input("Customer ID", min_value=1, step=1)
    income = st.slider("Annual Income", 10, 200, 50)
    spending = st.slider("Spending Score", 1, 100, 50)

    if st.button("🚀 Predict Cluster", use_container_width=True):

        input_df = pd.DataFrame(
            [[customer_id, income, spending]],
            columns=FEATURES
        )

        input_scaled = scaler.transform(input_df)
        predicted_cluster = dt_model.predict(input_scaled)[0]

        st.success(f"Predicted Cluster: {predicted_cluster}")

# ---------------- VISUALIZATION ----------------
st.markdown("---")
st.subheader("📊 Cluster Visualization")

fig = px.scatter(
    df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    color=df['Predicted_Cluster'].astype(str),
    title="Customer Segments (Decision Tree)"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Hierarchical Clustering + Decision Tree Classification</p>",
    unsafe_allow_html=True
)
