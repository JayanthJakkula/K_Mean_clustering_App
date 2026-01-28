import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------
# PAGE SETUP
# ---------------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="centered")

st.title("🟢 Customer Segmentation Dashboard")
st.write("This system uses K-Means Clustering to group customers based on purchasing behavior.")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Wholesale customers data.csv")

df = load_data()

# Only spending features
features = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]

# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------
st.sidebar.header("Clustering Controls")

feature1 = st.sidebar.selectbox("Feature 1", features)
feature2 = st.sidebar.selectbox("Feature 2", features)

k = st.sidebar.slider("Clusters (K)", 2, 10, 3)
random_state = st.sidebar.number_input("Random State", value=42)

run = st.sidebar.button("🟦 Run Clustering")

# Validation
if feature1 == feature2:
    st.warning("Please select two different features.")

# ---------------------------
# RUN CLUSTERING
# ---------------------------
if run and feature1 != feature2:

    X = df[[feature1, feature2]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=random_state)
    df["Cluster"] = model.fit_predict(X_scaled)

    centers = scaler.inverse_transform(model.cluster_centers_)

    # ---------------------------
    # VISUALIZATION
    # ---------------------------
    st.subheader("Cluster Visualization")

    fig = plt.figure()
    plt.scatter(df[feature1], df[feature2], c=df["Cluster"])
    plt.scatter(centers[:,0], centers[:,1], marker="X", s=200)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    st.pyplot(fig)

    # ---------------------------
    # SUMMARY
    # ---------------------------
    st.subheader("Cluster Summary")

    summary = df.groupby("Cluster")[[feature1, feature2]].mean().reset_index()
    summary["Customers"] = df["Cluster"].value_counts().sort_index().values

    st.dataframe(summary)

    # ---------------------------
    # BUSINESS INTERPRETATION
    # ---------------------------
    st.subheader("Business Interpretation")

    for i in summary["Cluster"]:
        f1_avg = summary.loc[summary["Cluster"]==i, feature1].values[0]
        f2_avg = summary.loc[summary["Cluster"]==i, feature2].values[0]

        if f1_avg > X[feature1].mean() and f2_avg > X[feature2].mean():
            st.write(f"🟢 Cluster {i}: High-spending customers")
        elif f1_avg < X[feature1].mean() and f2_avg < X[feature2].mean():
            st.write(f"🟡 Cluster {i}: Budget-conscious customers")
        else:
            st.write(f"🔵 Cluster {i}: Moderate spenders")

    # ---------------------------
    # USER GUIDANCE
    # ---------------------------
    st.info("Customers in the same cluster exhibit similar purchasing behaviour and can be targeted with similar strategies.")
