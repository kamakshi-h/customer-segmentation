import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(page_title="Customer Segmentation for Marketing", page_icon="🛍️", layout="centered")

st.title("🛍️ Customer Segmentation Model for Marketing Campaigns")
st.write("Group customers into clusters based on their Annual Income and Spending Score using K-Means Clustering.")

# 1. Dataset Generation / Loading
@st.cache_data
def load_data():
    np.random.seed(42)
    # Simulating customer data (e.g., Mall Customers dataset style)
    n_customers = 200
    age = np.random.randint(18, 70, n_customers)
    annual_income = np.random.randint(15, 140, n_customers) # in $k
    spending_score = np.random.randint(1, 100, n_customers) # 1-100 scale
    
    df = pd.DataFrame({
        'Age': age,
        'Annual_Income': annual_income,
        'Spending_Score': spending_score
    })
    return df

df = load_data()

st.subheader("1. Customer Dataset Preview")
st.dataframe(df.head())

# 2. Model Parameters
st.subheader("2. K-Means Clustering Settings")
n_clusters = st.slider("Select number of customer segments (Clusters):", min_value=2, max_value=6, value=5)

# Selecting features for clustering
X = df[['Annual_Income', 'Spending_Score']]

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Training K-Means Model
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

if st.button("Run Customer Segmentation"):
    st.success(f"Customers successfully segmented into {n_clusters} distinct groups for targeted marketing!")
    
    # Display Cluster Summary Statistics
    st.subheader("📊 Segment Profiles (Average Metrics)")
    cluster_summary = df.groupby('Cluster')[['Age', 'Annual_Income', 'Spending_Score']].mean().reset_index()
    st.dataframe(cluster_summary)

    # Visualization
    st.subheader("📈 Customer Segments Visualization")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        x='Annual_Income', 
        y='Spending_Score', 
        hue='Cluster', 
        data=df, 
        palette='viridis', 
        s=100, 
        ax=ax, 
        style='Cluster'
    )
    plt.title('Customer Segments (Income vs Spending Score)')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    st.pyplot(fig)

    # Marketing Strategy Recommendations
    st.subheader("🎯 Targeted Marketing Recommendations")
    st.write("""
    - **High Income, High Spending (VIP Customers):** Focus on premium products, loyalty rewards, and exclusive offers.
    - **High Income, Low Spending (Cautious/Savers):** Target with value-for-money deals and long-term investment schemes.
    - **Low Income, High Spending (Extravagant/Target Risk):** Provide discount codes, EMI options, and budget-friendly promotions.
    - **Low Income, Low Spending (Conservative):** Focus on essential utility products and basic awareness campaigns.
    """)