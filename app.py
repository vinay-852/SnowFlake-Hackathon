import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load preprocessed data
df4 = pd.read_csv("datasets/Table_4_1_a_preprocessed.csv")
df5 = pd.read_csv("datasets/Table_5_7_preprocessed.csv")

# Load saved models
rf_model_df4 = joblib.load("models/random_forest_df4.pkl")
kmeans_df5 = joblib.load("models/kmeans_df5.pkl")

st.title("Indian Construction Industry Compliance Dashboard")

# Display options for each dataset
dataset_option = st.selectbox("Select Dataset", ["df4 - Compliance Data", "df5 - Material Quality Data"])

if dataset_option == "df4 - Compliance Data":
    st.subheader("Compliance Ratios by State (df4)")
    state = st.selectbox("Select State", df4['State'].unique())
    state_data = df4[df4['State'] == state]
    
    # Display a bar plot for compliance ratios
    fig, ax = plt.subplots()
    sns.barplot(x=state_data['State'], y=state_data['compliance_ratio_1000_pop'], ax=ax)
    ax.set_title('Compliance Ratio for 1000+ Population Habitations')
    ax.set_xlabel('State')
    ax.set_ylabel('Compliance Ratio')
    st.pyplot(fig)
    
    # Display a line plot for compliance ratios over time if 'Year' column exists
    if 'Year' in state_data.columns:
        fig, ax = plt.subplots()
        sns.lineplot(data=state_data, x='Year', y='compliance_ratio_1000_pop', ax=ax)
        ax.set_title('Compliance Ratio for 1000+ Population Habitations Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Compliance Ratio')
        st.pyplot(fig)
    
    # User input for compliance ratios
    compliance_ratio_1000_pop = st.number_input(
        "Enter Compliance Ratio for 1000+ Population Habitations", 
        min_value=0.0, max_value=100.0, 
        value=float(state_data['compliance_ratio_1000_pop'].values[0])
    )
    compliance_ratio_500_pop = st.number_input(
        "Enter Compliance Ratio for 500+ Population Habitations", 
        min_value=0.0, max_value=100.0, 
        value=float(state_data['compliance_ratio_500_pop'].values[0])
    )
    
    # Prepare input for prediction
    X_input = [[compliance_ratio_1000_pop, compliance_ratio_500_pop]]
    prediction = rf_model_df4.predict(X_input)[0]
    
    # Display prediction result
    compliance_status = "High Compliance" if prediction == 1 else "Low Compliance"
    st.write(f"Predicted Compliance Level for {state}: {compliance_status}")

elif dataset_option == "df5 - Material Quality Data":
    st.subheader("Material Quality Ratio by State (df5)")
    state = st.selectbox("Select State", df5['State'].unique())
    state_data = df5[df5['State'] == state]
    
    # Display a box plot for quality material ratios
    fig, ax = plt.subplots()
    sns.boxplot(x=state_data['State'], y=state_data['quality_material_ratio'], ax=ax)
    ax.set_title('Quality Material Ratio by State')
    ax.set_xlabel('State')
    ax.set_ylabel('Quality Material Ratio')
    st.pyplot(fig)
    
    # Display a histogram for quality material ratios
    fig, ax = plt.subplots()
    sns.histplot(state_data['quality_material_ratio'], bins=10, kde=True, ax=ax)
    ax.set_title('Distribution of Quality Material Ratios')
    ax.set_xlabel('Quality Material Ratio')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    # User input for quality material ratio
    quality_material_ratio = st.number_input(
        "Enter Quality Material Ratio", 
        min_value=0.0, max_value=100.0, 
        value=float(state_data['quality_material_ratio'].values[0])
    )
    
    # Cluster prediction using KMeans
    X_input = [[quality_material_ratio]]
    cluster_label = kmeans_df5.predict(X_input)[0]
    
    # Display cluster label result
    st.write(f"Cluster for {state} based on Material Quality Ratio: {cluster_label}")
