#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Print current working directory and list of files
print("Current Working Directory:", os.getcwd())
print("Files in Current Directory:", os.listdir())

# Load the trained model
logistic_regression_model = joblib.load('best_model_logistic_regression.pkl')

# Load the dataset 
csv_path = "hotel_bookings.csv"
df = pd.read_csv(csv_path)

# Page title
st.title("Hotel Booking Analysis")

# Display the dataset
st.subheader("Dataset")
st.dataframe(df)

# Visualizations
st.subheader("Data Visualizations")

# Histogram for lead time
fig_hist_lead_time = plt.figure(figsize=(10, 6))
sns.histplot(df['lead_time'], bins=30, kde=True)
st.pyplot(fig_hist_lead_time)

# Count plot for canceled and not canceled bookings
fig_count_plot = plt.figure(figsize=(8, 5))
sns.countplot(x='is_canceled', data=df)
st.pyplot(fig_count_plot)

# Heatmap for correlation matrix (excluding non-numeric columns)
numeric_columns = df.select_dtypes(include=['number']).columns
fig_heatmap_corr = plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm")
st.pyplot(fig_heatmap_corr)


# In[ ]:




