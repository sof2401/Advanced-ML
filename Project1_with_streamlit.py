import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
# Add title
st.title("Project 1 Part A with Streamlit")

# Loading the data
@st.cache_data
def load_data():
    fpath =  "Data/sales_prediction_2023_modified.csv"
    df = pd.read_csv(fpath)
    return df

df = load_data()
# Display an interactive dataframe
st.header("Displaying a DataFrame")
st.dataframe(df, width=800)

st.subheader("Descriptive Statistics")
if st.button('Show Descriptive Statistics'):
    display_descriptive_stats()

st.subheader("Summary Info")
if st.button('Show Summary Information'):
    display_summary_info()
    
st.subheader("Null Values")
if st.button('Show Null Values'):
    display_null_values()