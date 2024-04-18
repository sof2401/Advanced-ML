import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
# Add title
st.title("Project 1 Part A with Streamlit")

columns= ['Item_Weight',
          'Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP',
          'Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Outlet_Sales']

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


# Function for returning a histogram fig
def plot_hist(df, column):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data=df, x=column, ax=ax)
    plt.xticks(rotation=90)
    return fig
# Define the fig object with the function    
fig = plot_hist(df, 'Item_Type')
st.markdown("#### Display a plt plot")
st.pyplot(fig)

def explore_categorical(df, x, fillna = True, placeholder = 'MISSING',
                        figsize = (6,4), order = None):
 
  # Make a copy of the dataframe and fillna 
  temp_df = df.copy()
  # Before filling nulls, save null value counts and percent for printing 
  null_count = temp_df[x].isna().sum()
  null_perc = null_count/len(temp_df)* 100
  # fillna with placeholder
  if fillna == True:
    temp_df[x] = temp_df[x].fillna(placeholder)
  # Create figure with desired figsize
  fig, ax = plt.subplots(figsize=figsize)
  # Plotting a count plot 
  sns.countplot(data=temp_df, x=x, ax=ax, order=order)
  # Rotate Tick Labels for long names
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
  # Add a title with the feature name included
  ax.set_title(f"Column: {x}")
  
  # Fix layout and show plot (before print statements)
  fig.tight_layout()
  plt.show()
    
  return fig, ax


def explore_numeric(df, x, figsize=(6,5) ):
  """Source: https://login.codingdojo.com/m/606/13765/117605"""
  # Making our figure with gridspec for subplots
  gridspec = {'height_ratios':[0.7,0.3]}
  fig, axes = plt.subplots(nrows=2, figsize=figsize,
                           sharex=True, gridspec_kw=gridspec)
  # Histogram on Top
  sns.histplot(data=df, x=x, ax=axes[0])
  # Boxplot on Bottom
  sns.boxplot(data=df, x=x, ax=axes[1])
  ## Adding a title
  axes[0].set_title(f"Column: {x}", fontweight='bold')
  ## Adjusting subplots to best fill Figure
  fig.tight_layout()
  # Ensure plot is shown before message
  plt.show()
  return fig
st.markdown("#### Displaying a plot from explore_numeric function")
fig = explore_numeric(df, 'Item_Outlet_Sales')
st.pyplot(fig)

# Add a selectbox for all possible features
column = st.selectbox(label="Select a column", options=columns)
# Conditional statement to determine which function to use
if df[column].dtype == 'object':
    fig, ax  = explore_categorical(df, column)
else:
    fig = explore_numeric(df, column)
    
st.markdown("#### Displaying appropriate plot based on selected column")
# Display appropriate eda plots
st.pyplot(fig)