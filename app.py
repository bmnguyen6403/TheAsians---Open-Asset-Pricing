import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- Helper Functions ---
def generate_placeholder_data(num_rows):
    """
    Generates placeholder data for demonstration purposes.

    Returns:
        pd.DataFrame: A DataFrame with placeholder data.
    """
    data = {
        'Date': pd.date_range(start='2024-01-01', periods=num_rows, freq='D'),
        'Metric1': np.random.rand(num_rows) * 100,
        'Metric2': np.random.rand(num_rows) * 50,
        'Metric3': np.random.rand(num_rows) * 200,
        'Category': np.random.choice(['A', 'B', 'C', 'D'], num_rows),
        'Value': np.random.randint(1, 100, num_rows),
    }
    return pd.DataFrame(data)

# --- Configuration ---
PAGE_CONFIG = {
    "page_title": "Operations Dashboard",
    "page_icon": "📊",  # You can use an emoji here
    "layout": "wide",  # Use the wide layout
    "initial_sidebar_state": "expanded",  # Keep the sidebar expanded
}
st.set_page_config(**PAGE_CONFIG)

# --- Data Loading ---
# Generate placeholder data
num_rows = 365  # Example: data for one year
df = generate_placeholder_data(num_rows)

# --- Sidebar ---
st.sidebar.title("Filters")
date_range = st.sidebar.date_input("Date Range", [df['Date'].min(), df['Date'].max()], key="date_range")
category_filter = st.sidebar.multiselect("Category", df['Category'].unique(), default=df['Category'].unique())

# --- Filtering ---
# Apply date filter
start_date, end_date = date_range
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Apply category filter
filtered_df = filtered_df[filtered_df['Category'].isin(category_filter)]

# --- Main Content ---
st.title("Operations Dashboard")

# --- KPI Metrics ---
st.header("Key Performance Indicators")
kpi_cols = st.columns(3)  # Adjust the number of columns as needed

# Calculate some example KPIs.  Replace these with your actual KPI calculations.
kpi1_value = filtered_df['Metric1'].mean()
kpi2_value = filtered_df['Metric2'].sum()
kpi3_value = filtered_df['Metric3'].max()

with kpi_cols[0]:
    st.metric("Average Metric 1", f"{kpi1_value:.2f}")
with kpi_cols[1]:
    st.metric("Total Metric 2", f"{kpi2_value:.2f}")
with kpi_cols[2]:
    st.metric("Max Metric 3", f"{kpi3_value:.2f}")

# --- Charts ---
st.header("Data Visualizations")
chart_cols = st.columns(2)  # Create two columns for charts

# Example 1: Line chart
with chart_cols[0]:
    line_chart = alt.Chart(filtered_df).mark_line().encode(
        x='Date',
        y='Metric1',
        tooltip=['Date', 'Metric1']
    ).properties(
        title='Metric 1 Over Time'
    ).interactive()
    st.altair_chart(line_chart, use_container_width=True)

# Example 2: Bar chart
with chart_cols[1]:
    bar_chart = alt.Chart(filtered_df).mark_bar().encode(
        x='Category',
        y='Value',
        color='Category',
        tooltip=['Category', 'Value']
    ).properties(
        title='Value by Category'
    ).interactive()
    st.altair_chart(bar_chart, use_container_width=True)

# Example 3: Area Chart
area_chart = alt.Chart(filtered_df).mark_area().encode(
        x='Date',
        y='Metric2',
        tooltip = ['Date', 'Metric2']
    ).properties(
        title = "Metric 2 Over Time"
    ).interactive()
st.altair_chart(area_chart, use_container_width=True)

# --- Popup ---
# Use st.session_state to control popup visibility
if 'show_popup' not in st.session_state:
    st.session_state.show_popup = True  # S
