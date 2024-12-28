# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import numpy as np
from src import LTVSyntheticData
from src import LTVexploratory
import plotly.express as px
import plotly.tools as tls  
import os  
import streamlit as st

st.title("Customer Lifetime Value Analysis using LTVision")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")
n_users = st.sidebar.slider("Number of Users", min_value=1000, max_value=100000, value=20000)
random_seed = st.sidebar.number_input("Random Seed", value=42)

# Set days limit to either 30 or 60 days
days_limit = st.sidebar.radio("Days Limit for Analysis", [30, 60], index=0)  # Default to 30 days

# Generate Synthetic Data
st.header("Synthetic Data Generation")
synth_data_gen = LTVSyntheticData(n_users=n_users, random_seed=random_seed)
customer_table = synth_data_gen.get_customers_data()
event_table = synth_data_gen.get_events_data()

# Display Customer and Event Data
st.subheader("Customer Data")
st.write("Displays the first few rows of the generated customer data, including registration dates and demographics.")
st.write(customer_table.head())

st.subheader("Event Data")
st.write("Displays the first few rows of the generated event data, including purchase dates and values.")
st.write(event_table.head())

# Perform Exploratory Analysis
st.header("Exploratory Analysis")
da = LTVexploratory(
    customer_table,
    event_table,
    registration_time_col='registration_date',
    event_time_col='event_date',
    event_name_col='event_name',
    value_col='value',
    rounding_precision=1
)

# Purchase Frequency Distribution
st.subheader("Purchase Frequency Distribution")
st.write("Shows the distribution of customers by the number of purchases they made within the selected time frame.")
fig_purchase_freq, data_purchase_freq = da.plot_purchases_distribution(days_limit=days_limit, truncate_share=0.999)
plotly_fig_purchase_freq = tls.mpl_to_plotly(fig_purchase_freq.fig)  # Convert Seaborn plot to Plotly figure
st.plotly_chart(plotly_fig_purchase_freq)

# Top Spenders' Contribution to Total Revenue
st.subheader("Top Spenders' Contribution to Total Revenue")
st.write("Visualizes the cumulative revenue contribution of the top spenders within the selected time frame.")
fig_revenue_pareto, data_revenue_pareto = da.plot_revenue_pareto(days_limit=days_limit)
plotly_fig_revenue_pareto = tls.mpl_to_plotly(fig_revenue_pareto.fig)  # Convert Seaborn plot to Plotly figure
st.plotly_chart(plotly_fig_revenue_pareto)

# Time to First Purchase
st.subheader("Time to First Purchase")
st.write("Displays the distribution of customers by the number of days it took for their first purchase.")
fig_time_to_first_purchase, data_time_to_first_purchase = da.plot_customers_histogram_per_conversion_day(days_limit=days_limit)
plotly_fig_time_to_first_purchase = tls.mpl_to_plotly(fig_time_to_first_purchase.fig)  # Convert Seaborn plot to Plotly figure
st.plotly_chart(plotly_fig_time_to_first_purchase)

# Correlation Between Short-Term and Long-Term Revenue
st.subheader("Correlation Between Short-Term and Long-Term Revenue")
st.write("Shows the correlation between revenue generated in the short term (e.g., 7 days) and long term (e.g., 30 or 60 days).")
fig_correlation, data_correlation = da.plot_early_late_revenue_correlation(days_limit=days_limit)

# Convert the correlation data into a Plotly heatmap
plotly_fig_correlation = px.imshow(
    data_correlation,
    labels=dict(x="Days Since Registration", y="Days Since Registration", color="Correlation"),
    x=data_correlation.columns,
    y=data_correlation.index,
    color_continuous_scale="Viridis",  # Choose a color scale (e.g., "Viridis", "Plasma", "Inferno")
)

# Highlight the day when correlation drops below 50%
correlation_threshold = 0.5
day_below_threshold = 0
for i in range(len(data_correlation.columns)):
    if data_correlation.iloc[i, 0] < correlation_threshold:
        day_below_threshold = data_correlation.columns[i]
        break

# Add a vertical line to highlight the threshold
plotly_fig_correlation.add_vline(
    x=day_below_threshold,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Correlation < {correlation_threshold}",
    annotation_position="top right"
)

# Display the Plotly heatmap in Streamlit
st.plotly_chart(plotly_fig_correlation)

# Purchaser Flow Over Time
st.subheader("Purchaser Flow Over Time")
fig_customer_flow, data_customer_flow = da.plot_paying_customers_flow(days_limit=days_limit, early_limit=7, spending_breaks={}, end_spending_breaks={})
st.plotly_chart(fig_customer_flow)  # This is already a Plotly figure

# pLTV Opportunity Size Estimation
st.header("pLTV Opportunity Size Estimation")

# Scenario 1: User identification before first purchase
st.subheader("Scenario 1: User Identification Before First Purchase")
st.write("Estimates revenue impact for businesses where user identification happens before the first purchase (e.g., mobile apps).")
data_ltv_impact_mobile = da.estimate_ltv_impact(
    days_limit=days_limit,
    early_limit=7,
    spending_breaks={},
    is_mobile=True
)
st.write(data_ltv_impact_mobile)

# Scenario 2: User identification relies on first purchase
st.subheader("Scenario 2: User Identification Relies on First Purchase")
st.write("Estimates revenue impact for businesses where user identification relies on the first purchase (e.g., eCommerce).")
data_ltv_impact_ecomm = da.estimate_ltv_impact(
    days_limit=days_limit,
    early_limit=7,
    spending_breaks={},
    is_mobile=False
)
st.write(data_ltv_impact_ecomm)


# DeepSeek LLM Integration for Summary of Insights
st.header("Summary of Insights & Recommendations")

# Fetch the DeepSeek API key from environment variables
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")  # Fetch the key from the environment

if not deepseek_api_key:
    st.error("DeepSeek API key not found in environment variables. Please set the 'DEEPSEEK_API_KEY' environment variable.")
else:
    # Initialize the OpenAI client
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

    # Prepare the prompt for DeepSeek
    prompt = """
    Summarize the key insights from the LTVision analysis in bullet points. Focus on the following:
    1. Purchase frequency distribution.
    2. Top spenders' contribution to total revenue.
    3. Time to first purchase.
    4. Correlation between short-term and long-term revenue.
    5. Purchaser flow over time.
    6. pLTV opportunity size estimation for both scenarios.
    """

    # Call the DeepSeek API
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes insights in clear and concise bullet points."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        # Extract the generated summary
        summary = response.choices[0].message.content

        # Display the summary in Streamlit
        st.write(summary)

    except Exception as e:
        st.error(f"An error occurred while generating the summary: {e}")


# Footer
st.write("Thank you Meta Incubator Team for LTVision!  \nLTVision is licensed under the BSD-style license, as found in the [LICENSE file](https://github.com/facebookincubator/LTVision/blob/main/LICENSE.md).")
