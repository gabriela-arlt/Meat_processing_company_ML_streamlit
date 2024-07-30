import streamlit as st
import pandas as pd
import numpy as np
import warnings
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import silhouette_score
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings("ignore")

# Set a random seed for reproducibility
np.random.seed(42)

# Set Streamlit layout to wide
st.set_page_config(layout="wide")

# Center the logo image
col_left, col_center, col_right = st.columns([1, 1, 1.15])

# Display images in each column with inline styling
with col_left:
    st.image("apriori.png", use_column_width=True)
with col_center:
    st.image("arima.png", use_column_width=True)
with col_right:
    st.image("kmenas.png", use_column_width=True)

col_empty = st.columns(1)
with col_empty[0]:
    st.text("..................................................................................................................................... ")

# Load CSS from file
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
load_css("ML_style.css")

# Sidebar content
with st.sidebar:
    # Clear cache button
    if st.button('Clear Cache'):
        st.cache_data.clear()
        st.success('Cache cleared!')
    st.image("tools_1.png", use_column_width=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        with st.spinner('Loading data...'):
            data = pd.read_csv(uploaded_file)

        st.image("apriori_checks.png", use_column_width=True)
        with st.expander("Data Preview"):
            st.write("Data Preview:", data.head())

        # Prepare the data for Apriori
        basket = (data.groupby(['Document_number', 'Product'])['Product']
                  .count().unstack().reset_index().fillna(0)
                  .set_index('Document_number'))
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)
        min_support = 0.1
        frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))

        with st.expander("Frequent Itemsets"):
            st.write("Frequent Itemsets")
            st.dataframe(frequent_itemsets)

        with st.expander("Association Rules"):
            st.write("Association Rules")
            st.dataframe(rules)

if uploaded_file is not None:
    # Relationship Plot
    G = nx.DiGraph()

    for _, row in rules.iterrows():
        G.add_edge(row['antecedents'], row['consequents'], weight=row['lift'])

    pos = nx.spring_layout(G)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1.0, color='white'),  # Custom line width and color
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            showscale=False,  # Disable the color scale bar
            color='red',  # Custom node color
            size=15,  # Custom node size
            line=dict(width=2, color='white')  # Custom node border color
        ),
        textfont=dict(color='white')  # Custom text color
    )

    for node in pos:
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)

        node_trace['text'] += (str(node),)

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=None,
                        showlegend=False,
                        hovermode='closest',
                        annotations=[dict(
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            font=dict(color='white'))],  # Custom annotation color
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False),
                        paper_bgcolor='black',  # Overall layout background color (acts as a border)
                        plot_bgcolor='black',  # Custom plot background color
                        margin=dict(l=14, r=18, t=32, b=18),  # Add margins to create space for the border
                        width=700,  # Adjust width if necessary
                        height=500,  # Adjust height if necessary
                    ))
    col_apriori_left, col_apriori = st.columns([3, 5])
    with col_apriori_left:
        st.image("apriori_frame.png", use_column_width=True)
    with col_apriori:
        st.plotly_chart(fig)

    #### ARIMA ########

    # Daily Sales Analysis
    data['Order_date'] = pd.to_datetime(data['Order_date'])
    daily_sales = data.resample('D', on='Order_date')['Sales'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']

    # Remove outliers using IQR method for ARIMA and K-means
    Q1 = daily_sales['y'].quantile(0.25)
    Q3 = daily_sales['y'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    cleaned_sales = daily_sales[(daily_sales['y'] >= lower_bound) & (daily_sales['y'] <= upper_bound)]

    # Shorten the historical data visualization
    historical_data_length = 365  # Adjust as needed
    daily_sales_viz = cleaned_sales.tail(historical_data_length)

    # Split the data into training and testing sets
    train_size = int(len(cleaned_sales) * 0.9)
    train, test = cleaned_sales[:train_size], cleaned_sales[train_size:]

    # Check the length of the training data
    st.write(f"Length of training data: {len(train)}")

    # Automatic ARIMA parameter selection
    stepwise_model = auto_arima(train['y'], start_p=1, start_q=1,
                                max_p=7, max_q=7, seasonal=False,
                                trace=True, error_action='ignore', 
                                suppress_warnings=True, stepwise=True)

    # Define the forecast period
    forecast_start = len(cleaned_sales) - 30  # Start forecast 30 days before the end of the dataset
    forecast_end = forecast_start + 90  # Extend forecast to 90 days beyond the historical data

    # Define the ARIMA model
    model = ARIMA(cleaned_sales['y'], order=stepwise_model.order)
    # Fit the model
    model_fit = model.fit()

    # Forecast for the selected period
    forecast = model_fit.get_forecast(steps=forecast_end - forecast_start)
    forecast_index = pd.date_range(start=cleaned_sales['ds'].iloc[forecast_start], periods=forecast_end - forecast_start, freq='D')
    forecast_df = pd.DataFrame({'ds': forecast_index, 'yhat': forecast.predicted_mean})

    # Combine historical and forecasted data for visualization
    combined_df = pd.concat([cleaned_sales, forecast_df], ignore_index=True)
    col_arima1, col_arima2 = st.columns([2, 1])
    with col_arima1:
        # Plot historical data and forecasted data using Plotly
        fig = go.Figure()

        # Plot historical data in darker wine red (shortened for visualization)
        fig.add_trace(go.Scatter(x=daily_sales_viz['ds'], y=daily_sales_viz['y'], mode='lines', name='Historical Sales', line=dict(color='white', width=2)))

        # Plot forecasted data in red (overlapping the historical data)
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines+markers', name='ARIMA Forecast', line=dict(color='red', width=2)))

        # Update layout for transparency and labels
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Sales',
            legend=dict(
                x=0, 
                y=1,
                font=dict(
                    color='white'
                )
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                tickcolor='white',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            yaxis=dict(
                tickcolor='white',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            )
        )
        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    with col_arima2:
        st.image("arima_frame.png", use_column_width=True)

######### Kmeans #################

    # Sidebar content for K-means clustering
    with st.sidebar:
        st.image("kmeans_modification1.png", use_column_width=True)

        # Calculate Recency for each partner
        current_date = data['Order_date'].max()
        recency_data = data.groupby('Partner_code')['Order_date'].max().reset_index()
        recency_data['Recency'] = (current_date - recency_data['Order_date']).dt.days

        # Calculate frequency of unique transactions for each partner based on unique Document_number
        frequency_data = data.groupby('Partner_code')['Document_number'].nunique().reset_index(name='Frequency')

        # Prepare data for clustering
        kmeans_data = data.groupby('Partner_code').agg({'Sales': 'sum'}).reset_index()
        kmeans_data = kmeans_data.merge(frequency_data, on='Partner_code')
        kmeans_data = kmeans_data.merge(recency_data[['Partner_code', 'Recency']], on='Partner_code')

        # Remove outliers
        def remove_outliers(df, features):
            for feature in features:
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
            return df

        kmeans_data_clean = remove_outliers(kmeans_data, ['Sales', 'Recency', 'Frequency'])

        # Standardize features using MinMaxScaler
        scaler = MinMaxScaler()
        kmeans_data_scaled = scaler.fit_transform(kmeans_data_clean[['Sales', 'Recency', 'Frequency']])

        # Determine the optimal number of clusters using the Elbow method
        st.write("Determining the optimal number of clusters...")
        sse = []
        silhouette_scores = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(kmeans_data_scaled)
            sse.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(kmeans_data_scaled, kmeans.labels_))

        # Plot the SSE for the Elbow method
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=list(range(2, 11)), y=sse, mode='lines+markers', name='Elbow'))
        fig_elbow.update_layout(title='Elbow Method For Optimal k',
                                xaxis_title='Number of clusters',
                                yaxis_title='Sum of Squared Distances',
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_elbow, use_container_width=True)

        # Plot the Silhouette scores
        fig_silhouette = go.Figure()
        fig_silhouette.add_trace(go.Scatter(x=list(range(2, 11)), y=silhouette_scores, mode='lines+markers', name='Silhouette'))
        fig_silhouette.update_layout(title='Silhouette Scores For Optimal k',
                                     xaxis_title='Number of clusters',
                                     yaxis_title='Silhouette Score',
                                     plot_bgcolor='rgba(0,0,0,0)',
                                     paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_silhouette, use_container_width=True)

        # Select the number of clusters
        optimal_k = st.number_input("Select number of clusters", min_value=2, max_value=10, value=3)

    # K-means clustering with the selected number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans_data_clean['Cluster'] = kmeans.fit_predict(kmeans_data_scaled)

    # Option to download the clustered data
    csv = kmeans_data_clean.to_csv(index=False).encode('utf-8')
    with st.sidebar:
        st.download_button(
            label="Download Clustered Data",
            data=csv,
            file_name='clustered_data.csv',
            mime='text/csv',
        )

    # 3D plot of K-means clusters
    fig_kmeans = px.scatter_3d(kmeans_data_clean, x='Sales', y='Recency', z='Frequency', color='Cluster', 
                               labels={'Sales': 'Total Sales', 'Recency': 'Recency (days)', 'Frequency': 'Transaction Frequency'},
                               color_continuous_scale=px.colors.sequential.Reds)
    fig_kmeans.update_layout(scene=dict(
                                    xaxis=dict(backgroundcolor="black",
                                               gridcolor="white",
                                               showbackground=True,
                                               zerolinecolor="white",),
                                    yaxis=dict(backgroundcolor="black",
                                               gridcolor="white",
                                               showbackground=True,
                                               zerolinecolor="white"),
                                    zaxis=dict(backgroundcolor="black",
                                               gridcolor="white",
                                               showbackground=True,
                                               zerolinecolor="white"),),
                              paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
    # Update marker properties
    fig_kmeans.update_traces(marker=dict(size=5, opacity=0.9))
    col_kmeans1, col_kmeans2 = st.columns([2, 4])
    with col_kmeans1:
        st.image("kmeans_frame.png", use_column_width=True)
    with col_kmeans2:
        st.plotly_chart(fig_kmeans, use_container_width=True)

    # Calculate and display the number of points per cluster
    with st.sidebar:
        cluster_counts = kmeans_data_clean['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        with st.expander("Data Preview"):
             st.write("Number of points per cluster:", cluster_counts.head())

        # Categorize frequency into high, middle, low
        frequency_bins = pd.qcut(kmeans_data_clean['Frequency'], q=3, labels=['Low', 'Middle', 'High'])
        kmeans_data_clean['Frequency Category'] = frequency_bins
        frequency_bins = pd.qcut(kmeans_data_clean['Recency'], q=3, labels=['Low', 'Middle', 'High'])
        kmeans_data_clean['Recency Category'] = frequency_bins
        frequency_bins = pd.qcut(kmeans_data_clean['Sales'], q=3, labels=['Low', 'Middle', 'High'])
        kmeans_data_clean['Sales Category'] = frequency_bins
    
        # Create the summary DataFrame
        cluster_summary = kmeans_data_clean.groupby('Cluster').agg({
            'Frequency Category': lambda x: x.mode()[0],  # Most common frequency category
            'Recency Category': lambda x: x.mode()[0], 
            'Sales Category': lambda x: x.mode()[0]
        }).reset_index()
    
        cluster_summary.columns = ['Cluster Number', 'Frequency Category', 'Recency Category', 'Sales Category']
        with st.expander("Cluster Summary"):
                 st.write("Number of points per cluster:", cluster_summary.head(10))

    ######### DBSCAN #################

    # Add header for DBSCAN section
    #st.header("DBSCAN Clustering")
#
    ## Sidebar content for DBSCAN clustering options
    #with st.sidebar:
    #    st.header("DBSCAN Clustering Options")
    #    eps = st.number_input("Select eps value", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
    #    min_samples = st.number_input("Select min_samples value", min_value=1, max_value=100, value=5)
#
    ## Apply DBSCAN clustering using the scaled data for Frequency, Recency, and Sales
    #dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    #kmeans_data_clean['DBSCAN_Cluster'] = dbscan.fit_predict(kmeans_data_scaled)
    #
    #  # **Add this line to convert 'Cluster' column to string type:**
    #kmeans_data_clean['DBSCAN_Cluster'] = kmeans_data_clean['DBSCAN_Cluster'].astype(object)
#
    ## Label noise points as a separate cluster
    #kmeans_data_clean['DBSCAN_Cluster'] = kmeans_data_clean['DBSCAN_Cluster'].apply(lambda x: 'Noise' if x == -1 else x)
#
    ## 3D plot of DBSCAN clusters
    #fig_dbscan = px.scatter_3d(kmeans_data_clean, x='Sales', y='Recency', z='Frequency', color='DBSCAN_Cluster', 
    #                           title='DBSCAN Clustering',
    #                           labels={'Sales': 'Total Sales', 'Recency': 'Recency', 'Frequency': 'Transaction Frequency'},
    #                           color_discrete_sequence=px.colors.sequential.Reds)
    #fig_dbscan.update_layout(scene=dict(
    #                                xaxis=dict(backgroundcolor="black",
    #                                           gridcolor="white",
    #                                           showbackground=True,
    #                                           zerolinecolor="white",),
    #                                yaxis=dict(backgroundcolor="black",
    #                                           gridcolor="white",
    #                                           showbackground=True,
    #                                           zerolinecolor="white"),
    #                                zaxis=dict(backgroundcolor="black",
    #                                           gridcolor="white",
    #                                           showbackground=True,
    #                                           zerolinecolor="white"),),
    #                         paper_bgcolor='rgba(0,0,0,0)',
    #                         plot_bgcolor='rgba(0,0,0,0)')
    #col_dbscan1, col_dbscan2 = st.columns([2, 4])
    #with col_dbscan1:
    #    st.image("kmeans_frame.png", use_column_width=True)
    #with col_dbscan2:
    #    st.plotly_chart(fig_dbscan, use_container_width=True)

    ## Calculate and display the number of points per cluster for DBSCAN
    #dbscan_cluster_counts = kmeans_data_clean['DBSCAN_Cluster'].value_counts().reset_index()
    #dbscan_cluster_counts.columns = ['Cluster', 'Count']
    #st.write("Number of points per DBSCAN cluster:")
    #st.dataframe(dbscan_cluster_counts)

else:
    st.info('Please upload a CSV file to proceed.')
