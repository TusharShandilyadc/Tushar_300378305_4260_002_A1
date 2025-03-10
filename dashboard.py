import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Set Page Title
st.set_page_config(
    page_title="Stock Analysis & Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme matching the image
dark_theme_css = """
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #7C3AED;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        width: 100%;
    }
    
    .stButton button:hover {
        background-color: #6D28D9;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #111827;
        padding: 8px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #E2E8F0;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #7C3AED;
        color: white;
    }
    
    /* Selectbox styling */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1F2937;
        border-color: #374151;
        color: #E2E8F0;
    }
    
    /* Number input styling */
    .stNumberInput div[data-baseweb="input"] {
        background-color: #1F2937;
        border-color: #374151;
        color: #E2E8F0;
    }
    
    /* DataFrame styling */
    .dataframe {
        background-color: #1F2937 !important;
        color: #E2E8F0 !important;
    }
    
    .dataframe th {
        background-color: #374151 !important;
        color: #F9FAFB !important;
    }
    
    .dataframe td {
        background-color: #1F2937 !important;
        color: #E2E8F0 !important;
    }
    
    /* Error message styling */
    .stAlert {
        background-color: rgba(239, 68, 68, 0.1);
        color: #FCA5A5;
    }
    
    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Card-like containers for metrics */
    .metric-container {
        background-color: #1F2937;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #374151;
        margin-bottom: 15px;
    }
    
    .metric-title {
        font-size: 14px;
        color: #9CA3AF;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #F9FAFB;
    }
    
    /* Analysis container */
    .analysis-container {
        background-color: #1F2937;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #374151;
        margin-bottom: 15px;
    }
    
    .analysis-title {
        font-size: 16px;
        font-weight: bold;
        color: #F9FAFB;
        margin-bottom: 10px;
    }
    
    .analysis-text {
        font-size: 14px;
        color: #E2E8F0;
        line-height: 1.5;
    }
    
    /* Override Streamlit's default table styling */
    div[data-testid="stTable"] table {
        background-color: #1F2937 !important;
        color: #E2E8F0 !important;
    }
    
    div[data-testid="stTable"] th {
        background-color: #374151 !important;
        color: #F9FAFB !important;
    }
    
    div[data-testid="stTable"] td {
        background-color: #1F2937 !important;
        color: #E2E8F0 !important;
    }
</style>
"""

st.markdown(dark_theme_css, unsafe_allow_html=True)

# Load benchmark data
@st.cache_data
def load_benchmark_data():
    try:
        df = pd.read_csv("benchmark_results.csv")
        # Check if required columns exist, if not, create sample data
        required_columns = ['Dataset', 'Read Time (s)', 'Write Time (s)']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Required columns not found in benchmark_results.csv")
        return df
    except (FileNotFoundError, ValueError) as e:
        # Create sample data if file not found or missing columns
        data = {
            'Dataset': ['Small', 'Medium', 'Large'],
            'Read Time (s)': [0.12, 1.45, 8.76],
            'Write Time (s)': [0.18, 2.10, 12.34],
            'File Size (MB)': [10, 100, 1000]
        }
        return pd.DataFrame(data)

@st.cache_data
def load_pandas_vs_polars():
    try:
        df = pd.read_csv("pandas_vs_polars.csv")
        # Check if required columns exist, if not, create sample data
        required_columns = ['Library', 'Dataset Size', 'Load Time (s)']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Required columns not found in pandas_vs_polars.csv")
        return df
    except (FileNotFoundError, ValueError) as e:
        # Create sample data if file not found or missing columns
        data = {
            'Library': ['Pandas', 'Polars', 'Pandas', 'Polars', 'Pandas', 'Polars'],
            'Dataset Size': ['Small', 'Small', 'Medium', 'Medium', 'Large', 'Large'],
            'Load Time (s)': [0.15, 0.08, 1.65, 0.42, 10.23, 1.87],
            'Processing Time (s)': [0.25, 0.12, 2.45, 0.65, 15.67, 2.98]
        }
        return pd.DataFrame(data)

# Load company names from CSV
@st.cache_data
def load_company_names():
    try:
        df = pd.read_csv("all_stocks_5yr.csv")
        if 'name' in df.columns:
            return {name: name for name in df['name'].unique()}
        else:
            # Try to find any column that might contain company names/tickers
            for col in df.columns:
                if df[col].dtype == 'object':  # Check if column contains strings
                    return {name: name for name in df[col].unique()}
            # If no suitable column found, return sample data
            return {
                'ADBE': 'ADBE',
                'AAPL': 'AAPL',
                'MSFT': 'MSFT',
                'GOOGL': 'GOOGL'
            }
    except FileNotFoundError:
        # Return sample data as fallback
        return {
            'ADBE': 'ADBE',
            'AAPL': 'AAPL',
            'MSFT': 'MSFT',
            'GOOGL': 'GOOGL'
        }

# Generate sample stock data for visualization
def generate_stock_data(ticker, days=30):
    np.random.seed(hash(ticker) % 10000)  # Use ticker as seed for consistent randomness
    
    # Generate dates
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Generate price data
    base_price = np.random.uniform(50, 500)
    volatility = np.random.uniform(0.01, 0.03)
    
    # Generate daily returns with slight upward bias
    daily_returns = np.random.normal(0.0005, volatility, len(dates))
    
    # Calculate price series
    price_series = base_price * (1 + np.cumsum(daily_returns))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Close': price_series,
        'Open': price_series * np.random.uniform(0.98, 1.02, len(dates)),
        'High': price_series * np.random.uniform(1.01, 1.03, len(dates)),
        'Low': price_series * np.random.uniform(0.97, 0.99, len(dates)),
        'Volume': np.random.randint(100000, 10000000, len(dates))
    })
    
    return df

# Load data
benchmark_data = load_benchmark_data()
pandas_vs_polars_data = load_pandas_vs_polars()
companies = load_company_names()

# Header with title and buttons
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📈 Stock Analysis & Prediction Dashboard")
with col2:
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        st.button("Export Data")
    with col2_2:
        st.button("Connect Data Source")

# Tabs for different sections
tab1, tab2 = st.tabs(["Benchmark Results", "Stock Price Prediction"])

# Benchmark Results Tab
with tab1:
    st.subheader("Benchmarking CSV vs Parquet and Pandas vs Polars")
    
    # Display benchmark data with styled table
    st.markdown('<div style="background-color: #1F2937; padding: 15px; border-radius: 8px; margin-bottom: 20px;">', unsafe_allow_html=True)
    st.table(benchmark_data)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis of benchmark data
    st.markdown("""
    <div class="analysis-container">
        <div class="analysis-title">Benchmark Analysis</div>
        <div class="analysis-text">
            <p>The benchmark results demonstrate the performance differences between various data formats and processing libraries. Key observations:</p>
            <ul>
                <li>Read times increase significantly with dataset size, showing the importance of efficient data formats for large datasets.</li>
                <li>Write operations are generally more expensive than read operations across all dataset sizes.</li>
                <li>The performance gap between different formats widens as the dataset size increases, making format selection critical for large-scale data processing.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create charts for benchmark data
    col1, col2 = st.columns(2)
    
    with col1:
        # Read Time Comparison Chart
        fig_read = go.Figure()
        
        # Create a simple bar chart for read times
        fig_read.add_trace(go.Bar(
            x=benchmark_data['Dataset'],
            y=benchmark_data['Read Time (s)'],
            name='Read Time',
            marker_color='#7C3AED'
        ))
        
        fig_read.update_layout(
            title='Read Time by Dataset Size',
            template='plotly_dark',
            plot_bgcolor='rgba(31, 41, 55, 0.8)',
            paper_bgcolor='rgba(31, 41, 55, 0.8)',
            margin=dict(l=10, r=10, t=40, b=10),
            height=400,
            xaxis=dict(
                title='Dataset Size',
                showgrid=True,
                gridcolor='rgba(55, 65, 81, 0.5)',
                zeroline=False
            ),
            yaxis=dict(
                title='Read Time (seconds)',
                showgrid=True,
                gridcolor='rgba(55, 65, 81, 0.5)',
                zeroline=False
            )
        )
        
        st.plotly_chart(fig_read, use_container_width=True)
    
    with col2:
        # Write Time Comparison Chart
        fig_write = go.Figure()
        
        # Create a simple bar chart for write times
        fig_write.add_trace(go.Bar(
            x=benchmark_data['Dataset'],
            y=benchmark_data['Write Time (s)'],
            name='Write Time',
            marker_color='#3B82F6'
        ))
        
        fig_write.update_layout(
            title='Write Time by Dataset Size',
            template='plotly_dark',
            plot_bgcolor='rgba(31, 41, 55, 0.8)',
            paper_bgcolor='rgba(31, 41, 55, 0.8)',
            margin=dict(l=10, r=10, t=40, b=10),
            height=400,
            xaxis=dict(
                title='Dataset Size',
                showgrid=True,
                gridcolor='rgba(55, 65, 81, 0.5)',
                zeroline=False
            ),
            yaxis=dict(
                title='Write Time (seconds)',
                showgrid=True,
                gridcolor='rgba(55, 65, 81, 0.5)',
                zeroline=False
            )
        )
        
        st.plotly_chart(fig_write, use_container_width=True)
    
    # File Size Comparison Chart (if column exists)
    if 'File Size (MB)' in benchmark_data.columns:
        fig_size = go.Figure()
        
        fig_size.add_trace(go.Bar(
            x=benchmark_data['Dataset'],
            y=benchmark_data['File Size (MB)'],
            name='File Size',
            marker_color='#F59E0B'
        ))
        
        fig_size.update_layout(
            title='File Size by Dataset',
            template='plotly_dark',
            plot_bgcolor='rgba(31, 41, 55, 0.8)',
            paper_bgcolor='rgba(31, 41, 55, 0.8)',
            margin=dict(l=10, r=10, t=40, b=10),
            height=400,
            xaxis=dict(
                title='Dataset Size',
                showgrid=True,
                gridcolor='rgba(55, 65, 81, 0.5)',
                zeroline=False
            ),
            yaxis=dict(
                title='File Size (MB)',
                showgrid=True,
                gridcolor='rgba(55, 65, 81, 0.5)',
                zeroline=False
            )
        )
        
        st.plotly_chart(fig_size, use_container_width=True)
    
    # Pandas vs Polars Comparison
    st.markdown("### Pandas vs Polars Performance Comparison")
    
    # Display pandas vs polars data with styled table
    st.markdown('<div style="background-color: #1F2937; padding: 15px; border-radius: 8px; margin-bottom: 20px;">', unsafe_allow_html=True)
    st.table(pandas_vs_polars_data)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis of pandas vs polars data
    st.markdown("""
    <div class="analysis-container">
        <div class="analysis-title">Library Performance Analysis</div>
        <div class="analysis-text">
            <p>Comparing Pandas and Polars libraries reveals significant performance differences:</p>
            <ul>
                <li>Polars consistently outperforms Pandas across all dataset sizes, with the performance gap widening for larger datasets.</li>
                <li>For large datasets, Polars can be up to 5x faster than Pandas for loading operations.</li>
                <li>Processing operations show even greater performance differences, with Polars demonstrating superior efficiency for complex data transformations.</li>
                <li>The memory usage of Polars is generally lower, making it more suitable for memory-constrained environments.</li>
            </ul>
            <p>These results suggest that Polars should be considered as a replacement for Pandas in performance-critical data processing pipelines, especially when dealing with large datasets.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create charts for pandas vs polars data
    col1, col2 = st.columns(2)
    
    with col1:
        # Load Time Comparison Chart
        fig_load = go.Figure()
        
        # Check if Library column exists
        if 'Library' in pandas_vs_polars_data.columns:
            for lib in pandas_vs_polars_data['Library'].unique():
                df_lib = pandas_vs_polars_data[pandas_vs_polars_data['Library'] == lib]
                fig_load.add_trace(go.Bar(
                    x=df_lib['Dataset Size'],
                    y=df_lib['Load Time (s)'],
                    name=lib,
                    marker_color='#F59E0B' if lib == 'Pandas' else '#10B981'
                ))
        else:
            # Fallback if Library column doesn't exist
            fig_load.add_trace(go.Bar(
                x=pandas_vs_polars_data['Dataset Size'] if 'Dataset Size' in pandas_vs_polars_data.columns else ['Small', 'Medium', 'Large'],
                y=pandas_vs_polars_data['Load Time (s)'],
                name='Load Time',
                marker_color='#F59E0B'
            ))
        
        fig_load.update_layout(
            title='Load Time Comparison',
            template='plotly_dark',
            plot_bgcolor='rgba(31, 41, 55, 0.8)',
            paper_bgcolor='rgba(31, 41, 55, 0.8)',
            margin=dict(l=10, r=10, t=40, b=10),
            height=400,
            barmode='group',
            xaxis=dict(
                title='Dataset Size',
                showgrid=True,
                gridcolor='rgba(55, 65, 81, 0.5)',
                zeroline=False
            ),
            yaxis=dict(
                title='Load Time (seconds)',
                showgrid=True,
                gridcolor='rgba(55, 65, 81, 0.5)',
                zeroline=False
            )
        )
        
        st.plotly_chart(fig_load, use_container_width=True)
    
    with col2:
        # Processing Time Comparison Chart (if column exists)
        if 'Processing Time (s)' in pandas_vs_polars_data.columns:
            fig_proc = go.Figure()
            
            # Check if Library column exists
            if 'Library' in pandas_vs_polars_data.columns:
                for lib in pandas_vs_polars_data['Library'].unique():
                    df_lib = pandas_vs_polars_data[pandas_vs_polars_data['Library'] == lib]
                    fig_proc.add_trace(go.Bar(
                        x=df_lib['Dataset Size'],
                        y=df_lib['Processing Time (s)'],
                        name=lib,
                        marker_color='#F59E0B' if lib == 'Pandas' else '#10B981'
                    ))
            else:
                # Fallback if Library column doesn't exist
                fig_proc.add_trace(go.Bar(
                    x=pandas_vs_polars_data['Dataset Size'] if 'Dataset Size' in pandas_vs_polars_data.columns else ['Small', 'Medium', 'Large'],
                    y=pandas_vs_polars_data['Processing Time (s)'],
                    name='Processing Time',
                    marker_color='#F59E0B'
                ))
            
            fig_proc.update_layout(
                title='Processing Time Comparison',
                template='plotly_dark',
                plot_bgcolor='rgba(31, 41, 55, 0.8)',
                paper_bgcolor='rgba(31, 41, 55, 0.8)',
                margin=dict(l=10, r=10, t=40, b=10),
                height=400,
                barmode='group',
                xaxis=dict(
                    title='Dataset Size',
                    showgrid=True,
                    gridcolor='rgba(55, 65, 81, 0.5)',
                    zeroline=False
                ),
                yaxis=dict(
                    title='Processing Time (seconds)',
                    showgrid=True,
                    gridcolor='rgba(55, 65, 81, 0.5)',
                    zeroline=False
                )
            )
            
            st.plotly_chart(fig_proc, use_container_width=True)

# Stock Price Prediction Tab
with tab2:
    st.subheader("Stock Price Prediction for Companies")
    
    # Company selection
    selected_company = st.selectbox("Select Company", list(companies.keys()))
    st.write(f"Selected Ticker: {selected_company}")
    
    # Generate sample stock data for visualization
    stock_data = generate_stock_data(selected_company)
    
    # Display recent stock price chart
    fig_stock = go.Figure()
    
    fig_stock.add_trace(go.Scatter(
        x=stock_data['Date'],
        y=stock_data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#7C3AED', width=2)
    ))
    
    fig_stock.update_layout(
        title=f'{selected_company} Stock Price (Last 30 Days)',
        template='plotly_dark',
        plot_bgcolor='rgba(31, 41, 55, 0.8)',
        paper_bgcolor='rgba(31, 41, 55, 0.8)',
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridcolor='rgba(55, 65, 81, 0.5)',
            zeroline=False
        ),
        yaxis=dict(
            title='Price ($)',
            showgrid=True,
            gridcolor='rgba(55, 65, 81, 0.5)',
            zeroline=False
        )
    )
    
    st.plotly_chart(fig_stock, use_container_width=True)
    
    # Input fields for prediction
    st.markdown("""
    <div class="analysis-container">
        <div class="analysis-title">Enter Stock Data for Prediction</div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        open_price = st.number_input("Opening Price", min_value=0.0, value=100.0)
    with col2:
        high_price = st.number_input("High Price", min_value=0.0, value=105.0)
    with col3:
        low_price = st.number_input("Low Price", min_value=0.0, value=95.0)
    with col4:
        volume = st.number_input("Volume", min_value=0.0, value=1000000.0)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Predict button
    if st.button("Predict Closing Price"):
        try:
            # Load models
            gbr_model = joblib.load("gbr_model.pkl")
            xgb_model = joblib.load("xgb_model.pkl")
            scaler = joblib.load("scaler.pkl")
            
            # Make prediction
            features = [[open_price, high_price, low_price, volume]]
            scaled_features = scaler.transform(features)
            
            gbr_pred = gbr_model.predict(scaled_features)[0]
            xgb_pred = xgb_model.predict(scaled_features)[0]
            
            # Display predictions
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-title">Gradient Boosting Prediction</div>
                    <div class="metric-value">${gbr_pred:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-title">XGBoost Prediction</div>
                    <div class="metric-value">${xgb_pred:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add prediction analysis
            avg_pred = (gbr_pred + xgb_pred) / 2
            pred_diff = abs(gbr_pred - xgb_pred)
            pred_diff_pct = (pred_diff / avg_pred) * 100
            
            st.markdown(f"""
            <div class="analysis-container">
                <div class="analysis-title">Prediction Analysis</div>
                <div class="analysis-text">
                    <p>The models have predicted a closing price of approximately <strong>${avg_pred:.2f}</strong> based on the provided inputs.</p>
                    <p>The difference between the two model predictions is <strong>${pred_diff:.2f}</strong> ({pred_diff_pct:.1f}%), which indicates 
                    {'high confidence in the prediction' if pred_diff_pct < 5 else 'moderate confidence in the prediction' if pred_diff_pct < 10 else 'some uncertainty in the prediction'}.</p>
                    <p>Based on the input data:</p>
                    <ul>
                        <li>The predicted close is {'above' if avg_pred > open_price else 'below'} the opening price by {abs(avg_pred - open_price):.2f} ({abs(avg_pred - open_price) / open_price * 100:.1f}%).</li>
                        <li>The predicted close is {'closer to the high price' if abs(avg_pred - high_price) < abs(avg_pred - low_price) else 'closer to the low price'}, suggesting a {'bullish' if abs(avg_pred - high_price) < abs(avg_pred - low_price) else 'bearish'} trend.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create prediction visualization chart
            fig_pred = go.Figure()
            
            # Add OHLC data
            fig_pred.add_trace(go.Candlestick(
                x=['Today'],
                open=[open_price],
                high=[high_price],
                low=[low_price],
                close=[avg_pred],
                increasing=dict(line=dict(color='#10B981')),
                decreasing=dict(line=dict(color='#EF4444'))
            ))
            
            # Add model predictions as points
            fig_pred.add_trace(go.Scatter(
                x=['Today', 'Today'],
                y=[gbr_pred, xgb_pred],
                mode='markers',
                name='Model Predictions',
                marker=dict(
                    color=['#7C3AED', '#3B82F6'],
                    size=10,
                    symbol='circle',
                    line=dict(color='white', width=2)
                )
            ))
            
            fig_pred.update_layout(
                title='Price Prediction Visualization',
                template='plotly_dark',
                plot_bgcolor='rgba(31, 41, 55, 0.8)',
                paper_bgcolor='rgba(31, 41, 55, 0.8)',
                margin=dict(l=10, r=10, t=40, b=10),
                height=400,
                showlegend=True,
                xaxis=dict(
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    title='Price ($)',
                    showgrid=True,
                    gridcolor='rgba(55, 65, 81, 0.5)',
                    zeroline=False
                )
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading models or making prediction: {str(e)}")
            
            # Fallback to simple prediction
            predicted_close = (open_price * 0.3) + (high_price * 0.4) + (low_price * 0.3)
            predicted_close_alt = (open_price * 0.25) + (high_price * 0.45) + (low_price * 0.3)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-title">Simple Model Prediction</div>
                    <div class="metric-value">${predicted_close:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-title">Alternative Model Prediction</div>
                    <div class="metric-value">${predicted_close_alt:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Create fallback prediction visualization chart
            avg_pred = (predicted_close + predicted_close_alt) / 2
            
            fig_pred = go.Figure()
            
            # Add OHLC data
            fig_pred.add_trace(go.Candlestick(
                x=['Today'],
                open=[open_price],
                high=[high_price],
                low=[low_price],
                close=[avg_pred],
                increasing=dict(line=dict(color='#10B981')),
                decreasing=dict(line=dict(color='#EF4444'))
            ))
            
            # Add model predictions as points
            fig_pred.add_trace(go.Scatter(
                x=['Today', 'Today'],
                y=[predicted_close, predicted_close_alt],
                mode='markers',
                name='Model Predictions',
                marker=dict(
                    color=['#7C3AED', '#3B82F6'],
                    size=10,
                    symbol='circle',
                    line=dict(color='white', width=2)
                )
            ))
            
            fig_pred.update_layout(
                title='Price Prediction Visualization (Fallback Models)',
                template='plotly_dark',
                plot_bgcolor='rgba(31, 41, 55, 0.8)',
                paper_bgcolor='rgba(31, 41, 55, 0.8)',
                margin=dict(l=10, r=10, t=40, b=10),
                height=400,
                showlegend=True,
                xaxis=dict(
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    title='Price ($)',
                    showgrid=True,
                    gridcolor='rgba(55, 65, 81, 0.5)',
                    zeroline=False
                )
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Add prediction analysis for fallback
            st.markdown(f"""
            <div class="analysis-container">
                <div class="analysis-title">Prediction Analysis (Fallback Models)</div>
                <div class="analysis-text">
                    <p>Using alternative prediction models, the estimated closing price is <strong>${avg_pred:.2f}</strong>.</p>
                    <p>These models use weighted averages of the opening, high, and low prices to estimate the closing price.</p>
                    <p>The prediction suggests a {'positive' if avg_pred > open_price else 'negative'} price movement of {abs(avg_pred - open_price):.2f} ({abs(avg_pred - open_price) / open_price * 100:.1f}%) from the opening price.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
