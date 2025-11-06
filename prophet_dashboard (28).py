import streamlit as st
import pandas as pd
import json
from prophet import Prophet
from prophet.serialize import model_from_json
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# Page configuration
st.set_page_config(page_title="Restaurant Forecast Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Title
st.title("🍽️ Restaurant Forecasting Dashboard")

# HORIZONTAL LAYOUT - Selection controls at the top
selection_col1, selection_col2, selection_col3 = st.columns([1, 1, 2])

with selection_col1:
    # Restaurant selection
    restaurants = [f"Restaurant_{i+1}" for i in range(6)]
    selected_restaurant = st.selectbox(
        "Restaurant:",
        restaurants,
        key="restaurant_select"
    )

with selection_col2:
    # Metric selection
    metric_options = ["Scontrini", "Totale"]
    selected_metric = st.selectbox(
        "Metric:",
        metric_options,
        key="metric_select"
    )

# Create model key based on selections
metric_lower = selected_metric.lower()  # 'scontrini' or 'totale'
restaurant_num = selected_restaurant.split('_')[1]  # Extract number
model_key = f"{metric_lower}_restaurant_{restaurant_num}"

st.markdown("---")  # Horizontal separator

# Load model function - now with better error handling and connection stability
@st.cache_resource(show_spinner=False)
def load_model(model_path, metadata_path):
    """Load Prophet model and metadata from JSON files"""
    try:
        # Load model - it's already a JSON string from model_to_json()
        with open(model_path, 'r') as f:
            model_json_str = f.read()
        
        # Deserialize directly (no wrapping, no nested 'model' key)
        model = model_from_json(model_json_str)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Function to make forecast
def make_forecast(model, metadata, periods):
    """Generate forecast for specified periods with regressors from metadata"""
    future = model.make_future_dataframe(periods=periods, freq='D')
    
    # Add regressors based on metadata
    # 1. Handle weekly_prepost
    if metadata.get('weekly_prepost', False):
        future['pre_covid'] = 0
        future['post_covid'] = 1
    
    # 2. Handle custom regressors
    if metadata.get('regressors') is not None:
        regressors_list = metadata['regressors']
        for idx, regressor_name in enumerate(regressors_list):
            if idx == 0:
                future[regressor_name] = 1  # First regressor = 1
            else:
                future[regressor_name] = 0  # All others = 0
    
    forecast = model.predict(future)
    return forecast

def plot_trend_holidays(forecast, metadata, selected_metric):
    """
    Create a single plot showing Trend + Holidays (both full history)
    """
    fig = go.Figure()
    
    # Add trend (full history)
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['trend'],
            mode='lines',
            name='Trend',
            line=dict(color='blue', width=3)
        )
    )
    
    # Add holidays if present (full history)
    if metadata.get('holidays', False) and 'holidays' in forecast.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['holidays'],
                mode='lines',
                name='Holidays Effect',
                line=dict(color='red', width=2, dash='dot')
            )
        )
    
    fig.update_layout(
        title="Trend & Holidays Effect",
        xaxis_title='Date',
        yaxis_title=selected_metric,
        hovermode='x unified',
        height=600,
        showlegend=True
    )
    
    return fig

def plot_seasonalities(forecast, metadata, selected_metric):
    """
    Create separate plots for each seasonality component using Prophet's built-in functions
    """
    from plotly.subplots import make_subplots
    from prophet.plot import seasonality_plot_df
    import pandas as pd
    
    # Get model from session state
    model = st.session_state.get('current_model', None)
    
    if model is None:
        st.error("Model not found in session state")
        return go.Figure()
    
    # Collect seasonalities data
    seasonalities = []
    seasonality_names = []
    seasonality_data_list = []
    
    # Check for yearly
    if metadata.get('yearly', False):
        seasonalities.append('yearly')
        seasonality_names.append('Yearly Seasonality')
        
        # Use Prophet's method: create 365 days sequence
        days = pd.date_range(start='2017-01-01', periods=365)
        df_y = seasonality_plot_df(model, days)
        seas = model.predict_seasonal_components(df_y)
        
        data = pd.DataFrame({
            'x_label': list(range(1, 366)),
            'value': seas['yearly'].values
        })
        seasonality_data_list.append(data)
    
    # Check for weekly
    if metadata.get('weekly', False):
        seasonalities.append('weekly')
        seasonality_names.append('Weekly Seasonality')
        
        # Use Prophet's method: create Mon-Sun sequence (start=2017-01-02 is Monday)
        days = pd.date_range(start='2017-01-02', periods=7)
        df_w = seasonality_plot_df(model, days)
        seas = model.predict_seasonal_components(df_w)
        
        data = pd.DataFrame({
            'x_label': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            'value': seas['weekly'].values
        })
        seasonality_data_list.append(data)
    
    # Check for monthly
    if metadata.get('monthly', False):
        seasonalities.append('monthly')
        seasonality_names.append('Monthly Seasonality (Average Pattern)')
        
        # Create 30 days sequence
        days = pd.date_range(start='2017-01-01', periods=30)
        df_m = seasonality_plot_df(model, days)
        seas = model.predict_seasonal_components(df_m)
        
        data = pd.DataFrame({
            'x_label': list(range(1, 31)),
            'value': seas['monthly'].values
        })
        seasonality_data_list.append(data)
    
    # Check for weekly_prepost
    if metadata.get('weekly_prepost', False):
        # Create Mon-Sun sequence
        days = pd.date_range(start='2017-01-02', periods=7)
        
        # For PRE-COVID: activate pre_covid condition
        if 'weekly_pre_covid' in model.seasonalities:
            seasonalities.append('weekly_pre_covid')
            seasonality_names.append('Weekly Pre-COVID')
            
            df_pre = seasonality_plot_df(model, days)
            # Set condition for pre-covid (if conditional seasonality)
            if model.seasonalities['weekly_pre_covid'].get('condition_name'):
                df_pre[model.seasonalities['weekly_pre_covid']['condition_name']] = True
            
            seas_pre = model.predict_seasonal_components(df_pre)
            
            data_pre = pd.DataFrame({
                'x_label': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                'value': seas_pre.get('weekly_pre_covid', seas_pre.get('pre_covid', pd.Series([0]*7))).values
            })
            seasonality_data_list.append(data_pre)
        
        # For POST-COVID: activate post_covid condition
        if 'weekly_post_covid' in model.seasonalities:
            seasonalities.append('weekly_post_covid')
            seasonality_names.append('Weekly Post-COVID')
            
            df_post = seasonality_plot_df(model, days)
            # Set condition for post-covid (if conditional seasonality)
            if model.seasonalities['weekly_post_covid'].get('condition_name'):
                df_post[model.seasonalities['weekly_post_covid']['condition_name']] = True
            
            seas_post = model.predict_seasonal_components(df_post)
            
            data_post = pd.DataFrame({
                'x_label': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                'value': seas_post.get('weekly_post_covid', seas_post.get('post_covid', pd.Series([0]*7))).values
            })
            seasonality_data_list.append(data_post)
    
    if len(seasonalities) == 0:
        # No seasonalities to plot
        fig = go.Figure()
        fig.add_annotation(
            text="No seasonalities found in model",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=400)
        return fig
    
    # Create subplots - one for each seasonality
    n_plots = len(seasonalities)
    
    fig = make_subplots(
        rows=n_plots, 
        cols=1,
        subplot_titles=seasonality_names,
        vertical_spacing=0.15
    )
    
    # Add each seasonality
    for idx, (seasonality, name, data) in enumerate(zip(seasonalities, seasonality_names, seasonality_data_list), start=1):
        
        fig.add_trace(
            go.Scatter(
                x=data['x_label'],
                y=data['value'],
                mode='lines+markers',
                name=name,
                line=dict(color='green', width=2),
                marker=dict(size=6),
                showlegend=False
            ),
            row=idx,
            col=1
        )
        
        # Customize x-axis based on seasonality type
        if seasonality in ['yearly', 'monthly']:
            # Numeric axis for many points
            fig.update_xaxes(
                title_text=f"Day of {'Year' if seasonality == 'yearly' else 'Month'}",
                tickmode='linear',
                dtick=30 if seasonality == 'yearly' else 5,
                row=idx, 
                col=1
            )
        else:
            # Categorical for weekly
            fig.update_xaxes(title_text="Day of Week", row=idx, col=1)
        
        fig.update_yaxes(title_text='Effect', row=idx, col=1)
    
    # Update layout
    fig.update_layout(
        height=450 * n_plots,
        title_text="Seasonality Components",
        showlegend=False
    )
    
    return fig
    
    return fig


# Function to preprocess uploaded CSV
def preprocess_csv(df, selected_metric):
    """
    Preprocess uploaded CSV:
    - Rename 'data' column to 'ds'
    - Rename metric column (Totale/totale or Scontrini/scontrini) to 'y'
    - Convert ds to datetime
    """
    df_processed = df.copy()
    
    # Rename 'data' to 'ds' (handle both 'data' and 'scontrino_data')
    if 'data' in df_processed.columns:
        df_processed = df_processed.rename(columns={'data': 'ds'})
    elif 'scontrino_data' in df_processed.columns:
        df_processed = df_processed.rename(columns={'scontrino_data': 'ds'})
    
    # Find and rename the metric column (case-insensitive)
    metric_lower = selected_metric.lower()  # 'scontrini' or 'totale'
    
    # Look for the column (case-insensitive)
    for col in df_processed.columns:
        if col.lower() == metric_lower:
            df_processed = df_processed.rename(columns={col: 'y'})
            break
    
    # Verify we have both required columns
    if 'ds' not in df_processed.columns or 'y' not in df_processed.columns:
        raise ValueError(f"Could not find required columns. Looking for date column and '{selected_metric}' column")
    
    # Convert ds to datetime
    df_processed['ds'] = pd.to_datetime(df_processed['ds'])
    
    # Keep only ds and y columns
    df_processed = df_processed[['ds', 'y']]
    
    return df_processed

def plot_comparison(model, forecast, actual_data, selected_metric, show_last_months=2):
    """
    Create comparison plot showing:
    - Last 2 months of historical data (white line)
    - Forecast for period where we have actual data (blue line)
    - Actual new data (red stars) - only for the forecast period
    """
    fig = go.Figure()
    
    # Calculate dates
    last_historical_date = model.history['ds'].max()
    cutoff_date = last_historical_date - pd.DateOffset(months=show_last_months)
    
    # Get actual data date range
    actual_start = actual_data['ds'].min()
    actual_end = actual_data['ds'].max()
    
    # Filter historical data to last N months (before forecast period)
    recent_history = model.history[model.history['ds'] >= cutoff_date]
    
    # Plot historical data (white line)
    fig.add_trace(go.Scatter(
        x=recent_history['ds'],
        y=recent_history['y'],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='white', width=2),
        marker=dict(color='white', size=4, line=dict(color='gray', width=1))
    ))
    
    # Filter forecast to only the period covered by actual data (after last_historical_date)
    forecast_comparison = forecast[
        (forecast['ds'] > last_historical_date) & 
        (forecast['ds'] <= actual_end)
    ]
    
    # Plot forecast only for the period we're comparing
    if len(forecast_comparison) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_comparison['ds'],
            y=forecast_comparison['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='blue', width=3)
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_comparison['ds'],
            y=forecast_comparison['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_comparison['ds'],
            y=forecast_comparison['yhat_lower'],
            mode='lines',
            name='Confidence Interval',
            line=dict(width=0),
            fillcolor='rgba(0, 100, 250, 0.2)',
            fill='tonexty',
            showlegend=True
        ))
    
    # Plot actual data - only for dates after last_historical_date (red LINE)
    actual_comparison = actual_data[actual_data['ds'] > last_historical_date]
    if len(actual_comparison) > 0:
        fig.add_trace(go.Scatter(
            x=actual_comparison['ds'],
            y=actual_comparison['y'],
            mode='lines+markers',
            name='Actual Data',
            line=dict(color='red', width=2),
            marker=dict(color='red', size=4)
        ))
    
    # Add vertical line at forecast start
    fig.add_vline(
        x=last_historical_date.value / 10**6,
        line_dash="dash",
        line_color="red",
        annotation_text="Forecast Start",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Forecast vs Actual Comparison",
        xaxis_title='Date',
        yaxis_title=selected_metric,
        hovermode='x unified',
        height=600
    )
    
    return fig

# Function to plot forecast
def plot_forecast(model, forecast, actual_data=None, title="Forecast", show_last_months=2):
    """Create interactive plotly chart for forecast - showing last N months + forecast"""
    fig = go.Figure()
    
    # Calculate cutoff date (last N months from the end of historical data)
    last_historical_date = model.history['ds'].max()
    cutoff_date = last_historical_date - pd.DateOffset(months=show_last_months)
    
    # Filter historical data to last N months
    recent_history = model.history[model.history['ds'] >= cutoff_date]
    
    # Plot recent historical data (real values only)
    fig.add_trace(go.Scatter(
        x=recent_history['ds'],
        y=recent_history['y'],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='white', width=2),
        marker=dict(color='white', size=4, line=dict(color='gray', width=1))
    ))
    
    # Get only future forecast (after last historical date)
    forecast_future = forecast[forecast['ds'] > last_historical_date]
    
    # Plot forecast for future period - bold blue
    if len(forecast_future) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_future['ds'],
            y=forecast_future['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='blue', width=3)
        ))
    
        # Plot confidence interval only for future forecast
        fig.add_trace(go.Scatter(
            x=forecast_future['ds'],
            y=forecast_future['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_future['ds'],
            y=forecast_future['yhat_lower'],
            mode='lines',
            name='Confidence Interval',
            line=dict(width=0),
            fillcolor='rgba(0, 100, 250, 0.2)',
            fill='tonexty',
            showlegend=True
        ))
    
    # Add vertical line to mark the boundary between historical and forecast
    fig.add_vline(
        x=last_historical_date.value / 10**6,  # Convert to milliseconds for plotly
        line_dash="dash",
        line_color="red",
        annotation_text="Forecast Start",
        annotation_position="top"
    )
    
    # Plot actual data if provided (for comparison)
    if actual_data is not None:
        actual_filtered = actual_data[actual_data['ds'] >= cutoff_date]
        fig.add_trace(go.Scatter(
            x=actual_filtered['ds'],
            y=actual_filtered['y'],
            mode='markers',
            name='Actual New Data',
            marker=dict(color='red', size=6, symbol='star')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=selected_metric,
        hovermode='x unified',
        height=600
    )
    
    return fig

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Current Forecast", "🔮 Retrain & Forecast", "📈 Compare Forecast vs Actual", "🔍 Components Analysis"])

# TAB 1: Forecast with existing model
with tab1:
    st.header(f"Forecast for {selected_restaurant} - {selected_metric}")
    
    # Load the model
    model_path = f"models/prophet_model_{model_key}.json"
    metadata_path = f"models/{model_key}_metadata.json"
    model, metadata = load_model(model_path, metadata_path)
    
    # Store model in session state for seasonality plotting
    if model is not None:
        st.session_state.current_model = model
    
    if model is not None and metadata is not None:
        # Forecast horizon selection with compact horizontal buttons
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 3])
        
        with col1:
            if st.button("7d", key="forecast_7", use_container_width=True):
                st.session_state.forecast_days = 7
        with col2:
            if st.button("14d", key="forecast_14", use_container_width=True):
                st.session_state.forecast_days = 14
        with col3:
            if st.button("30d", key="forecast_30", use_container_width=True):
                st.session_state.forecast_days = 30
        with col4:
            if st.button("60d", key="forecast_60", use_container_width=True):
                st.session_state.forecast_days = 60
        with col5:
            if st.button("90d", key="forecast_90", use_container_width=True):
                st.session_state.forecast_days = 90
        
        # Initialize default if not set
        if 'forecast_days' not in st.session_state:
            st.session_state.forecast_days = 30
        
        with col6:
            st.info(f"📅 Horizon: **{st.session_state.forecast_days} days**")
        
        if st.button("📊 Generate Forecast", key="forecast_btn", type="primary"):
            with st.spinner("Generating forecast..."):
                forecast = make_forecast(model, metadata, st.session_state.forecast_days)
                
                # Display plot
                fig = plot_forecast(model, forecast, 
                                   title=f"{selected_restaurant} - {selected_metric} Forecast (Next {st.session_state.forecast_days} days)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast table
                st.subheader("Forecast Data (Last 10 days)")
                forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
                forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                st.dataframe(forecast_display, use_container_width=True)
                
                # Download forecast
                csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast as CSV",
                    data=csv,
                    file_name=f"{model_key}_forecast.csv",
                    mime="text/csv"
                )

# TAB 2: Upload new data and retrain
with tab2:
    st.header("Retrain Model with New Data")
    
    st.info(f"Upload a CSV file with 'data' column (date) and '{selected_metric}' column (value)")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="retrain_upload")
    
    if uploaded_file is not None:
        # Read the uploaded data
        raw_data = pd.read_csv(uploaded_file)
        
        # Preprocess the data
        try:
            new_data = preprocess_csv(raw_data, selected_metric)
            
            st.success(f"✅ Data uploaded successfully! {len(new_data)} rows loaded.")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(new_data.head(10), use_container_width=True)
            
            # Retrain options - compact horizontal layout
            col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 3])
            
            with col1:
                if st.button("7d", key="retrain_7", use_container_width=True):
                    st.session_state.retrain_days = 7
            with col2:
                if st.button("14d", key="retrain_14", use_container_width=True):
                    st.session_state.retrain_days = 14
            with col3:
                if st.button("30d", key="retrain_30", use_container_width=True):
                    st.session_state.retrain_days = 30
            with col4:
                if st.button("60d", key="retrain_60", use_container_width=True):
                    st.session_state.retrain_days = 60
            with col5:
                if st.button("90d", key="retrain_90", use_container_width=True):
                    st.session_state.retrain_days = 90
            
            # Initialize default if not set
            if 'retrain_days' not in st.session_state:
                st.session_state.retrain_days = 30
            
            with col6:
                st.info(f"📅 Horizon: **{st.session_state.retrain_days} days**")
            
            # Load current metadata for retraining
            metadata_path = f"models/{model_key}_metadata.json"
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except:
                st.warning("⚠️ Could not load metadata. Using default settings.")
                metadata = {'weekly': False, 'monthly': False, 'yearly': True, 
                           'weekly_prepost': False, 'holidays': False, 'regressors': None}
            
            if st.button("Retrain Model & Forecast", key="retrain_btn"):
                with st.spinner("Retraining model... This may take a moment."):
                    try:
                        # Check for overlapping dates with existing model
                        last_train_date = model.history['ds'].max()
                        first_new_date = new_data['ds'].min()
                        
                        if first_new_date <= last_train_date:
                            st.warning(f"⚠️ Your new data overlaps with existing training data (model trained until {last_train_date.date()})")
                            st.info("The model will be retrained with ALL the new data. Overlapping dates will use the new values.")
                        
                        # Initialize new model with settings from metadata
                        new_model = Prophet(
                            yearly_seasonality=metadata.get('yearly', True),
                            weekly_seasonality=metadata.get('weekly', True),
                            daily_seasonality=False
                        )
                        
                        # Add weekly_prepost if needed
                        if metadata.get('weekly_prepost', False):
                            new_model.add_seasonality(name='weekly_prepost', period=7, fourier_order=3)
                        
                        # Add regressors if they exist
                        if metadata.get('regressors') is not None:
                            for regressor in metadata['regressors']:
                                new_model.add_regressor(regressor)
                            
                            # Add regressor columns to training data
                            regressors_list = metadata['regressors']
                            for idx, regressor_name in enumerate(regressors_list):
                                if idx == 0:
                                    new_data[regressor_name] = 1
                                else:
                                    new_data[regressor_name] = 0
                        
                        # Add pre/post covid if needed
                        if metadata.get('weekly_prepost', False):
                            new_data['pre_covid'] = 0
                            new_data['post_covid'] = 1
                        
                        # Fit model
                        new_model.fit(new_data)
                        
                        # Make forecast
                        future = new_model.make_future_dataframe(periods=st.session_state.retrain_days, freq='D')
                        
                        # Add regressors to future dataframe
                        if metadata.get('weekly_prepost', False):
                            future['pre_covid'] = 0
                            future['post_covid'] = 1
                        
                        if metadata.get('regressors') is not None:
                            for idx, regressor_name in enumerate(metadata['regressors']):
                                if idx == 0:
                                    future[regressor_name] = 1
                                else:
                                    future[regressor_name] = 0
                        
                        new_forecast = new_model.predict(future)
                        
                        st.success("✅ Model retrained successfully!")
                        
                        # Plot new forecast
                        fig = plot_forecast(new_model, new_forecast,
                                           title=f"New Forecast - {selected_restaurant} - {selected_metric}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show forecast table
                        st.subheader("New Forecast Data (Last 10 days)")
                        forecast_display = new_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
                        forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                        st.dataframe(forecast_display, use_container_width=True)
                        
                        # Save retrained model option
                        from prophet.serialize import model_to_json
                        model_json_str = model_to_json(new_model)
                        st.download_button(
                            label="💾 Download Retrained Model (JSON)",
                            data=model_json_str,
                            file_name=f"prophet_model_{model_key}_retrained.json",
                            mime="application/json"
                        )
                        
                        # Download new forecast
                        csv = new_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
                        st.download_button(
                            label="📥 Download New Forecast as CSV",
                            data=csv,
                            file_name=f"{model_key}_new_forecast.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error retraining model: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error preprocessing data: {str(e)}")
            st.info("Make sure your CSV has 'data' column and either 'Totale' or 'Scontrini' column")

# TAB 3: Compare forecast vs actual
with tab3:
    st.header("Compare Last Forecast vs Actual Data")
    
    st.info(f"Upload actual data (CSV with 'data' and '{selected_metric}' columns) to compare with forecast")
    
    # Load model and metadata
    model_path = f"models/prophet_model_{model_key}.json"
    metadata_path = f"models/{model_key}_metadata.json"
    model, metadata = load_model(model_path, metadata_path)
    
    if model is not None and metadata is not None:
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 3])
        
        with col1:
            if st.button("7d", key="compare_7", use_container_width=True):
                st.session_state.compare_days = 7
        with col2:
            if st.button("14d", key="compare_14", use_container_width=True):
                st.session_state.compare_days = 14
        with col3:
            if st.button("30d", key="compare_30", use_container_width=True):
                st.session_state.compare_days = 30
        with col4:
            if st.button("60d", key="compare_60", use_container_width=True):
                st.session_state.compare_days = 60
        with col5:
            if st.button("90d", key="compare_90", use_container_width=True):
                st.session_state.compare_days = 90
        
        # Initialize default if not set
        if 'compare_days' not in st.session_state:
            st.session_state.compare_days = 30
        
        with col6:
            st.info(f"📅 Horizon: **{st.session_state.compare_days} days**")
        
        uploaded_actual = st.file_uploader(f"Upload actual data (CSV with 'data' and '{selected_metric}' columns)", 
                                          type="csv", key="actual_upload")
        
        if uploaded_actual is not None:
            raw_actual = pd.read_csv(uploaded_actual)
            
            try:
                actual_data = preprocess_csv(raw_actual, selected_metric)
                
                st.success(f"✅ Actual data uploaded! {len(actual_data)} rows loaded.")
                
                if st.button("Generate Comparison", key="compare_btn"):
                    with st.spinner("Generating comparison..."):
                        # Generate forecast
                        forecast = make_forecast(model, metadata, st.session_state.compare_days)
                        
                        # Plot with specialized comparison function
                        fig = plot_comparison(model, forecast, actual_data, selected_metric)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate error metrics
                        # Get last historical date
                        last_historical_date = model.history['ds'].max()
                        
                        # Merge forecast and actual - only for dates after last_historical_date
                        forecast_after = forecast[forecast['ds'] > last_historical_date]
                        actual_after = actual_data[actual_data['ds'] > last_historical_date]
                        
                        comparison_df = forecast_after[['ds', 'yhat']].merge(
                            actual_after[['ds', 'y']], 
                            on='ds', 
                            how='inner'
                        )
                        
                        if len(comparison_df) > 0:
                            # Calculate error columns
                            comparison_df['error'] = comparison_df['yhat'] - comparison_df['y']
                            comparison_df['error_pct'] = (comparison_df['error'] / comparison_df['y'] * 100).round(2)
                            comparison_df.columns = ['Date', 'Forecasted', 'Actual', 'Error', 'Error %']
                            
                            # Show comparison table
                            st.subheader("Detailed Comparison")
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            # Download comparison
                            csv = comparison_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Comparison as CSV",
                                data=csv,
                                file_name=f"{model_key}_comparison.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("⚠️ No overlapping dates between forecast and actual data.")
            
            except Exception as e:
                st.error(f"❌ Error preprocessing data: {str(e)}")
                st.info(f"Make sure your CSV has 'data' column and '{selected_metric}' column")


# TAB 4: Components Analysis
with tab4:
    st.header("Forecast Components Analysis")
    
    st.info("Visualize trend, seasonalities, and holidays effects from the model")
    
    # Load model and metadata
    model_path = f"models/prophet_model_{model_key}.json"
    metadata_path = f"models/{model_key}_metadata.json"
    model, metadata = load_model(model_path, metadata_path)
    
    if model is not None and metadata is not None:
        
        # View selection buttons
        st.write("**Select view:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📈 Trend & Holidays", key="view_trend", use_container_width=True):
                st.session_state.component_view = "trend"
        
        with col2:
            if st.button("🔄 Seasonalities", key="view_seasonality", use_container_width=True):
                st.session_state.component_view = "seasonality"
        
        # Initialize default view
        if 'component_view' not in st.session_state:
            st.session_state.component_view = "trend"
        
        st.info(f"Current view: **{'Trend & Holidays' if st.session_state.component_view == 'trend' else 'Seasonalities'}**")
        
        if st.button("Show Components", key="components_btn"):
            with st.spinner("Generating component analysis..."):
                # Generate forecast with standard horizon (90 days into future)
                forecast = make_forecast(model, metadata, 90)
                
                # Create appropriate plot based on selected view
                if st.session_state.component_view == "trend":
                    fig = plot_trend_holidays(forecast, metadata, selected_metric)
                else:
                    fig = plot_seasonalities(forecast, metadata, selected_metric)
                
                st.plotly_chart(fig, use_container_width=True)

# Footer info
st.markdown("---")
st.caption("💡 Tip: Select restaurant and metric above, then use tabs to explore forecasts, retrain models, compare results, or analyze components.")
