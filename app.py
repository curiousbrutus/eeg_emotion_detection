import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
import time
import json
import os
import sys
from scipy import signal

# Add the current directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eeg_processor import EEGProcessor, load_eeg_file

# Page config with custom theme and wider layout
st.set_page_config(
    page_title="EEG Emotion Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {text-align: center; color: #4F8BF9; margin-bottom: 0px;}
    .sub-header {text-align: center; color: #8C8C8C; margin-top: 0px; margin-bottom: 30px;}
    .stButton>button {width: 100%; background-color: #4F8BF9; color: white;}
    .emotion-box {padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    .sidebar-header {color: #4F8BF9; font-weight: bold; margin-top: 20px; margin-bottom: 10px;}
    .status-indicator {height: 10px; width: 10px; border-radius: 50%; display: inline-block; margin-right: 5px;}
    .active {background-color: #00FF00;}
    .inactive {background-color: #FF0000;}
    .section-divider {margin-top: 20px; margin-bottom: 20px; border-bottom: 1px solid #EEEEEE;}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'running' not in st.session_state:
    st.session_state.running = False
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = pd.DataFrame(columns=['time', 'emotion'])
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = datetime.now()
if 'selected_channels' not in st.session_state:
    st.session_state.selected_channels = []
if 'show_all_channels' not in st.session_state:
    st.session_state.show_all_channels = True
if 'eeg_processor' not in st.session_state:
    st.session_state.eeg_processor = EEGProcessor(use_real_device=False)
    st.session_state.eeg_processor.connect()

# Improved helper functions
def get_emotion_color(emotion):
    colors = {
        'Happy': '#FFD700',    # Gold
        'Sad': '#4169E1',      # Royal Blue
        'Neutral': '#808080',  # Gray
        'Excited': '#FF4500',  # Orange Red
        'Calm': '#20B2AA'      # Light Sea Green
    }
    return colors.get(emotion, '#000000')

def get_emotion_emoji(emotion):
    emojis = {
        'Happy': '😊',
        'Sad': '😢',
        'Neutral': '😐',
        'Excited': '😃',
        'Calm': '😌'
    }
    return emojis.get(emotion, '')

def get_emotion_description(emotion):
    descriptions = {
        'Happy': 'A positive state with elevated mood',
        'Sad': 'A negative state with depressed mood',
        'Neutral': 'A balanced emotional state',
        'Excited': 'A highly aroused positive state',
        'Calm': 'A relaxed, peaceful state'
    }
    return descriptions.get(emotion, '')

def save_emotion_data():
    data_file = "emotion_history.json"
    st.session_state.emotion_history.to_json(data_file, orient='records')
    
def load_emotion_data():
    data_file = "emotion_history.json"
    if os.path.exists(data_file):
        try:
            return pd.read_json(data_file)
        except:
            return pd.DataFrame(columns=['time', 'emotion'])
    return pd.DataFrame(columns=['time', 'emotion'])

def start_detection():
    st.session_state.running = True
    st.session_state.eeg_processor.start_acquisition()
    
def stop_detection():
    st.session_state.running = False
    st.session_state.eeg_processor.stop_acquisition()

# Main app header
st.markdown("<h1 class='main-header'>EEG Emotion Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Real-time emotion classification using EEG signals</p>", unsafe_allow_html=True)

# Improved sidebar with better organization
with st.sidebar:
    # Status indicator
    status_color = "active" if st.session_state.running else "inactive"
    st.markdown(f"<div><span class='status-indicator {status_color}'></span> {'Active' if st.session_state.running else 'Inactive'}</div>", unsafe_allow_html=True)
    
    # EEG Device Settings Section
    st.markdown("<h3 class='sidebar-header'>EEG Device Settings</h3>", unsafe_allow_html=True)
    
    use_real_device = st.checkbox("Use Real EEG Device", value=False)
    
    if use_real_device:
        device_path = st.text_input("Device Path/Port", "COM3")
    else:
        device_path = None
        
    if st.button("Connect to Device"):
        with st.spinner("Connecting to EEG device..."):
            st.session_state.eeg_processor = EEGProcessor(use_real_device=use_real_device, device_path=device_path)
            if st.session_state.eeg_processor.connect():
                st.success("Connected successfully!")
            else:
                st.error("Failed to connect to EEG device. Using simulation mode.")
    
    # Control buttons
    st.markdown("<h3 class='sidebar-header'>Detection Controls</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Detection", key="start", disabled=st.session_state.running):
            start_detection()
    with col2:
        if st.button("Stop Detection", key="stop", disabled=not st.session_state.running):
            stop_detection()
    
    # Data Import/Export Section
    st.markdown("<h3 class='sidebar-header'>Data Import/Export</h3>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload EEG Data", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        # Show file preview
        try:
            preview_df = pd.read_csv(uploaded_file)
            st.write("File Preview (First 5 rows):")
            st.dataframe(preview_df.head(5), height=150)
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Process file button
            if st.button("Process EEG File"):
                with st.spinner("Processing EEG data..."):
                    try:
                        # Use the standalone function
                        eeg_data, eeg_channels = load_eeg_file(uploaded_file)
                        
                        if eeg_data is not None and eeg_channels:
                            st.session_state.eeg_processor.disconnect()
                            st.session_state.eeg_processor = EEGProcessor(use_real_device=False)
                            st.session_state.eeg_processor.is_connected = True
                            st.session_state.eeg_processor.channels = eeg_channels
                            st.session_state.eeg_processor.data_buffer = eeg_data
                            
                            # Reset channel selection
                            st.session_state.selected_channels = eeg_channels[:min(3, len(eeg_channels))]
                            st.session_state.show_all_channels = False
                            
                            st.success(f"File processed successfully! Found {len(eeg_channels)} EEG channels.")
                        else:
                            st.error("Failed to extract EEG data from the file.")
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
        except Exception as e:
            st.error(f"Error previewing file: {str(e)}")
    
    if not st.session_state.emotion_history.empty:
        st.download_button(
            label="Download Emotion Data",
            data=st.session_state.emotion_history.to_csv(index=False),
            file_name="emotion_data.csv",
            mime="text/csv"
        )
    
    # Advanced Settings Section
    st.markdown("<h3 class='sidebar-header'>Advanced Settings</h3>", unsafe_allow_html=True)
    update_interval = st.slider("Update Interval (seconds)", 1, 10, 2)
    
    # Show channel selection if channels are available
    if st.session_state.eeg_processor.channels:
        st.markdown("<h3 class='sidebar-header'>Channel Selection</h3>", unsafe_allow_html=True)
        show_all = st.checkbox("Show All Channels", value=st.session_state.show_all_channels)
        
        if show_all != st.session_state.show_all_channels:
            st.session_state.show_all_channels = show_all
        
        if not st.session_state.show_all_channels:
            selected = st.multiselect(
                "Select Channels to Display",
                st.session_state.eeg_processor.channels,
                default=st.session_state.selected_channels
            )
            
            if selected != st.session_state.selected_channels:
                st.session_state.selected_channels = selected

# Main content area
# Create two columns for real-time and historical data
col1, col2 = st.columns([3, 2])

with col1:
    # Current Emotion (highlighted)
    st.subheader("Current Detected Emotion")
    emotion_placeholder = st.empty()
    
    # Real-time EEG Signal with improved visualization
    st.subheader("Real-time EEG Signal")
    signal_settings_col1, signal_settings_col2 = st.columns([1, 1])
    with signal_settings_col1:
        smoothing = st.slider("Signal Smoothing", 1, 20, 1, help="Higher values smooth the signal more")
    with signal_settings_col2:
        amplitude_scale = st.slider("Amplitude Scale", 0.1, 5.0, 1.0, help="Scale the signal amplitude")
    
    real_time_chart = st.empty()
    
    # Add frequency band visualization with radar chart option
    st.subheader("EEG Frequency Bands")
    band_viz_type = st.radio("Visualization Type", ["Bar Chart", "Radar Chart"], horizontal=True)
    bands_chart = st.empty()

with col2:
    st.subheader("Historical Emotion Data")
    
    # Improved date selection
    date_col1, date_col2 = st.columns([1, 1])
    with date_col1:
        start_date = st.date_input("Start Date", date.today() - timedelta(days=7))
    with date_col2:
        end_date = st.date_input("End Date", date.today())
    
    if start_date > end_date:
        st.error("End date must be after start date")
        end_date = start_date
    
    # Timeline of detected emotions
    timeline_chart = st.empty()
    
    # Statistics with improved visualization
    st.subheader("Emotion Statistics")
    stats_viz_type = st.radio("Chart Type", ["Bar Chart", "Pie Chart"], horizontal=True)
    stats_placeholder = st.empty()

# Enhanced function to update real-time visualization
def update_real_time_viz():
    try:
        # Get latest EEG data
        eeg_data = st.session_state.eeg_processor.get_latest_data(num_samples=200)
        
        # Check if eeg_data is None or empty
        if eeg_data is None or eeg_data.size == 0 or len(st.session_state.eeg_processor.channels) == 0:
            real_time_chart.warning("No EEG data available for visualization.")
            return
        
        # EEG Signal Plot - more interactive and responsive
        num_channels = len(st.session_state.eeg_processor.channels)
        available_channels = st.session_state.eeg_processor.channels
        
        # Determine which channels to display
        if st.session_state.show_all_channels:
            display_channels = list(range(num_channels))
            channel_names = available_channels
        else:
            # Get indices of selected channels
            display_channels = [available_channels.index(ch) for ch in st.session_state.selected_channels if ch in available_channels]
            channel_names = st.session_state.selected_channels
        
        if not display_channels:
            real_time_chart.info("Please select at least one channel to display")
            return
        
        # Create plot with multiple traces (one per channel)
        fig = go.Figure()
        
        # Apply color palette for different channels
        colors = px.colors.qualitative.Plotly
        
        for i, channel_idx in enumerate(display_channels):
            if channel_idx < eeg_data.shape[1]:
                # Get channel data
                channel_data = eeg_data[:, channel_idx]
                
                # Apply smoothing if requested
                if smoothing > 1:
                    kernel_size = smoothing
                    kernel = np.ones(kernel_size) / kernel_size
                    channel_data = np.convolve(channel_data, kernel, mode='same')
                
                # Apply amplitude scaling
                channel_data = channel_data * amplitude_scale
                
                # Add trace
                color_idx = i % len(colors)
                fig.add_trace(go.Scatter(
                    y=channel_data, 
                    mode='lines', 
                    name=channel_names[i],
                    line=dict(color=colors[color_idx], width=1.5)
                ))
        
        # Improve layout
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Time (samples)",
            yaxis_title="Amplitude (μV)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        real_time_chart.plotly_chart(fig, use_container_width=True)
        
        # Frequency band plot - with radar chart option
        if len(eeg_data) >= 256:  # Enough data for FFT
            # Calculate for first displayed channel
            first_channel_idx = display_channels[0]
            if first_channel_idx < eeg_data.shape[1]:
                channel_data = eeg_data[:, first_channel_idx]
                
                # Use Welch's method to get power spectral density
                freqs, psd = signal.welch(channel_data, fs=st.session_state.eeg_processor.sample_rate, nperseg=256)
                
                # Define frequency bands
                bands = {
                    'Delta (0.5-4 Hz)': (0.5, 4),
                    'Theta (4-8 Hz)': (4, 8),
                    'Alpha (8-13 Hz)': (8, 13),
                    'Beta (13-30 Hz)': (13, 30),
                    'Gamma (30+ Hz)': (30, 100)
                }
                
                # Calculate power in each band
                band_powers = {}
                for band_name, (low, high) in bands.items():
                    idx = np.logical_and(freqs >= low, freqs <= high)
                    band_powers[band_name] = np.mean(psd[idx])
                
                # Create visualization based on selected type
                if band_viz_type == "Bar Chart":
                    bands_fig = go.Figure()
                    bands_fig.add_trace(go.Bar(
                        x=list(band_powers.keys()),
                        y=list(band_powers.values()),
                        marker_color=['#8B4513', '#4682B4', '#32CD32', '#FFD700', '#FF6347']
                    ))
                    bands_fig.update_layout(
                        height=250,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="Frequency Band",
                        yaxis_title="Power",
                        template="plotly_white"
                    )
                    bands_chart.plotly_chart(bands_fig, use_container_width=True)
                else:  # Radar Chart
                    # Prepare data for radar chart
                    band_names = list(band_powers.keys())
                    values = list(band_powers.values())
                    # Add the first value again to close the loop
                    band_names.append(band_names[0])
                    values.append(values[0])
                    
                    bands_fig = go.Figure()
                    bands_fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=band_names,
                        fill='toself',
                        line=dict(color='rgb(31, 119, 180)')
                    ))
                    bands_fig.update_layout(
                        height=300,
                        polar=dict(
                            radialaxis=dict(visible=True),
                        ),
                        showlegend=False
                    )
                    bands_chart.plotly_chart(bands_fig, use_container_width=True)
        
        # Current Emotion with enhanced visualization
        emotion = st.session_state.eeg_processor.get_current_emotion()
        emotion_color = get_emotion_color(emotion)
        emoji = get_emotion_emoji(emotion)
        description = get_emotion_description(emotion)
        
        # Create a more visually appealing emotion display
        emotion_html = f"""
        <div class="emotion-box" style="background-color: {emotion_color}20; border: 2px solid {emotion_color};">
            <h1 style="color: {emotion_color}; font-size: 3em; margin-bottom: 10px;">{emotion} {emoji}</h1>
            <p style="color: #666; font-size: 1.2em;">{description}</p>
        </div>
        """
        emotion_placeholder.markdown(emotion_html, unsafe_allow_html=True)
        
        # Add to history
        new_data = pd.DataFrame({'time': [datetime.now()], 'emotion': [emotion]})
        st.session_state.emotion_history = pd.concat([st.session_state.emotion_history, new_data], ignore_index=True)
        
        # Save data periodically
        if datetime.now() - st.session_state.last_update_time > timedelta(minutes=1):
            save_emotion_data()
            st.session_state.last_update_time = datetime.now()
    
    except Exception as e:
        st.error(f"Error in real-time visualization: {str(e)}")

# Enhanced function to update historical visualization
def update_historical_viz():
    try:
        # Filter data for the selected date range
        df = st.session_state.emotion_history
        if not df.empty:
            # Ensure time column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'])
            
            # Filter by date range
            filtered_df = df[(df['time'].dt.date >= start_date) & (df['time'].dt.date <= end_date)]
            
            if not filtered_df.empty:
                # Timeline Plot - improved with better color coding
                fig = go.Figure()
                
                # Add emotion trend line
                fig.add_trace(go.Scatter(
                    x=filtered_df['time'], 
                    y=filtered_df['emotion'], 
                    mode='lines+markers',
                    marker=dict(
                        color=[get_emotion_color(e) for e in filtered_df['emotion']],
                        size=8
                    ),
                    line=dict(color='rgba(128, 128, 128, 0.5)', width=1)
                ))
                
                fig.update_layout(
                    height=300, 
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Time",
                    yaxis_title="Emotion",
                    template="plotly_white",
                    hovermode="closest"
                )
                timeline_chart.plotly_chart(fig, use_container_width=True)
                
                # Statistics - with bar or pie chart option
                emotion_counts = filtered_df['emotion'].value_counts().reset_index()
                emotion_counts.columns = ['Emotion', 'Count']
                
                if stats_viz_type == "Bar Chart":
                    stats_fig = go.Figure(data=[
                        go.Bar(
                            x=emotion_counts['Emotion'],
                            y=emotion_counts['Count'],
                            marker_color=[get_emotion_color(e) for e in emotion_counts['Emotion']]
                        )
                    ])
                    stats_fig.update_layout(
                        height=250,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="Emotion",
                        yaxis_title="Frequency",
                        template="plotly_white"
                    )
                else:  # Pie Chart
                    stats_fig = go.Figure(data=[
                        go.Pie(
                            labels=emotion_counts['Emotion'],
                            values=emotion_counts['Count'],
                            marker=dict(colors=[get_emotion_color(e) for e in emotion_counts['Emotion']]),
                            textinfo='label+percent',
                            insidetextorientation='radial'
                        )
                    ])
                    stats_fig.update_layout(
                        height=250,
                        margin=dict(l=0, r=0, t=10, b=0),
                        template="plotly_white",
                        showlegend=False
                    )
                
                stats_placeholder.plotly_chart(stats_fig, use_container_width=True)
                
                # Also display numeric summary
                st.markdown(f"**Total recordings:** {len(filtered_df)}")
                st.markdown(f"**Date range:** {filtered_df['time'].min().date()} to {filtered_df['time'].max().date()}")
                st.markdown(f"**Most common emotion:** {emotion_counts['Emotion'].iloc[0]} ({emotion_counts['Count'].iloc[0]} times)")
            else:
                timeline_chart.info("No data available for selected date range")
                stats_placeholder.info("No statistics available")
        else:
            timeline_chart.info("No historical data available")
            stats_placeholder.info("No statistics available")
    except Exception as e:
        st.error(f"Error in historical visualization: {str(e)}")
        timeline_chart.info("Error processing data")
        stats_placeholder.info("No statistics available")

# Load previous data on startup
if st.session_state.emotion_history.empty:
    st.session_state.emotion_history = load_emotion_data()

# Update visualizations
update_historical_viz()

# Only update real-time data if detection is running
if st.session_state.running:
    update_real_time_viz()

    # Add automatic rerun instead of infinite loop
    time.sleep(update_interval)
    st.rerun()
