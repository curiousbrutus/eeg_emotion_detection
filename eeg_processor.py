import numpy as np
import pandas as pd
from scipy import signal
import pickle
import os
from datetime import datetime
import time
from threading import Thread, Lock

class EEGProcessor:
    def __init__(self, use_real_device=False, device_path=None):
        """
        Initialize the EEG processor
        
        Parameters:
        -----------
        use_real_device : bool
            Whether to use a real EEG device or simulated data
        device_path : str
            Path to the EEG device (e.g., COM port)
        """
        self.use_real_device = use_real_device
        self.device_path = device_path
        self.is_connected = False
        self.data_buffer = np.array([])
        self.buffer_size = 1000  # 1000 samples (~4 seconds at 256 Hz)
        self.sample_rate = 256   # Common EEG sample rate
        self.emotion_model = self._load_model()
        self.current_emotion = "Neutral"
        self.lock = Lock()
        
        # EEG channel names (standard 10-20 system)
        self.channels = [] # Initialize as empty list
        
        # Available emotions
        self.emotions = ['Happy', 'Sad', 'Neutral', 'Excited', 'Calm']
        
        # Thread for continuous data acquisition
        self.acquisition_thread = None
        self.running = False
        
    def _load_model(self):
        """Load pre-trained emotion classification model or create dummy"""
        model_path = "emotion_model.pkl"
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        # Return dummy model if no model file exists
        return DummyEmotionModel()
    
    def connect(self):
        """Connect to EEG device or initialize simulated data"""
        if self.use_real_device:
            try:
                # Code to connect to real EEG device would go here
                # This is a placeholder for actual device connection code
                print(f"Connecting to EEG device at {self.device_path}")
                time.sleep(1)  # Simulate connection time
                self.is_connected = True
                print("Connected to EEG device")
            except Exception as e:
                print(f"Failed to connect to EEG device: {e}")
                self.is_connected = False
                self.use_real_device = False  # Fall back to simulation
        else:
            self.is_connected = True
            print("Using simulated EEG data")
        
        return self.is_connected
    
    def disconnect(self):
        """Disconnect from EEG device"""
        if self.use_real_device and self.is_connected:
            # Code to disconnect from real device would go here
            pass
        
        self.is_connected = False
        self.stop_acquisition()
        print("Disconnected from EEG device")
    
    def _generate_simulated_eeg(self, num_samples=100):
        """
        Generate simulated EEG data with realistic properties
        
        Parameters:
        -----------
        num_samples : int
            Number of samples to generate
            
        Returns:
        --------
        numpy.ndarray
            Simulated EEG data
        """
        # Create base signal: combination of different frequency bands
        t = np.linspace(0, num_samples/self.sample_rate, num_samples)
        
        # Generate different brain wave components
        # Delta waves (0.5-4 Hz)
        delta = np.sin(2 * np.pi * 2 * t) * 20
        
        # Theta waves (4-8 Hz) 
        theta = np.sin(2 * np.pi * 6 * t) * 10
        
        # Alpha waves (8-13 Hz)
        alpha = np.sin(2 * np.pi * 10 * t) * 15
        
        # Beta waves (13-30 Hz)
        beta = np.sin(2 * np.pi * 20 * t) * 5
        
        # Combine waveforms with random variations
        base_signal = delta + theta + alpha + beta
        
        # Add some random noise
        noise = np.random.normal(0, 5, num_samples)
        
        # Combine everything
        eeg_signal = base_signal + noise
        
        return eeg_signal
    
    def get_eeg_data(self, num_samples=100):
        """
        Get EEG data either from device or simulation
        
        Parameters:
        -----------
        num_samples : int
            Number of samples to get
            
        Returns:
        --------
        numpy.ndarray
            EEG data
        """
        if not self.is_connected:
            return np.zeros(num_samples)
        
        if self.use_real_device:
            # Code to get data from real device would go here
            # This is a placeholder for actual data acquisition
            #return self._generate_simulated_eeg(num_samples)  # Temporary fallback
            
            # Simulate reading from multiple channels
            num_channels = len(self.channels)
            eeg_data = np.random.randn(num_samples, num_channels)  # Generate random data for each channel
            return eeg_data
        else:
            #return self._generate_simulated_eeg(num_samples)
            
            # Simulate reading from multiple channels
            num_channels = len(self.channels)
            eeg_data = np.random.randn(num_samples, num_channels)  # Generate random data for each channel
            return eeg_data
    
    def start_acquisition(self):
        """Start continuous data acquisition in a separate thread"""
        if self.running:
            return
        
        self.running = True
        self.acquisition_thread = Thread(target=self._acquisition_loop)
        self.acquisition_thread.daemon = True
        self.acquisition_thread.start()
    
    def stop_acquisition(self):
        """Stop continuous data acquisition"""
        self.running = False
        if self.acquisition_thread:
            self.acquisition_thread.join(timeout=1.0)
            self.acquisition_thread = None
    
    def _acquisition_loop(self):
        """Background thread for continuous data acquisition"""
        while self.running:
            # Get new batch of data
            #new_data = self.get_eeg_data(num_samples=int(self.sample_rate/10))  # 0.1 sec worth of data
            
            # Simulate reading from multiple channels
            num_channels = len(self.channels)
            new_data = self.get_eeg_data(num_samples=int(self.sample_rate/10))  # 0.1 sec worth of data
            
            # Add to buffer
            with self.lock:
                if len(self.data_buffer) == 0:
                    self.data_buffer = new_data
                else:
                    self.data_buffer = np.concatenate([self.data_buffer, new_data], axis=0)
                
                # Keep buffer at specified size
                if len(self.data_buffer) > self.buffer_size:
                    self.data_buffer = self.data_buffer[-self.buffer_size:]
                
                # Process data and classify emotion
                if len(self.data_buffer) >= 256:  # At least 1 second of data
                    self._process_and_classify()
            
            time.sleep(0.1)  # Sleep to prevent CPU overuse
    
    def _extract_features(self, eeg_data):
        """
        Extract relevant features from EEG data
        
        Parameters:
        -----------
        eeg_data : numpy.ndarray
            EEG data (num_samples x num_channels)
            
        Returns:
        --------
        dict
            Dictionary of extracted features
        """
        features = {}
        
        # Check if eeg_data is empty or has no channels
        if eeg_data is None or eeg_data.size == 0 or len(eeg_data.shape) < 2:
            print("Warning: No EEG data available for feature extraction.")
            return features
        
        num_channels = eeg_data.shape[1]
        
        # Calculate power in different frequency bands for each channel
        freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        for i in range(num_channels):
            channel_data = eeg_data[:, i]
            
            # Use Welch's method to estimate power spectral density
            freqs, psd = signal.welch(channel_data, fs=self.sample_rate, nperseg=256)
            
            for band_name, band_range in freq_bands.items():
                # Find the indices corresponding to the frequency range
                idx_band = np.logical_and(freqs >= band_range[0], freqs <= band_range[1])
                # Calculate power in band
                power_band = np.mean(psd[idx_band])
                features[f'channel_{i}_{band_name}_power'] = power_band
            
            # Add statistical features
            features[f'channel_{i}_mean'] = np.mean(channel_data)
            features[f'channel_{i}_std'] = np.std(channel_data)
            features[f'channel_{i}_kurtosis'] = np.mean((channel_data - np.mean(channel_data))**4) / (np.std(channel_data)**4)
            features[f'channel_{i}_skewness'] = np.mean((channel_data - np.mean(channel_data))**3) / (np.std(channel_data)**3)
        
        return features
    
    def _process_and_classify(self):
        """Process EEG data and classify emotion"""
        # Extract features
        features = self._extract_features(self.data_buffer)
        
        # Use model to predict emotion
        self.current_emotion = self.emotion_model.predict(features)
    
    def get_current_emotion(self):
        """Get the current classified emotion"""
        return self.current_emotion
    
    def get_latest_data(self, num_samples=100):
        """Get the latest N samples of EEG data"""
        with self.lock:
            if len(self.data_buffer) == 0:
                return np.zeros((num_samples, len(self.channels)))  # Return 2D array
            elif len(self.data_buffer) < num_samples:
                # Pad with zeros if not enough data
                padding = np.zeros((num_samples - len(self.data_buffer), len(self.channels)))
                return np.concatenate([self.data_buffer, padding], axis=0)
            else:
                return self.data_buffer[-num_samples:]

class DummyEmotionModel:
    """Simple dummy model for emotion classification when no real model is available"""
    def __init__(self):
        self.emotions = ['Happy', 'Sad', 'Neutral', 'Excited', 'Calm']
        self.weights = {
            'alpha_power': {'Happy': 0.5, 'Sad': -0.3, 'Neutral': 0.1, 'Excited': -0.2, 'Calm': 0.8},
            'beta_power': {'Happy': 0.2, 'Sad': -0.1, 'Neutral': 0.0, 'Excited': 0.7, 'Calm': -0.4},
            'theta_power': {'Happy': 0.1, 'Sad': 0.4, 'Neutral': 0.0, 'Excited': 0.1, 'Calm': 0.2},
            'delta_power': {'Happy': -0.1, 'Sad': 0.3, 'Neutral': 0.1, 'Excited': -0.1, 'Calm': 0.1},
            'gamma_power': {'Happy': 0.3, 'Sad': -0.2, 'Neutral': 0.0, 'Excited': 0.6, 'Calm': -0.3}
        }
    
    def predict(self, features):
        """
        Predict emotion based on features
        
        Parameters:
        -----------
        features : dict
            Dictionary of features
            
        Returns:
        --------
        str
            Predicted emotion
        """
        scores = {emotion: 0 for emotion in self.emotions}
        
        # Calculate score for each emotion based on feature weights
        for feature, value in features.items():
            if feature in self.weights:
                for emotion in self.emotions:
                    scores[emotion] += value * self.weights[feature][emotion]
        
        # Add small random variation
        for emotion in scores:
            scores[emotion] += np.random.normal(0, 0.2)
        
        # Return emotion with highest score
        return max(scores, key=scores.get)


def load_eeg_file(file_path):
    """
    Load EEG data from a file
    
    Parameters:
    -----------
    file_path : str or file-like object
        Path to EEG data file (CSV or similar) or file-like object
        
    Returns:
    --------
    tuple
        (numpy.ndarray of EEG data, list of channel names)
    """
    try:
        # Handle both string paths and file-like objects (from st.file_uploader)
        if isinstance(file_path, str):
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        else:
            # Assume it's a file-like object from st.file_uploader
            file_name = getattr(file_path, 'name', '')
            if file_name.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_name.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                # Try CSV as default
                df = pd.read_csv(file_path)
        
        print(f"Loaded data with columns: {df.columns.tolist()}")
        
        # Identify time column if it exists
        time_column = None
        for col in df.columns:
            if col.lower() == 'time' or 'time' in col.lower():
                time_column = col
                break
        
        # Identify EEG channels using standard naming conventions
        standard_eeg_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
                          'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 
                          'Pz', 'P4', 'T6', 'O1', 'O2']
        eeg_channels = [col for col in df.columns if col in standard_eeg_channels]
        
        # If no standard EEG channels found, try alternate approach
        if not eeg_channels:
            # Exclude common non-EEG columns
            non_eeg_columns = ['time', 'index', 'digital', 'battery', 'timestamp', 'marker']
            eeg_channels = []
            
            for col in df.columns:
                # Skip non-numeric columns and those that match non-EEG names
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                col_lower = col.lower()
                if any(non_name in col_lower for non_name in non_eeg_columns):
                    continue
                
                # Skip accelerometer columns
                if 'accel' in col_lower:
                    continue
                    
                # Add remaining columns as potential EEG channels
                eeg_channels.append(col)
        
        # Extract EEG data
        if eeg_channels:
            print(f"Found EEG channels: {eeg_channels}")
            eeg_data = df[eeg_channels].values
            
            # Handle 1D vs 2D array
            if len(eeg_data.shape) == 1:
                eeg_data = eeg_data.reshape(-1, 1)
            
            return eeg_data, eeg_channels
        else:
            print("Warning: No EEG channels found in the file.")
            return None, []
            
    except Exception as e:
        print(f"Error loading file: {e}")
        import traceback
        traceback.print_exc()
        return None, []
