# EEG Emotion Detection System

![EEG Emotion Detection](https://img.shields.io/badge/EEG-Emotion%20Detection-blue)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)

A real-time application for emotion classification using EEG (Electroencephalogram) data. This interactive dashboard visualizes brain activity and detects emotions such as Happy, Sad, Neutral, Excited, and Calm.

![Dashboard Preview](https://via.placeholder.com/800x400?text=EEG+Emotion+Detection+Dashboard) <!-- Replace with actual screenshot -->

## Features

- 🧠 Real-time EEG signal visualization
- 😊 Emotion classification with visual feedback
- 📊 Frequency band analysis (Delta, Theta, Alpha, Beta, Gamma)
- 📈 Historical emotion data tracking
- 📋 Support for importing external EEG data files
- 🔧 Adjustable visualization parameters (smoothing, amplitude)
- 🎚️ Multiple visualization options (bar charts, radar charts, pie charts)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Windows Installation

1. **Clone the repository**
   ```
   git clone https://github.com/yourusername/eeg-emotion-detection.git
   cd eeg-emotion-detection
   ```

2. **Set up a virtual environment (recommended)**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

### Unix/macOS Installation

1. **Clone the repository**
   ```
   git clone https://github.com/yourusername/eeg-emotion-detection.git
   cd eeg-emotion-detection
   ```

2. **Set up a virtual environment (recommended)**
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

## How to Run

### Windows

1. **Activate the virtual environment (if you created one)**
   ```
   venv\Scripts\activate
   ```

2. **Navigate to the project directory**
   ```
   cd eeg_emotion_detection
   ```

3. **Run the application**
   ```
   streamlit run app.py
   ```

4. **Open the application in your browser**
   - The application will automatically open in your default web browser
   - If not, open a browser and go to: http://localhost:8501

### Unix/macOS

1. **Activate the virtual environment (if you created one)**
   ```
   source venv/bin/activate
   ```

2. **Navigate to the project directory**
   ```
   cd eeg_emotion_detection
   ```

3. **Run the application**
   ```
   streamlit run app.py
   ```

4. **Open the application in your browser**
   - The application will automatically open in your default web browser
   - If not, open a browser and go to: http://localhost:8501

## Using the Application

1. **Device Connection**
   - Choose between real EEG device or simulated data
   - Connect to your EEG device if available

2. **Start Detection**
   - Click "Start Detection" to begin real-time analysis
   - Adjust update interval as needed

3. **Import Data**
   - Upload CSV or Excel files with EEG data
   - Preview data before processing

4. **Visualization Settings**
   - Select channels to display
   - Adjust smoothing and amplitude scale
   - Choose between different visualization types

5. **Historical Analysis**
   - View emotion history over time
   - Select date ranges for analysis
   - Examine emotion statistics

## Dependencies

- [Streamlit](https://streamlit.io/) - Interactive web interface
- [NumPy](https://numpy.org/) - Numerical computing
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [Plotly](https://plotly.com/) - Interactive visualizations
- [SciPy](https://scipy.org/) - Signal processing

## Requirements File

Create a `requirements.txt` file in your project root with the following content:

```
streamlit>=1.18.0
numpy>=1.20.0
pandas>=1.3.0
plotly>=5.5.0
scipy>=1.7.0
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [10-20 System](https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)) for EEG electrode placement
- [EEG Band Frequencies](https://en.wikipedia.org/wiki/Electroencephalography) for frequency band definitions
