# 🌾 Hybrid Crop Yield Prediction

A comprehensive machine learning application that predicts optimal crop recommendations based on environmental and agricultural parameters using multiple AI models.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=flat&logo=github)](https://github.com/Puttavamshi/Hybrid-Crop-Yield-Prediction.git)

## 🚀 Features

- **Multi-Model Approach**: Uses 4 different machine learning models:
  - Support Vector Machine (SVM)
  - Random Forest Classifier
  - Long Short-Term Memory (LSTM) Neural Network
  - Recurrent Neural Network (RNN)

- **Smart Model Persistence**: 
  - Automatically saves trained models to pickle files
  - Loads pre-trained models for instant predictions
  - Handles dataset changes gracefully
  - Manual retraining option available

- **Interactive Web Interface**:
  - Beautiful Streamlit-based UI with emojis
  - Real-time crop predictions
  - Model performance comparison
  - Data visualization with charts

- **Input Parameters**:
  - 🌡️ Temperature
  - 🌧️ Rainfall
  - 🧪 pH level
  - 💦 Humidity
  - 🏞️ Area

## 📋 Requirements

```bash
pip install -r requirements.txt
```

## 🛠️ Installation

### Option 1: Clone from GitHub
```bash
git clone https://github.com/Puttavamshi/Hybrid-Crop-Yield-Prediction.git
cd Hybrid-Crop-Yield-Prediction
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: Manual Installation
1. **Download the project files** from [GitHub Repository](https://github.com/Puttavamshi/Hybrid-Crop-Yield-Prediction.git)
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure you have the dataset file**: `reduced_crop_dataset_15_crops.csv`
4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
Hybrid-Crop-Yield-Prediction/
├── app.py                              # Main application file
├── reduced_crop_dataset_15_crops.csv   # Dataset file
├── requirements.txt                    # Python dependencies
└── models/                             # Auto-generated model storage
    ├── svm_model.pkl
    ├── random_forest_model.pkl
    ├── lstm_model.pth
    ├── rnn_model.pth
    ├── scaler.pkl
    ├── label_encoder.pkl
    └── features.pkl
```

## 🎯 How It Works

### 1. **Data Loading & Preprocessing**
- Loads crop dataset with environmental parameters
- Applies StandardScaler for feature normalization
- Uses LabelEncoder for crop name encoding

### 2. **Model Training/Loading**
- **First Run**: Trains all 4 models and saves them
- **Subsequent Runs**: Loads pre-trained models instantly
- **Compatibility Check**: Verifies model compatibility with current dataset

### 3. **Prediction Interface**
- Interactive sliders for input parameters
- Real-time predictions from all models
- Performance metrics comparison

### 4. **Visualization**
- Model accuracy comparison charts
- Crop distribution pie chart
- Performance metrics table

## 🎮 Usage

1. **Launch the app**: `streamlit run app.py`
2. **Wait for model loading/training** (first run takes longer)
3. **Adjust input parameters** using the sidebar sliders
4. **Click "Predict Crop"** to get recommendations
5. **View model performance** in the main dashboard
6. **Retrain models** if needed using the "🔄 Retrain Models" button

## 🔄 Model Management

### Automatic Features:
- **Smart Loading**: Models load automatically if compatible
- **Auto-Save**: Newly trained models are saved automatically
- **Compatibility Check**: Handles dataset changes automatically

### Manual Controls:
- **Retrain Button**: Removes old models and forces retraining
- **Refresh**: Reloads the page to apply changes

**Happy Farming! 🌱** 