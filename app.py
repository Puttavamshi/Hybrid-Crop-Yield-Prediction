import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import warnings
import pickle
import os

warnings.filterwarnings('ignore')
st.set_page_config(layout="wide")
st.title("🌾 Hybrid Crop Yield Prediction")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("reduced_crop_dataset_15_crops.csv")
    except FileNotFoundError:
        st.error("The data file 'reduced_crop_dataset_15_crops.csv' was not found.")
        st.stop()
    features = ['temperature', 'rainfall', 'ph', 'humidity', 'area']
    X = df[features].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['crop'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled, y, scaler, label_encoder, features

df, X_scaled, y, scaler, le, features = load_data()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def save_models(models, scaler, label_encoder, features):
    os.makedirs('models', exist_ok=True)
    with open('models/svm_model.pkl', 'wb') as f:
        pickle.dump(models['SVM'], f)
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(models['Random Forest'], f)
    torch.save(models['LSTM'].state_dict(), 'models/lstm_model.pth')
    torch.save(models['RNN'].state_dict(), 'models/rnn_model.pth')
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('models/features.pkl', 'wb') as f:
        pickle.dump(features, f)
    st.success("💾 Models saved to 'models/' directory!")

def load_models():
    try:
        with open('models/svm_model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        with open('models/random_forest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        with open('models/features.pkl', 'rb') as f:
            features = pickle.load(f)

        current_input_size = len(features)
        current_num_classes = len(np.unique(np.array(y)))
        lstm_model = LSTMModel(current_input_size, 256, 2, current_num_classes)
        lstm_model.load_state_dict(torch.load('models/lstm_model.pth'))
        lstm_model.eval()
        rnn_model = RNNModel(current_input_size, 256, 2, current_num_classes)
        rnn_model.load_state_dict(torch.load('models/rnn_model.pth'))
        rnn_model.eval()

        models = {
            'SVM': svm_model,
            'Random Forest': rf_model,
            'LSTM': lstm_model,
            'RNN': rnn_model
        }
        return models, scaler, label_encoder, features
    except:
        return None, None, None, None

def check_model_compatibility():
    try:
        for f in ['models/svm_model.pkl', 'models/random_forest_model.pkl',
                  'models/lstm_model.pth', 'models/rnn_model.pth',
                  'models/scaler.pkl', 'models/label_encoder.pkl', 'models/features.pkl']:
            if not os.path.exists(f):
                return False, "Some model files are missing"
        with open('models/features.pkl', 'rb') as f:
            saved_features = pickle.load(f)
        if saved_features != features:
            return False, "Dataset features have changed"
        with open('models/label_encoder.pkl', 'rb') as f:
            saved_le = pickle.load(f)
        if len(saved_le.classes_) != len(np.unique(np.array(y))):
            return False, "Label classes changed"
        return True, "Models are compatible"
    except Exception as e:
        return False, str(e)

@st.cache_resource
def train_and_evaluate_models():
    global X_train, X_test, y_train, y_test, scaler, le, features
    compatible, msg = check_model_compatibility()
    if compatible:
        models, scaler, label_encoder, features = load_models()
        if models:
            st.success("✅ Loaded pre-trained models")
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
            predictions = {
                'SVM': models['SVM'].predict(X_test),
                'Random Forest': models['Random Forest'].predict(X_test),
                'LSTM': torch.max(models['LSTM'](X_test_tensor), 1)[1].numpy(),
                'RNN': torch.max(models['RNN'](X_test_tensor), 1)[1].numpy()
            }
            return models, predictions, scaler, label_encoder, features

    st.warning(f"⚠️ {msg}. Models will be retrained.")
    st.info("🔄 Training models...")

    svm_model = SVC(kernel='rbf', C=10).fit(X_train, y_train)
    rf_model = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    input_size = X_scaled.shape[1]
    num_classes = len(np.unique(np.array(y)))
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)

    def train_model(model, loader, epochs=20):
        model.train()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()
        return model.eval()

    lstm = train_model(LSTMModel(input_size, 256, 2, num_classes), train_loader)
    rnn = train_model(RNNModel(input_size, 256, 2, num_classes), train_loader)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    lstm_pred = torch.max(lstm(X_test_tensor), 1)[1].numpy()
    rnn_pred = torch.max(rnn(X_test_tensor), 1)[1].numpy()

    models = {'SVM': svm_model, 'Random Forest': rf_model, 'LSTM': lstm, 'RNN': rnn}
    predictions = {'SVM': svm_pred, 'Random Forest': rf_pred, 'LSTM': lstm_pred, 'RNN': rnn_pred}
    save_models(models, scaler, le, features)
    return models, predictions, scaler, le, features

models, predictions, scaler, le, features = train_and_evaluate_models()

if st.sidebar.button('🔄 Retrain Models'):
    for f in os.listdir('models'):
        os.remove(os.path.join('models', f))
    st.success("Model files removed. Refresh to retrain.")
    st.stop()

st.sidebar.header('Make a Prediction')
feature_emojis = {
    'temperature': '🌡️',
    'rainfall': '🌧️',
    'ph': '🧪',
    'humidity': '💦',
    'area': '🏞️'
}

if features is not None and scaler is not None and le is not None:
    input_vals = {}
    for feat in features:
        input_vals[feat] = st.sidebar.slider(f"{feature_emojis.get(feat, '')} {feat}", float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()))
    if st.sidebar.button('Predict Crop'):
        input_df = pd.DataFrame([input_vals])
        scaled_input = scaler.transform(input_df)
        st.sidebar.subheader("Model Predictions")
        st.sidebar.write(f"**SVM:** {le.inverse_transform(models['SVM'].predict(scaled_input))[0]}")
        st.sidebar.write(f"**Random Forest:** {le.inverse_transform(models['Random Forest'].predict(scaled_input))[0]}")
        tensor_input = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(1)
        st.sidebar.write(f"**LSTM:** {le.inverse_transform(torch.max(models['LSTM'](tensor_input), 1)[1].numpy())[0]}")
        st.sidebar.write(f"**RNN:** {le.inverse_transform(torch.max(models['RNN'](tensor_input), 1)[1].numpy())[0]}")
else:
    st.sidebar.warning("Models not loaded properly. Please refresh the page.")

col1, col2 = st.columns(2)
with col1:
    st.header("Model Performance")
    perf = {}
    for name, pred in predictions.items():
        perf[name] = [
            accuracy_score(y_test, pred),
            precision_score(y_test, pred, average='weighted'),
            recall_score(y_test, pred, average='weighted'),
            f1_score(y_test, pred, average='weighted'),
        ]
    perf_df = pd.DataFrame(perf).T
    perf_df.columns = ["Accuracy", "Precision", "Recall", "F1-Score"]
    st.dataframe(perf_df)
    st.bar_chart(perf_df["Accuracy"])

with col2:
    st.header("Crop Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    crop_counts = df['crop'].value_counts()
    ax.pie(crop_counts, labels=[str(x) for x in crop_counts.index], autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
    ax.axis("equal")
    st.pyplot(fig)
