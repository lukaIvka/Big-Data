import streamlit as st
import pandas as pd
import numpy as np
import torch
from model import LeakDetectorModel, predict_leakage
import joblib
import matplotlib.pyplot as plt

# Učitaj model i scaler
model = LeakDetectorModel(input_size=13)  # Prilagodi prema broju feature-a
model.load_state_dict(torch.load('best_leak_detector_model.pth'))
model.eval()
scaler = joblib.load('../data/scaler.pkl')

st.title("Detekcija Curenja Vode u Pametnim Gradovima")

# Unos podataka preko forme
st.header("Unesite podatke sa senzora")
pressure = st.number_input("Pritisak (Pressure)", min_value=0.0, value=60.0)
flow_rate = st.number_input("Protok (Flow_Rate)", min_value=0.0, value=80.0)
temperature = st.number_input("Temperatura (Temperature)", min_value=0.0, value=100.0)
vibration = st.number_input("Vibracije (Vibration)", min_value=0.0, value=3.0)
rpm = st.number_input("RPM", min_value=0.0, value=2000.0)
operational_hours = st.number_input("Radni sati (Operational_Hours)", min_value=0.0, value=5000.0)

# Podrazumevane vrednosti za lokacijske parametre (prosjeci iz dataset-a)
zone = 2.0
block = 2.0
pipe = 2.0
location_code = 62.4
latitude = 25.18
longitude = 55.25

# Kreiraj ulazni niz
input_data = np.array([[pressure, flow_rate, temperature, vibration, rpm, operational_hours,
                       zone, block, pipe, location_code, latitude, longitude, 0.0]])  # 0.0 za Pressure_Delta
input_scaled = scaler.transform(input_data)

if st.button("Predvidi curenje"):
    prediction = predict_leakage(model, input_scaled, threshold=0.4)
    result = "Curenje detektovano!" if prediction[0] == 1 else "Nema curenja."
    st.success(f"Rezultat: {result}")

# Grafikon predikcija vs stvarne vrednosti
st.header("Poređenje Predikcija i Stvarnih Vrednosti")
predictions = np.loadtxt('test_predictions.csv', delimiter=',')
labels = np.loadtxt('test_labels.csv', delimiter=',')
fig, ax = plt.subplots()
ax.scatter(range(len(predictions)), predictions + 0.01, color='blue', label='Predikcije', alpha=0.6, s=100)
ax.scatter(range(len(labels)), labels, color='red', label='Stvarne vrednosti')
ax.set_xlabel('Uzorci')
ax.set_ylabel('Vrednost (0 ili 1)')
ax.set_title('Predikcije vs Stvarne Vrednosti')
ax.legend()
st.pyplot(fig)

# Grafikon istorije treniranja
st.header("Istorija Treniranja (Train/Val Loss)")
with open('training_history.txt', 'r') as f:
    lines = f.readlines()
epochs = [int(line.split('Epoch ')[1].split(',')[0]) for line in lines]  # Ispravljeno parsiranje epohe
train_loss = [float(line.split('Train Loss: ')[1].split(',')[0]) for line in lines]
val_loss = [float(line.split('Val Loss: ')[1].strip()) for line in lines]
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(epochs, train_loss, color='blue', label='Train Loss')
ax2.plot(epochs, val_loss, color='red', label='Val Loss')
ax2.set_xlabel('Epohe')
ax2.set_ylabel('Loss')
ax2.set_title("Istorija Treniranja")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# Performanse modela u tabeli
st.header("Performanse Modela")
data = {
    'Threshold': [0.4, 0.5, 0.6],
    'Accuracy': [0.9880, 0.9920, 0.9860],
    'Precision': [0.9143, 1.0000, 1.0000],
    'Recall': [0.9143, 0.8857, 0.8000]
}
df = pd.DataFrame(data)
st.table(df.style.set_properties(**{'text-align': 'center'}).set_table_styles(
    [{'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('font-weight', 'bold')]}]
))