import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Odczyt modelu
model = load_model('model.keras')

# Odczyt scalera
scaler = joblib.load('scaler.pkl')

# Nowa nieruchomość, dla której chcemy oszacować cenę
new_property = {
    'MedInc': 8.32,
    'HouseAge': 25,
    'AveRooms': 6.23,
    'AveBedrms': 1.01,
    'Population': 1800,
    'AveOccup': 3.5,
    'Latitude': 37.88,
    'Longitude': -122.23
}

# Tworzenie DataFrame z nowymi danymi
new_property_df = pd.DataFrame([new_property])

# Skalowanie danych wejściowych przy użyciu tego samego skalera użytego do trenowania modelu
scaled_new_property = scaler.transform(new_property_df)

# Prognozowanie ceny przy użyciu wytrenowanego modelu
predicted_price = model.predict(scaled_new_property)

# Wypisanie prognozowanej ceny
print(f'Prognozowana cena nieruchomości: {predicted_price[0][0]:.3f} (w jednostkach 100.000 USD)')

# Obliczenie wartości nieruchomości
property_value = predicted_price[0][0] * 100000
# Formatowanie wartości z separatorem
formatted_value = format(property_value, ',.0f').replace(',', '.')
# Wydrukowanie prognozowanej ceny nieruchomości z separatorem
print(f'Prognozowana cena nieruchomości: {formatted_value} dolarów.')
