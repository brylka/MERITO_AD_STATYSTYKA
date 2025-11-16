from flask import Flask, render_template, request
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Odczyt modelu
model = load_model('model.keras')

# Odczyt scalera
scaler = joblib.load('scaler.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    formatted_value = None
    new_property = []
    if request.method == 'POST':
        new_property = {
            'MedInc':       float(request.form['MedInc']),
            'HouseAge':     float(request.form['HouseAge']),
            'AveRooms':     float(request.form['AveRooms']),
            'AveBedrms':    float(request.form['AveBedrms']),
            'Population':   float(request.form['Population']),
            'AveOccup':     float(request.form['AveOccup']),
            'Latitude':     float(request.form['Latitude']),
            'Longitude':    float(request.form['Longitude'])
        }

        # Tworzenie DataFrame z nowymi danymi
        new_property_df = pd.DataFrame([new_property])

        # Skalowanie danych wejściowych przy użyciu tego samego skalera użytego do trenowania modelu
        scaled_new_property = scaler.transform(new_property_df)

        # Prognozowanie ceny przy użyciu wytrenowanego modelu
        predicted_price = model.predict(scaled_new_property)

        # Wypisanie prognozowanej ceny
        #print(f'Prognozowana cena nieruchomości: {predicted_price[0][0]:.3f} (w jednostkach 100.000 USD)')

        # Obliczenie wartości nieruchomości
        property_value = predicted_price[0][0] * 100000
        # Formatowanie wartości z separatorem
        formatted_value = format(property_value, ',.0f').replace(',', '.')
        # Wydrukowanie prognozowanej ceny nieruchomości z separatorem
        #print(f'Prognozowana cena nieruchomości: {formatted_value} dolarów.')


    return render_template('housing.html', formatted_value=formatted_value, new_property=new_property)

if __name__ == '__main__':
    app.run(debug=True)