import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Dane z zadania
wyniki = np.array([65, 70, 75, 80, 80, 85, 85, 85, 90, 95, 100, 150])

# print("="*50, "STATYSTYKA OPISOWA - PODSTAWOWE MIARY", "="*50, sep="\n")
#
# # Miary tendencji centralnej
# print("1. MIARY TENDENCJI CENTRALNEJ:")
# print("-"*40)
# print(f"Średnia arytmetyczna: {np.mean(wyniki):.2f} pkt")
# print(f"Mediana:              {np.median(wyniki):.2f} pkt")
# print(f"Moda:                 {stats.mode(wyniki, keepdims=True).mode[0]} pkt")
#
# # Miary rozproszenia
# print("\n2. MIARY ROZPROSZENIA:")
# print("-"*40)
# print(f"Minimum:              {np.min(wyniki):.2f} pkt")
# print(f"Maksimum:             {np.max(wyniki):.2f} pkt")
# print(f"Rozstęp:              {np.ptp(wyniki):.2f} pkt")
# #print(f"Rozstęp:              {np.max(wyniki)-np.min(wyniki):.2f} pkt")
# print(f"Wariancja:            {np.var(wyniki, ddof=1):.2f} pkt^2")
# print(f"Odchylenie std:       {np.std(wyniki, ddof=1):.2f} pkt")
#
# # Kwartyle
# print("\n3. KWARTYLE:")
# print("-"*40)
# q1 = np.percentile(wyniki, 25)
# q2 = np.percentile(wyniki, 50)
# q3 = np.percentile(wyniki, 75)
# iqr = q3 - q1
#
# print(f"Q1 (25%):             {q1:.2f} pkt")
# print(f"Q2 (50% - mediana):   {q2:.2f} pkt")
# print(f"Q3 (75%):             {q3:.2f} pkt")
# print(f"IQR (Q3 - Q1):        {iqr:.2f} pkt")
#
# # Wykrywanie outlierów
# print("\n4. WYKRYWANIE WARTOŚCI ODSTAJĄCYCH:")
# print("-"*40)
# dolna_granica = q1 - 1.5 * iqr
# gorna_granica = q3 + 1.5 * iqr
#
# print(f"Dolna granica:        {dolna_granica:.2f} pkt")
# print(f"Górna granica:        {gorna_granica:.2f} pkt")
#
# outliery = wyniki[(wyniki < dolna_granica) | (wyniki > gorna_granica)]
# if len(outliery) > 0:
#     print(f"Wykryte outliery:     {outliery}")
# else:
#     print("Brak wartości odstających.")

# Tworzenie DataFrame z wynikami
df = pd.DataFrame({
    'student': [f"S{i}" for i in range(1,13)],
    'wynik': wyniki
})

print("="*50, "ANALIZA Z UŻYCIEM PANDASA", "="*50, sep="\n")

print("\nPięć pierwszych wierszy próbki:")
print(df.head())

print("\nInformacje o DataFrame:")
print(df.info())

print("\nStatystyki opisowe:")
print(df["wynik"].describe().round(2))

# Dodatkowe statystyki
print(f"\nSkośność:        {df["wynik"].skew():.4f}")
print(f"Kurtoza:         {df["wynik"].kurtosis():.4f}")

if df["wynik"].skew() > 0:
    print("Rozkład jest skośny prawostronnie (prawy ogon dłuższy)")
elif df["wynik"].skew() < 0:
    print("Rozkład jest skośny lewostronnie (lewy ogon dłuższy)")
else:
    print("Rozkład jest symetryczny")