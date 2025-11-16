import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Konfiguracja
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("REGRESJA LINIOWA - PODSTAWOWA IMPLEMENTACJA")
print("="*70)

# Dane z Zadania 1
X = np.array([2, 3, 4, 5, 6]).reshape(-1, 1)  # MUSI być 2D!
y = np.array([55, 65, 70, 80, 85])

print("\nDANE:")
print(f"X (godziny nauki): {X.flatten()}")
print(f"y (wynik): {y}")

# Tworzenie i trenowanie modelu
model = LinearRegression()
model.fit(X, y)

# Współczynniki
beta_0 = model.intercept_
beta_1 = model.coef_[0]

print("\n" + "-"*70)
print("WSPÓŁCZYNNIKI MODELU:")
print("-"*70)
print(f"β₀ (intercept):  {beta_0:.4f}")
print(f"β₁ (slope):      {beta_1:.4f}")
print(f"\nRównanie regresji:")
print(f"ŷ = {beta_0:.2f} + {beta_1:.2f}x")

# Przewidywania
y_pred = model.predict(X)

print("\n" + "-"*70)
print("WARTOŚCI PRZEWIDYWANE:")
print("-"*70)
for i in range(len(X)):
    print(f"X={X[i,0]}: y_rzeczywiste={y[i]}, y_przewidywane={y_pred[i]:.2f}, "
          f"błąd={y[i]-y_pred[i]:.2f}")

# Metryki
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)

print("\n" + "-"*70)
print("METRYKI JAKOŚCI MODELU:")
print("-"*70)
print(f"R² (współczynnik determinacji):  {r2:.4f} ({r2*100:.2f}%)")
print(f"MSE (błąd średniokwadratowy):    {mse:.4f}")
print(f"RMSE (pierwiastek MSE):          {rmse:.4f}")
print(f"MAE (średni błąd bezwzględny):   {mae:.4f}")

# Przewidywanie dla nowej wartości
print("\n" + "-"*70)
print("PRZEWIDYWANIE DLA NOWYCH DANYCH:")
print("-"*70)
new_X = np.array([[7], [8], [10]])
new_y = model.predict(new_X)

for i in range(len(new_X)):
    print(f"Dla {new_X[i,0]} godzin nauki → przewidywany wynik: {new_y[i]:.2f} pkt")

print("="*70)
