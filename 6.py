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

# Wizualizacja kompleksowa
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Analiza Regresji Liniowej', fontsize=16, fontweight='bold')

# 1. Scatter plot z linią regresji
axes[0, 0].scatter(X, y, color='blue', s=100, alpha=0.6,
                   edgecolors='black', linewidth=2, label='Dane rzeczywiste')
axes[0, 0].plot(X, y_pred, color='red', linewidth=2,
                label=f'ŷ = {beta_0:.1f} + {beta_1:.1f}x')

# Dodanie punktów przewidywanych
X_range = np.linspace(1, 11, 100).reshape(-1, 1)
y_range = model.predict(X_range)
axes[0, 0].plot(X_range, y_range, 'r--', alpha=0.5)

# Punkty dla nowych danych
axes[0, 0].scatter(new_X, new_y, color='green', s=150, marker='*',
                   edgecolors='black', linewidth=2, label='Przewidywania', zorder=5)

axes[0, 0].set_xlabel('Godziny nauki', fontsize=12)
axes[0, 0].set_ylabel('Wynik egzaminu [pkt]', fontsize=12)
axes[0, 0].set_title(f'Regresja Liniowa (R² = {r2:.4f})', fontsize=14)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 2. Wykres reszt (Residual Plot)
residuals = y - y_pred

axes[0, 1].scatter(y_pred, residuals, color='purple', s=100,
                   alpha=0.6, edgecolors='black', linewidth=2)
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Wartości przewidywane', fontsize=12)
axes[0, 1].set_ylabel('Reszty (y - ŷ)', fontsize=12)
axes[0, 1].set_title('Wykres Reszt', fontsize=14)
axes[0, 1].grid(True, alpha=0.3)

# Dodanie adnotacji
for i in range(len(y_pred)):
    axes[0, 1].annotate(f'{residuals[i]:.1f}',
                       (y_pred[i], residuals[i]),
                       textcoords="offset points",
                       xytext=(0,10), ha='center', fontsize=9)

# 3. Histogram reszt
axes[1, 0].hist(residuals, bins=5, edgecolor='black', alpha=0.7, color='coral')
axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Reszty', fontsize=12)
axes[1, 0].set_ylabel('Częstość', fontsize=12)
axes[1, 0].set_title('Rozkład Reszt', fontsize=14)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Wartości rzeczywiste vs przewidywane
axes[1, 1].scatter(y, y_pred, color='blue', s=100, alpha=0.6,
                   edgecolors='black', linewidth=2)
axes[1, 1].plot([y.min(), y.max()], [y.min(), y.max()],
                'r--', linewidth=2, label='Idealne dopasowanie')
axes[1, 1].set_xlabel('Wartości rzeczywiste', fontsize=12)
axes[1, 1].set_ylabel('Wartości przewidywane', fontsize=12)
axes[1, 1].set_title('Rzeczywiste vs Przewidywane', fontsize=14)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

# Dodanie R² na wykresie
axes[1, 1].text(0.05, 0.95, f'R² = {r2:.4f}',
               transform=axes[1, 1].transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('regresja_analiza.png', dpi=300, bbox_inches='tight')
plt.show()



