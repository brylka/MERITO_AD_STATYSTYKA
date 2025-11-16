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



print("\n" + "="*70)
print("REGRESJA WIELORAKA (MULTIPLE LINEAR REGRESSION)")
print("="*70)

# Generowanie danych
np.random.seed(42)
n = 100

# Zmienne niezależne
powierzchnia = np.random.uniform(30, 150, n)  # m²
liczba_pokoi = np.random.randint(1, 6, n)
wiek = np.random.uniform(0, 50, n)  # lata

# Zmienna zależna (z dodanym szumem)
cena = (100000 +
        5000 * powierzchnia +
        20000 * liczba_pokoi -
        1000 * wiek +
        np.random.normal(0, 50000, n))

# Tworzenie DataFrame
df = pd.DataFrame({
    'powierzchnia': powierzchnia,
    'liczba_pokoi': liczba_pokoi,
    'wiek': wiek,
    'cena': cena
})

print("\nPierwsze 5 wierszy danych:")
print(df.head())

print("\nStatystyki opisowe:")
print(df.describe())

# Macierz korelacji
print("\n" + "-"*70)
print("MACIERZ KORELACJI:")
print("-"*70)
correlation_matrix = df.corr()
print(correlation_matrix['cena'].sort_values(ascending=False))

# Wizualizacja korelacji
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
            center=0, fmt='.3f', linewidths=1)
plt.title('Macierz Korelacji', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Przygotowanie danych
X_multi = df[['powierzchnia', 'liczba_pokoi', 'wiek']]
y_multi = df['cena']

# Model regresji wielorakiej
model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

# Współczynniki
print("\n" + "-"*70)
print("WSPÓŁCZYNNIKI MODELU:")
print("-"*70)
print(f"β₀ (intercept):     {model_multi.intercept_:,.2f} zł")
print(f"β₁ (powierzchnia):  {model_multi.coef_[0]:,.2f} zł/m²")
print(f"β₂ (liczba_pokoi):  {model_multi.coef_[1]:,.2f} zł/pokój")
print(f"β₃ (wiek):          {model_multi.coef_[2]:,.2f} zł/rok")

print(f"\nRównanie regresji:")
print(f"cena = {model_multi.intercept_:,.0f} + "
      f"{model_multi.coef_[0]:.0f}×powierzchnia + "
      f"{model_multi.coef_[1]:.0f}×pokoje + "
      f"{model_multi.coef_[2]:.0f}×wiek")

# Przewidywania
y_pred_multi = model_multi.predict(X_multi)

# Metryki
r2_multi = r2_score(y_multi, y_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_multi, y_pred_multi))
mae_multi = mean_absolute_error(y_multi, y_pred_multi)

# Adjusted R²
n = len(y_multi)
p = X_multi.shape[1]
r2_adj = 1 - ((1 - r2_multi) * (n - 1) / (n - p - 1))

print("\n" + "-"*70)
print("METRYKI JAKOŚCI MODELU:")
print("-"*70)
print(f"R²:               {r2_multi:.4f} ({r2_multi*100:.2f}%)")
print(f"Adjusted R²:      {r2_adj:.4f} ({r2_adj*100:.2f}%)")
print(f"RMSE:             {rmse_multi:,.2f} zł")
print(f"MAE:              {mae_multi:,.2f} zł")

# Przykładowe przewidywanie
print("\n" + "-"*70)
print("PRZYKŁADOWE PRZEWIDYWANIE:")
print("-"*70)
przyklad = pd.DataFrame({
    'powierzchnia': [60, 80, 120],
    'liczba_pokoi': [3, 4, 5],
    'wiek': [10, 5, 20]
})

przewidywania = model_multi.predict(przyklad)

for i in range(len(przyklad)):
    print(f"\nMieszkanie {i+1}:")
    print(f"  Powierzchnia: {przyklad.iloc[i]['powierzchnia']} m²")
    print(f"  Pokoje: {przyklad.iloc[i]['liczba_pokoi']}")
    print(f"  Wiek: {przyklad.iloc[i]['wiek']} lat")
    print(f"  Przewidywana cena: {przewidywania[i]:,.2f} zł")

print("="*70)








from sklearn.model_selection import train_test_split

print("\n" + "="*70)
print("WALIDACJA MODELU - PODZIAŁ TRAIN/TEST")
print("="*70)

# Podział danych (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

print(f"\nRozmiar zbioru treningowego: {len(X_train)} ({len(X_train)/len(X_multi)*100:.0f}%)")
print(f"Rozmiar zbioru testowego:     {len(X_test)} ({len(X_test)/len(X_multi)*100:.0f}%)")

# Trenowanie modelu tylko na zbiorze treningowym
model_validated = LinearRegression()
model_validated.fit(X_train, y_train)

# Predykcje
y_train_pred = model_validated.predict(X_train)
y_test_pred = model_validated.predict(X_test)

# Metryki dla obu zbiorów
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("\n" + "-"*70)
print("PORÓWNANIE WYNIKÓW:")
print("-"*70)
print(f"{'Metryka':<20} {'Trening':>20} {'Test':>20}")
print("-"*70)
print(f"{'R²':<20} {r2_train:>20.4f} {r2_test:>20.4f}")
print(f"{'RMSE':<20} {rmse_train:>20,.2f} {rmse_test:>20,.2f}")
print("-"*70)

# Interpretacja
diff = abs(r2_train - r2_test)
if diff < 0.05:
    print("\nModel generalizuje dobrze (różnica R² < 0.05)")
elif diff < 0.1:
    print("\nModel może mieć lekkie przeuczenie (różnica R² < 0.1)")
else:
    print("\nModel jest prawdopodobnie przeuczony (różnica R² > 0.1)")

# Wizualizacja
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Trening
axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=50)
axes[0].plot([y_train.min(), y_train.max()],
             [y_train.min(), y_train.max()],
             'r--', linewidth=2)
axes[0].set_xlabel('Cena rzeczywista [zł]')
axes[0].set_ylabel('Cena przewidywana [zł]')
axes[0].set_title(f'Zbiór Treningowy (R² = {r2_train:.4f})')
axes[0].grid(True, alpha=0.3)

# Test
axes[1].scatter(y_test, y_test_pred, alpha=0.5, s=50, color='orange')
axes[1].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', linewidth=2)
axes[1].set_xlabel('Cena rzeczywista [zł]')
axes[1].set_ylabel('Cena przewidywana [zł]')
axes[1].set_title(f'Zbiór Testowy (R² = {r2_test:.4f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("="*70)

