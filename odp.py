"""
Autor: Bartosz Bryniarski
Data: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

# Konfiguracja
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
np.random.seed(42)

print("=" * 80)
print("ROZWIĄZANIA ZADAŃ PRAKTYCZNYCH - STATYSTYKA I REGRESJA")
print("=" * 80)

# ============================================================================
# ZESTAW 1: STATYSTYKA OPISOWA I INFERENCYJNA
# ============================================================================

print("\n" + "=" * 80)
print("ZESTAW 1: STATYSTYKA OPISOWA I INFERENCYJNA")
print("=" * 80)

# ----------------------------------------------------------------------------
# Zadanie 1.1: Tworzenie i Podstawowa Analiza Danych
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("ZADANIE 1.1: Tworzenie i Podstawowa Analiza Danych")
print("-" * 80)

# 1. Utwórz array NumPy
oceny = np.array([4.5, 3.0, 4.0, 5.0, 3.5, 4.5, 3.0, 4.0, 5.0, 3.5,
                  4.0, 4.5, 3.5, 4.0, 3.0, 5.0, 4.5, 4.0, 3.5, 2.0])

print("\n1. Array NumPy utworzony:")
print(f"   Oceny: {oceny}")

# 2. Podstawowe informacje
print("\n2. Podstawowe informacje:")
print(f"   Liczba ocen: {len(oceny)}")
print(f"   Typ danych: {oceny.dtype}")
print(f"   Kształt: {oceny.shape}")

# 3. Miary tendencji centralnej
print("\n3. Miary tendencji centralnej:")
srednia = np.mean(oceny)
mediana = np.median(oceny)
moda = stats.mode(oceny, keepdims=True)

print(f"   Średnia: {srednia:.4f}")
print(f"   Mediana: {mediana:.4f}")
print(f"   Moda: {moda.mode[0]:.1f} (wystąpiła {moda.count[0]} razy)")

# 4. Miary rozproszenia
print("\n4. Miary rozproszenia:")
minimum = np.min(oceny)
maksimum = np.max(oceny)
rozstep = np.ptp(oceny)
wariancja = np.var(oceny, ddof=1)
odch_std = np.std(oceny, ddof=1)

print(f"   Minimum: {minimum:.1f}")
print(f"   Maksimum: {maksimum:.1f}")
print(f"   Rozstęp: {rozstep:.1f}")
print(f"   Wariancja: {wariancja:.4f}")
print(f"   Odchylenie standardowe: {odch_std:.4f}")

# 5. Kwartyle
print("\n5. Kwartyle:")
q1 = np.percentile(oceny, 25)
q2 = np.percentile(oceny, 50)
q3 = np.percentile(oceny, 75)
iqr = q3 - q1

print(f"   Q1 (25%): {q1:.2f}")
print(f"   Q2 (50%): {q2:.2f}")
print(f"   Q3 (75%): {q3:.2f}")
print(f"   IQR: {iqr:.2f}")

# 6. Wykrywanie outlierów
print("\n6. Wykrywanie wartości odstających:")
dolna_granica = q1 - 1.5 * iqr
gorna_granica = q3 + 1.5 * iqr

print(f"   Dolna granica: {dolna_granica:.2f}")
print(f"   Górna granica: {gorna_granica:.2f}")

outliery = oceny[(oceny < dolna_granica) | (oceny > gorna_granica)]
if len(outliery) > 0:
    print(f"   Wykryte outliery: {outliery}")
else:
    print(f"   Brak wartości odstających")

# ----------------------------------------------------------------------------
# Zadanie 1.2: Wizualizacja Danych
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("ZADANIE 1.2: Wizualizacja Danych")
print("-" * 80)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Histogram
axes[0].hist(oceny, bins=10, edgecolor='black', alpha=0.7, color='skyblue')
axes[0].axvline(srednia, color='red', linestyle='--', linewidth=2, label=f'Średnia: {srednia:.2f}')
axes[0].axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
axes[0].set_xlabel('Ocena')
axes[0].set_ylabel('Częstość')
axes[0].set_title('Histogram Ocen')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Boxplot
bp = axes[1].boxplot(oceny, patch_artist=True, vert=True,
                     boxprops=dict(facecolor='lightblue'),
                     medianprops=dict(color='red', linewidth=2))
axes[1].set_ylabel('Ocena')
axes[1].set_title('Boxplot - Wykrywanie Outlierów')
axes[1].set_xticklabels(['Oceny'])
axes[1].grid(True, alpha=0.3, axis='y')

# Adnotacje dla boxplota
axes[1].text(1.15, q1, f'Q1: {q1:.2f}', fontsize=10)
axes[1].text(1.15, q2, f'Q2: {q2:.2f}', fontsize=10, color='red')
axes[1].text(1.15, q3, f'Q3: {q3:.2f}', fontsize=10)

# 3. Wykres słupkowy częstości
unique_vals, counts = np.unique(oceny, return_counts=True)
axes[2].bar(unique_vals, counts, edgecolor='black', alpha=0.7, color='coral')
axes[2].set_xlabel('Ocena')
axes[2].set_ylabel('Liczba wystąpień')
axes[2].set_title('Częstość Występowania Ocen')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/zadanie_1_2_wizualizacje.png', dpi=300, bbox_inches='tight')
print("Wizualizacje zapisane jako 'zadanie_1_2_wizualizacje.png'")
plt.close()

# ----------------------------------------------------------------------------
# Zadanie 1.3: Analiza DataFrame
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("ZADANIE 1.3: Analiza DataFrame")
print("-" * 80)

# 1. Utwórz DataFrame
df_studenci = pd.DataFrame({
    'Imię': ['Anna', 'Bartek', 'Celina', 'Damian', 'Ewa', 'Filip'],
    'Wiek': [20, 21, 19, 22, 20, 21],
    'Wzrost': [165, 180, 170, 175, 168, 182],
    'Waga': [55, 75, 60, 70, 58, 80],
    'Godziny_nauki': [15, 10, 20, 8, 18, 12]
})

print("\n1. DataFrame utworzony")

# 2. Wyświetl wiersze
print("\n2. Pierwsze 3 wiersze:")
print(df_studenci.head(3))

print("\nOstatnie 2 wiersze:")
print(df_studenci.tail(2))

print("\nInformacje o DataFrame:")
print(df_studenci.info())

# 3. Statystyki opisowe
print("\n3. Statystyki opisowe:")
print(df_studenci.describe())

# 4. Statystyki dla kolumn
print("\n4. Statystyki dla poszczególnych kolumn:")
print(f"   Średni wiek: {df_studenci['Wiek'].mean():.2f}")
print(f"   Mediana wzrostu: {df_studenci['Wzrost'].median():.2f}")
print(f"   Odch. std wagi: {df_studenci['Waga'].std():.2f}")

# 5. Znajdź
print("\n5. Wyszukiwanie:")
max_nauka = df_studenci.loc[df_studenci['Godziny_nauki'].idxmax()]
print(f"   Student z max godzin nauki: {max_nauka['Imię']} ({max_nauka['Godziny_nauki']} godz)")

min_wzrost = df_studenci.loc[df_studenci['Wzrost'].idxmin()]
print(f"   Student z min wzrostem: {min_wzrost['Imię']} ({min_wzrost['Wzrost']} cm)")

print(f"   Średni wiek: {df_studenci['Wiek'].mean():.2f}")

# 6. Nowa kolumna BMI
print("\n6. Obliczenie BMI:")
df_studenci['BMI'] = df_studenci['Waga'] / (df_studenci['Wzrost'] / 100) ** 2
print(df_studenci)

# ----------------------------------------------------------------------------
# Zadanie 1.4: Test t-Studenta dla Jednej Próbki
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("ZADANIE 1.4: Test t-Studenta dla Jednej Próbki")
print("-" * 80)

# 1. Dane
baterie = np.array([23.5, 24.2, 23.8, 24.5, 23.0, 24.8, 23.5, 24.0, 23.2, 24.3,
                    23.7, 24.1, 23.9, 24.6, 23.4])

print("\n1. Dane baterii:")
print(f"   {baterie}")

# 2. Statystyki
print("\n2. Podstawowe statystyki:")
mean_baterie = np.mean(baterie)
std_baterie = np.std(baterie, ddof=1)
n_baterie = len(baterie)

print(f"   Średnia z próbki: {mean_baterie:.4f} godz")
print(f"   Odchylenie std: {std_baterie:.4f} godz")
print(f"   Liczba obserwacji: {n_baterie}")

# 3. Hipotezy
print("\n3. Hipotezy:")
mu_0 = 24
print(f"   H₀: μ = {mu_0} (twierdzenie producenta)")
print(f"   H₁: μ ≠ {mu_0} (test dwustronny)")

# 4. Test t
print("\n4. Test t-Studenta:")
t_stat, p_value = stats.ttest_1samp(baterie, mu_0)
print(f"   Statystyka t: {t_stat:.4f}")
print(f"   P-value: {p_value:.4f}")

# 5. Decyzja
print("\n5. Decyzja (α = 0.05):")
alpha = 0.05
if p_value < alpha:
    print(f"   p-value ({p_value:.4f}) < α ({alpha})")
    print(f"   ODRZUCAMY H₀")
    print(f"   Wniosek: Średni czas pracy baterii RÓŻNI SIĘ od {mu_0} godz")
else:
    print(f"   p-value ({p_value:.4f}) ≥ α ({alpha})")
    print(f"   NIE ODRZUCAMY H₀")
    print(f"   Wniosek: Brak podstaw do odrzucenia twierdzenia producenta")

# 6. Przedział ufności
print("\n6. Przedział ufności (95%):")
ci = stats.t.interval(0.95, n_baterie - 1,
                      loc=mean_baterie,
                      scale=stats.sem(baterie))
print(f"   Przedział: [{ci[0]:.4f}, {ci[1]:.4f}] godz")
if ci[0] <= mu_0 <= ci[1]:
    print(f"   Wartość {mu_0} NALEŻY do przedziału ✓")
else:
    print(f"   Wartość {mu_0} NIE NALEŻY do przedziału ✗")

# ----------------------------------------------------------------------------
# Zadanie 1.5: Test t-Studenta dla Dwóch Grup
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("ZADANIE 1.5: Test t-Studenta dla Dwóch Grup")
print("-" * 80)

# 1. Dane
grupa_a = np.array([72, 68, 75, 70, 73, 69, 74, 71, 76, 70])
grupa_b = np.array([78, 82, 80, 85, 79, 83, 81, 84, 80, 82])

print("\n1. Dane:")
print(f"   Grupa A: {grupa_a}")
print(f"   Grupa B: {grupa_b}")

# 2. Statystyki
print("\n2. Statystyki dla obu grup:")
stats_comparison = pd.DataFrame({
    'Grupa A': [np.mean(grupa_a), np.median(grupa_a), np.std(grupa_a, ddof=1)],
    'Grupa B': [np.mean(grupa_b), np.median(grupa_b), np.std(grupa_b, ddof=1)]
}, index=['Średnia', 'Mediana', 'Odch. std'])
print(stats_comparison)

# 3. Sprawdzenie założeń
print("\n3. Sprawdzenie założeń:")

# Test normalności
_, p_shapiro_a = stats.shapiro(grupa_a)
_, p_shapiro_b = stats.shapiro(grupa_b)
print(f"   Test normalności (Shapiro-Wilk):")
print(f"   Grupa A: p-value = {p_shapiro_a:.4f}", "✓" if p_shapiro_a > 0.05 else "✗")
print(f"   Grupa B: p-value = {p_shapiro_b:.4f}", "✓" if p_shapiro_b > 0.05 else "✗")

# Test jednorodności wariancji
_, p_levene = stats.levene(grupa_a, grupa_b)
print(f"\n   Test Levene'a (jednorodność wariancji):")
print(f"   P-value = {p_levene:.4f}", "✓" if p_levene > 0.05 else "✗")

# 4. Test t
print("\n4. Test t-Studenta (2 grupy):")
t_stat_2, p_value_2 = stats.ttest_ind(grupa_a, grupa_b)
print(f"   Statystyka t: {t_stat_2:.4f}")
print(f"   P-value: {p_value_2:.4f}")

# 5. Decyzja
print("\n5. Decyzja (α = 0.05):")
if p_value_2 < alpha:
    print(f"   p-value ({p_value_2:.4f}) < α ({alpha})")
    print(f"   → Grupy RÓŻNIĄ SIĘ istotnie statystycznie")
else:
    print(f"   p-value ({p_value_2:.4f}) ≥ α ({alpha})")
    print(f"   → Brak istotnej różnicy")

# 6. Cohen's d
print("\n6. Wielkość efektu (Cohen's d):")
pooled_std = np.sqrt(((len(grupa_a) - 1) * np.var(grupa_a, ddof=1) +
                      (len(grupa_b) - 1) * np.var(grupa_b, ddof=1)) /
                     (len(grupa_a) + len(grupa_b) - 2))
cohens_d = (np.mean(grupa_b) - np.mean(grupa_a)) / pooled_std
print(f"   Cohen's d: {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    effect = "mały"
elif abs(cohens_d) < 0.5:
    effect = "średni"
elif abs(cohens_d) < 0.8:
    effect = "duży"
else:
    effect = "bardzo duży"
print(f"   Interpretacja: efekt {effect}")

# 7. Wizualizacja
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot([grupa_a, grupa_b], labels=['Grupa A', 'Grupa B'],
           patch_artist=True,
           boxprops=dict(facecolor='lightblue'),
           medianprops=dict(color='red', linewidth=2))
ax.set_ylabel('Wyniki')
ax.set_title('Porównanie Dwóch Grup')
ax.grid(True, alpha=0.3, axis='y')
plt.savefig('/mnt/user-data/outputs/zadanie_1_5_porownanie.png', dpi=300, bbox_inches='tight')
print("\n7. Wizualizacja zapisana jako 'zadanie_1_5_porownanie.png'")
plt.close()

# ============================================================================
# ZESTAW 2: ROZKŁADY PRAWDOPODOBIEŃSTWA
# ============================================================================

print("\n" + "=" * 80)
print("ZESTAW 2: ROZKŁADY PRAWDOPODOBIEŃSTWA")
print("=" * 80)

# ----------------------------------------------------------------------------
# Zadanie 2.1: Rozkład Dwumianowy - Rzut Monetą
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("ZADANIE 2.1: Rozkład Dwumianowy - Rzut Monetą")
print("-" * 80)

# 1. Parametry
n_moneta = 20
p_moneta = 0.5
print(f"\n1. Parametry: n={n_moneta}, p={p_moneta}")

# 2. Prawdopodobieństwa
print("\n2. Prawdopodobieństwa:")
prob_10 = stats.binom.pmf(10, n_moneta, p_moneta)
prob_15 = stats.binom.pmf(15, n_moneta, p_moneta)
prob_leq_8 = stats.binom.cdf(8, n_moneta, p_moneta)
prob_geq_12 = 1 - stats.binom.cdf(11, n_moneta, p_moneta)

print(f"   P(X = 10):  {prob_10:.4f} ({prob_10 * 100:.2f}%)")
print(f"   P(X = 15):  {prob_15:.4f} ({prob_15 * 100:.2f}%)")
print(f"   P(X ≤ 8):   {prob_leq_8:.4f} ({prob_leq_8 * 100:.2f}%)")
print(f"   P(X ≥ 12):  {prob_geq_12:.4f} ({prob_geq_12 * 100:.2f}%)")

# 3. Charakterystyki
print("\n3. Charakterystyki rozkładu:")
mean_binom = stats.binom.mean(n_moneta, p_moneta)
var_binom = stats.binom.var(n_moneta, p_moneta)
std_binom = stats.binom.std(n_moneta, p_moneta)

print(f"   E(X):   {mean_binom}")
print(f"   Var(X): {var_binom:.2f}")
print(f"   Std(X): {std_binom:.2f}")

# 4. Wizualizacja
k_values = np.arange(0, n_moneta + 1)
probabilities = stats.binom.pmf(k_values, n_moneta, p_moneta)

plt.figure(figsize=(12, 6))
plt.bar(k_values, probabilities, alpha=0.7, edgecolor='black', color='skyblue')
plt.axvline(mean_binom, color='red', linestyle='--', linewidth=2,
            label=f'E(X) = {mean_binom}')
plt.xlabel('Liczba orłów (k)')
plt.ylabel('Prawdopodobieństwo P(X=k)')
plt.title(f'Rozkład Dwumianowy B(n={n_moneta}, p={p_moneta})')
plt.xticks(k_values)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('/mnt/user-data/outputs/zadanie_2_1_binom.png', dpi=300, bbox_inches='tight')
print("\n4. Wizualizacja zapisana jako 'zadanie_2_1_binom.png'")
plt.close()

# 5. Symulacja
print("\n5. Symulacja (1000 prób):")
samples_binom = np.random.binomial(n_moneta, p_moneta, size=1000)
mean_symulacja = np.mean(samples_binom)
print(f"   Średnia z symulacji: {mean_symulacja:.2f}")
print(f"   Średnia teoretyczna: {mean_binom}")
print(f"   Różnica: {abs(mean_symulacja - mean_binom):.2f}")

# ----------------------------------------------------------------------------
# Zadanie 2.2: Rozkład Dwumianowy - Egzamin
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("ZADANIE 2.2: Rozkład Dwumianowy - Egzamin")
print("-" * 80)

# 1. Parametry
n_egzamin = 25
p_egzamin = 0.25
print(f"\n1. Parametry: n={n_egzamin}, p={p_egzamin}")

# 2. Prawdopodobieństwo zdania
print("\n2. Prawdopodobieństwo zdania (≥13 poprawnych):")
prob_zdanie = 1 - stats.binom.cdf(12, n_egzamin, p_egzamin)
print(f"   P(X ≥ 13) = {prob_zdanie:.6f} ({prob_zdanie * 100:.4f}%)")
print(f"   Praktycznie niemożliwe zdać zgadując!")

# 3. Różne prawdopodobieństwa
print("\n3. Inne prawdopodobieństwa:")
prob_half = stats.binom.pmf(12, n_egzamin, p_egzamin) + stats.binom.pmf(13, n_egzamin, p_egzamin)
prob_zero = stats.binom.pmf(0, n_egzamin, p_egzamin)
prob_all = stats.binom.pmf(25, n_egzamin, p_egzamin)

print(f"   P(X = 12 lub 13): {prob_half:.6f}")
print(f"   P(X = 0):         {prob_zero:.6f}")
print(f"   P(X = 25):        {prob_all:.2e} (prawie niemożliwe)")

# 4. Moda
print("\n4. Najbardziej prawdopodobna liczba poprawnych:")
k_egzamin = np.arange(0, n_egzamin + 1)
prob_egzamin = stats.binom.pmf(k_egzamin, n_egzamin, p_egzamin)
moda_idx = np.argmax(prob_egzamin)
print(f"   Moda: {k_egzamin[moda_idx]} poprawnych odpowiedzi")
print(f"   P(X = {k_egzamin[moda_idx]}) = {prob_egzamin[moda_idx]:.4f}")

# 5. Wizualizacja
plt.figure(figsize=(12, 6))
colors = ['red' if k >= 13 else 'skyblue' for k in k_egzamin]
plt.bar(k_egzamin, prob_egzamin, alpha=0.7, edgecolor='black', color=colors)
plt.xlabel('Liczba poprawnych odpowiedzi')
plt.ylabel('Prawdopodobieństwo')
plt.title(f'Rozkład Dwumianowy - Egzamin (n={n_egzamin}, p={p_egzamin})')
plt.axvline(12.5, color='green', linestyle='--', linewidth=2, label='Granica zdania')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('/mnt/user-data/outputs/zadanie_2_2_egzamin.png', dpi=300, bbox_inches='tight')
print("\n5. Wizualizacja zapisana")
plt.close()

# ----------------------------------------------------------------------------
# Zadanie 2.3: Rozkład Poissona
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("ZADANIE 2.3: Rozkład Poissona - Klienci w Sklepie")
print("-" * 80)

# 1. Parametr
lambda_sklep = 5
print(f"\n1. Parametr: λ = {lambda_sklep} klientów/godz")

# 2. Prawdopodobieństwa (1 godz)
print("\n2. Prawdopodobieństwa (1 godzina):")
prob_3 = stats.poisson.pmf(3, lambda_sklep)
prob_5 = stats.poisson.pmf(5, lambda_sklep)
prob_0 = stats.poisson.pmf(0, lambda_sklep)
prob_geq_8 = 1 - stats.poisson.cdf(7, lambda_sklep)

print(f"   P(X = 3):  {prob_3:.4f}")
print(f"   P(X = 5):  {prob_5:.4f}")
print(f"   P(X = 0):  {prob_0:.4f}")
print(f"   P(X ≥ 8):  {prob_geq_8:.4f}")

# 3. Dla 2 godzin
print("\n3. Prawdopodobieństwa (2 godziny):")
lambda_2h = 2 * lambda_sklep
prob_10_2h = stats.poisson.pmf(10, lambda_2h)
prob_leq_5_2h = stats.poisson.cdf(5, lambda_2h)

print(f"   λ = {lambda_2h}")
print(f"   P(X = 10): {prob_10_2h:.4f}")
print(f"   P(X ≤ 5):  {prob_leq_5_2h:.4f}")

# 4. Dla 30 minut
print("\n4. Prawdopodobieństwa (30 minut):")
lambda_30min = 0.5 * lambda_sklep
prob_geq_1_30min = 1 - stats.poisson.pmf(0, lambda_30min)

print(f"   λ = {lambda_30min}")
print(f"   P(X ≥ 1): {prob_geq_1_30min:.4f}")

# 5. Wizualizacja
fig, ax = plt.subplots(figsize=(14, 6))
k_poisson = np.arange(0, 25)

for lam, label in [(2, 'λ=2'), (5, 'λ=5'), (10, 'λ=10')]:
    prob_temp = stats.poisson.pmf(k_poisson, lam)
    ax.plot(k_poisson, prob_temp, marker='o', label=label, linewidth=2)

ax.set_xlabel('k (liczba klientów)')
ax.set_ylabel('P(X=k)')
ax.set_title('Porównanie Rozkładów Poissona')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('/mnt/user-data/outputs/zadanie_2_3_poisson.png', dpi=300, bbox_inches='tight')
print("\n5. Wizualizacja zapisana")
plt.close()

# 6. Symulacja
print("\n6. Symulacja (100 obserwacji):")
samples_poisson = np.random.poisson(lambda_sklep, size=100)
mean_poisson_sim = np.mean(samples_poisson)
print(f"   Średnia z symulacji: {mean_poisson_sim:.2f}")
print(f"   Średnia teoretyczna: {lambda_sklep}")

# ----------------------------------------------------------------------------
# Zadanie 2.4: Rozkład Normalny - Wzrost
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("ZADANIE 2.4: Rozkład Normalny - Wzrost Ludzi")
print("-" * 80)

# 1. Parametry
mu_wzrost = 178
sigma_wzrost = 7
print(f"\n1. Parametry: μ = {mu_wzrost} cm, σ = {sigma_wzrost} cm")

# 2. Prawdopodobieństwa
print("\n2. Prawdopodobieństwa:")
prob_lt_170 = stats.norm.cdf(170, mu_wzrost, sigma_wzrost)
prob_gt_185 = 1 - stats.norm.cdf(185, mu_wzrost, sigma_wzrost)
prob_170_185 = stats.norm.cdf(185, mu_wzrost, sigma_wzrost) - stats.norm.cdf(170, mu_wzrost, sigma_wzrost)
prob_175_180 = stats.norm.cdf(180, mu_wzrost, sigma_wzrost) - stats.norm.cdf(175, mu_wzrost, sigma_wzrost)

print(f"   P(X < 170):       {prob_lt_170:.4f} ({prob_lt_170 * 100:.2f}%)")
print(f"   P(X > 185):       {prob_gt_185:.4f} ({prob_gt_185 * 100:.2f}%)")
print(f"   P(170 < X < 185): {prob_170_185:.4f} ({prob_170_185 * 100:.2f}%)")
print(f"   P(175 < X < 180): {prob_175_180:.4f} ({prob_175_180 * 100:.2f}%)")

# 3. Percentyle
print("\n3. Percentyle:")
q25 = stats.norm.ppf(0.25, mu_wzrost, sigma_wzrost)
q50 = stats.norm.ppf(0.50, mu_wzrost, sigma_wzrost)
q75 = stats.norm.ppf(0.75, mu_wzrost, sigma_wzrost)
q95 = stats.norm.ppf(0.95, mu_wzrost, sigma_wzrost)
q99 = stats.norm.ppf(0.99, mu_wzrost, sigma_wzrost)

print(f"   25% percentyl (Q1): {q25:.2f} cm")
print(f"   50% percentyl (Me): {q50:.2f} cm")
print(f"   75% percentyl (Q3): {q75:.2f} cm")
print(f"   95% percentyl:      {q95:.2f} cm")
print(f"   99% percentyl:      {q99:.2f} cm")

# 4. Pytania odwrotne
print("\n4. Pytania odwrotne:")
najnizszych_10 = stats.norm.ppf(0.10, mu_wzrost, sigma_wzrost)
najwyzszych_5 = stats.norm.ppf(0.95, mu_wzrost, sigma_wzrost)

print(f"   Najniższych 10%:  < {najnizszych_10:.2f} cm")
print(f"   Najwyższych 5%:   > {najwyzszych_5:.2f} cm")
print(f"   Środkowych 50%:   [{q25:.2f}, {q75:.2f}] cm")

# 5. Reguła 68-95-99.7
print("\n5. Reguła 68-95-99.7:")
przedzialy = [
    (mu_wzrost - sigma_wzrost, mu_wzrost + sigma_wzrost, "68%"),
    (mu_wzrost - 2 * sigma_wzrost, mu_wzrost + 2 * sigma_wzrost, "95%"),
    (mu_wzrost - 3 * sigma_wzrost, mu_wzrost + 3 * sigma_wzrost, "99.7%")
]

for lower, upper, teoria in przedzialy:
    faktyczne = stats.norm.cdf(upper, mu_wzrost, sigma_wzrost) - stats.norm.cdf(lower, mu_wzrost, sigma_wzrost)
    print(f"   [{lower:.0f}, {upper:.0f}]: teoria={teoria}, faktyczne={faktyczne * 100:.2f}%")

# 6. Wizualizacja
x_wzrost = np.linspace(mu_wzrost - 4 * sigma_wzrost, mu_wzrost + 4 * sigma_wzrost, 1000)
y_wzrost = stats.norm.pdf(x_wzrost, mu_wzrost, sigma_wzrost)

plt.figure(figsize=(14, 6))
plt.plot(x_wzrost, y_wzrost, 'b-', linewidth=2, label='PDF')
plt.fill_between(x_wzrost, y_wzrost, where=(x_wzrost < 170), alpha=0.3, color='red', label='P(X<170)')
plt.fill_between(x_wzrost, y_wzrost, where=(x_wzrost > 185), alpha=0.3, color='orange', label='P(X>185)')
plt.fill_between(x_wzrost, y_wzrost, where=((x_wzrost >= 170) & (x_wzrost <= 185)),
                 alpha=0.3, color='green', label='P(170<X<185)')
plt.axvline(mu_wzrost, color='black', linestyle='--', linewidth=2, label=f'μ = {mu_wzrost}')
plt.axvline(mu_wzrost - sigma_wzrost, color='gray', linestyle=':', alpha=0.7)
plt.axvline(mu_wzrost + sigma_wzrost, color='gray', linestyle=':', alpha=0.7, label='μ±σ')
plt.axvline(mu_wzrost - 2 * sigma_wzrost, color='gray', linestyle=':', alpha=0.5)
plt.axvline(mu_wzrost + 2 * sigma_wzrost, color='gray', linestyle=':', alpha=0.5, label='μ±2σ')
plt.xlabel('Wzrost [cm]')
plt.ylabel('Gęstość prawdopodobieństwa')
plt.title(f'Rozkład Normalny N(μ={mu_wzrost}, σ={sigma_wzrost})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/mnt/user-data/outputs/zadanie_2_4_normalny.png', dpi=300, bbox_inches='tight')
print("\n6. Wizualizacja zapisana")
plt.close()

# 7. Standaryzacja
print("\n7. Standaryzacja:")
x_190 = 190
z_score = (x_190 - mu_wzrost) / sigma_wzrost
print(f"   Wzrost {x_190} cm")
print(f"   Z-score: {z_score:.2f}")
print(
    f"   Interpretacja: {x_190} cm jest {abs(z_score):.2f} odchyleń std {'powyżej' if z_score > 0 else 'poniżej'} średniej")

# ----------------------------------------------------------------------------
# Zadanie 2.5: Rozkład Normalny - IQ
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("ZADANIE 2.5: Rozkład Normalny - Wyniki IQ")
print("-" * 80)

# 1. Parametry
mu_iq = 100
sigma_iq = 15
print(f"\n1. Parametry: μ = {mu_iq}, σ = {sigma_iq}")

# 2. Klasyfikacja IQ
print("\n2. Klasyfikacja IQ:")
kategorie = [
    ("Bardzo niskie", 0, 70),
    ("Niskie", 70, 85),
    ("Przeciętne", 85, 115),
    ("Wysokie", 115, 130),
    ("Bardzo wysokie", 130, 200)
]

for nazwa, lower, upper in kategorie:
    if lower == 0:
        prob = stats.norm.cdf(upper, mu_iq, sigma_iq)
    elif upper == 200:
        prob = 1 - stats.norm.cdf(lower, mu_iq, sigma_iq)
    else:
        prob = stats.norm.cdf(upper, mu_iq, sigma_iq) - stats.norm.cdf(lower, mu_iq, sigma_iq)
    print(f"   {nazwa:20s}: {prob * 100:6.2f}%")

# 3. Symulacja
print("\n3. Symulacja (10000 osób):")
iq_samples = np.random.normal(mu_iq, sigma_iq, size=10000)
mean_iq = np.mean(iq_samples)
std_iq = np.std(iq_samples)

print(f"   Średnia z próbki: {mean_iq:.2f} (teoretyczna: {mu_iq})")
print(f"   Odch. std z próbki: {std_iq:.2f} (teoretyczne: {sigma_iq})")

# 4. Histogram vs rozkład teoretyczny
plt.figure(figsize=(12, 6))
plt.hist(iq_samples, bins=50, density=True, alpha=0.7, edgecolor='black', label='Histogram (symulacja)')
x_iq = np.linspace(40, 160, 1000)
y_iq = stats.norm.pdf(x_iq, mu_iq, sigma_iq)
plt.plot(x_iq, y_iq, 'r-', linewidth=2, label='Rozkład teoretyczny')
plt.xlabel('IQ')
plt.ylabel('Gęstość')
plt.title('Rozkład IQ - Symulacja vs Teoria')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/mnt/user-data/outputs/zadanie_2_5_iq_hist.png', dpi=300, bbox_inches='tight')
print("\n4. Histogram zapisany")
plt.close()

# 5. Test normalności
print("\n5. Test normalności:")
shapiro_stat, shapiro_p = stats.shapiro(iq_samples[:5000])  # Max 5000 dla Shapiro
print(f"   Test Shapiro-Wilka (5000 próbek):")
print(f"   Statystyka: {shapiro_stat:.4f}")
print(f"   P-value: {shapiro_p:.4f}")
if shapiro_p > 0.05:
    print(f"   Dane MAJĄ rozkład normalny ✓")
else:
    print(f"   Dane NIE MAJĄ rozkładu normalnego ✗")

# Q-Q plot
fig, ax = plt.subplots(figsize=(8, 8))
stats.probplot(iq_samples, dist="norm", plot=ax)
ax.set_title('Q-Q Plot - Test Normalności')
ax.grid(True, alpha=0.3)
plt.savefig('/mnt/user-data/outputs/zadanie_2_5_iq_qq.png', dpi=300, bbox_inches='tight')
print("   Q-Q plot zapisany")
plt.close()

# ----------------------------------------------------------------------------
# Zadanie 2.6: Centralne Twierdzenie Graniczne
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("ZADANIE 2.6: Centralne Twierdzenie Graniczne")
print("-" * 80)

# 1. Wybór rozkładu - Jednostajny
print("\n1. Rozkład początkowy: Jednostajny U(0, 10)")

# 2-4. Generowanie dla różnych n
sample_sizes = [5, 10, 30, 100]
num_samples = 1000

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Centralne Twierdzenie Graniczne - Rozkład Jednostajny', fontsize=16, fontweight='bold')

for idx, n in enumerate(sample_sizes):
    row = idx // 2
    col = idx % 2

    # Generowanie średnich
    means_list = []
    for _ in range(num_samples):
        sample = np.random.uniform(0, 10, n)
        means_list.append(np.mean(sample))

    means_array = np.array(means_list)

    # Histogram
    axes[row, col].hist(means_array, bins=30, density=True, alpha=0.7,
                        edgecolor='black', color='skyblue', label='Rozkład średnich')

    # Krzywa normalna
    mu_fit = np.mean(means_array)
    sigma_fit = np.std(means_array)
    x_fit = np.linspace(means_array.min(), means_array.max(), 100)
    y_fit = stats.norm.pdf(x_fit, mu_fit, sigma_fit)
    axes[row, col].plot(x_fit, y_fit, 'r-', linewidth=2, label='Rozkład normalny')

    axes[row, col].set_xlabel('Średnia z próby')
    axes[row, col].set_ylabel('Gęstość')
    axes[row, col].set_title(f'n = {n}')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

    # 5. Statystyki
    print(f"\nn = {n}:")
    print(f"   Średnia rozkładu średnich: {mu_fit:.4f}")
    print(f"   Odch. std rozkładu średnich: {sigma_fit:.4f}")
    print(f"   Teoretyczne σ/√n: {10 / np.sqrt(12) / np.sqrt(n):.4f}")

    # 6. Test normalności
    if n >= 30:  # Shapiro działa lepiej dla większych n
        _, p_shapiro_ctg = stats.shapiro(means_array[:5000])
        print(f"   Test Shapiro-Wilka: p-value = {p_shapiro_ctg:.4f}")

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/zadanie_2_6_ctg.png', dpi=300, bbox_inches='tight')
print("\nWizualizacja CTG zapisana")
plt.close()

print("\n" + "=" * 80)
print("WNIOSEK CTG:")
print("Niezależnie od rozkładu populacji (tutaj jednostajny),")
print("rozkład średnich zbliża się do normalnego wraz ze wzrostem n!")
print("=" * 80)

# ============================================================================
# ZESTAW 3: REGRESJA LINIOWA
# ============================================================================

print("\n" + "=" * 80)
print("ZESTAW 3: REGRESJA LINIOWA")
print("=" * 80)

# ----------------------------------------------------------------------------
# Zadanie 3.1: Godziny Nauki vs Wynik
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("ZADANIE 3.1: Godziny Nauki vs Wynik")
print("-" * 80)

# 1. Dane
X_nauka = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y_wynik = np.array([45, 55, 60, 68, 75, 82, 85, 90])

print("\n1. Dane:")
print(f"   X (godziny): {X_nauka.flatten()}")
print(f"   y (wynik): {y_wynik}")

# 2. Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(X_nauka, y_wynik, s=100, color='blue', alpha=0.6, edgecolors='black', linewidth=2)
plt.xlabel('Godziny nauki')
plt.ylabel('Wynik [pkt]')
plt.title('Godziny Nauki vs Wynik Egzaminu')
plt.grid(True, alpha=0.3)
plt.savefig('/mnt/user-data/outputs/zadanie_3_1_scatter.png', dpi=300, bbox_inches='tight')
print("\n2. Scatter plot zapisany")
plt.close()

# 3. Korelacja
corr = np.corrcoef(X_nauka.flatten(), y_wynik)[0, 1]
print(f"\n3. Współczynnik korelacji Pearsona: {corr:.4f}")
print(
    f"   Interpretacja: {'Silna' if abs(corr) > 0.7 else 'Umiarkowana' if abs(corr) > 0.4 else 'Słaba'} korelacja {'dodatnia' if corr > 0 else 'ujemna'}")

# 4. Model regresji
model_nauka = LinearRegression()
model_nauka.fit(X_nauka, y_wynik)

beta_0 = model_nauka.intercept_
beta_1 = model_nauka.coef_[0]

print(f"\n4. Model regresji:")
print(f"   β₀ (intercept): {beta_0:.4f}")
print(f"   β₁ (slope): {beta_1:.4f}")
print(f"   Równanie: ŷ = {beta_0:.2f} + {beta_1:.2f}x")

# 5. Metryki
y_pred_nauka = model_nauka.predict(X_nauka)
r2_nauka = r2_score(y_wynik, y_pred_nauka)
rmse_nauka = np.sqrt(mean_squared_error(y_wynik, y_pred_nauka))
mae_nauka = mean_absolute_error(y_wynik, y_pred_nauka)

print(f"\n5. Metryki:")
print(f"   R²:   {r2_nauka:.4f} (model wyjaśnia {r2_nauka * 100:.2f}% zmienności)")
print(f"   RMSE: {rmse_nauka:.4f} pkt")
print(f"   MAE:  {mae_nauka:.4f} pkt")

# 6. Wizualizacja modelu
plt.figure(figsize=(12, 6))
plt.scatter(X_nauka, y_wynik, s=100, color='blue', alpha=0.6,
            edgecolors='black', linewidth=2, label='Dane rzeczywiste')
plt.plot(X_nauka, y_pred_nauka, 'r-', linewidth=2,
         label=f'ŷ = {beta_0:.2f} + {beta_1:.2f}x')
plt.xlabel('Godziny nauki')
plt.ylabel('Wynik [pkt]')
plt.title(f'Regresja Liniowa (R² = {r2_nauka:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.text(1, 85, f'R² = {r2_nauka:.4f}\nRMSE = {rmse_nauka:.2f}',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.savefig('/mnt/user-data/outputs/zadanie_3_1_regresja.png', dpi=300, bbox_inches='tight')
print("\n6. Wizualizacja modelu zapisana")
plt.close()

# 7. Przewidywania
print(f"\n7. Przewidywania:")
new_X = np.array([[9], [3.5]])
new_y = model_nauka.predict(new_X)
print(f"   Dla 9 godzin nauki → {new_y[0]:.2f} pkt")
print(f"   Dla 3.5 godziny nauki → {new_y[1]:.2f} pkt")

# 8. Analiza reszt
residuals = y_wynik - y_pred_nauka

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residual plot
axes[0].scatter(y_pred_nauka, residuals, s=100, color='purple',
                alpha=0.6, edgecolors='black', linewidth=2)
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Wartości przewidywane')
axes[0].set_ylabel('Reszty')
axes[0].set_title('Wykres Reszt')
axes[0].grid(True, alpha=0.3)

# Histogram reszt
axes[1].hist(residuals, bins=5, edgecolor='black', alpha=0.7, color='coral')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Reszty')
axes[1].set_ylabel('Częstość')
axes[1].set_title('Rozkład Reszt')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/zadanie_3_1_reszty.png', dpi=300, bbox_inches='tight')
print("\n8. Analiza reszt zapisana")
plt.close()

# ----------------------------------------------------------------------------
# Zadanie 3.4: Regresja Wieloraka - Cena Mieszkania
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("ZADANIE 3.4: Regresja Wieloraka - Cena Mieszkania")
print("-" * 80)

# 1. Generowanie danych
np.random.seed(42)
n_mieszkania = 50

powierzchnia_m = np.random.uniform(30, 150, n_mieszkania)
liczba_pokoi_m = np.random.randint(1, 6, n_mieszkania)
pietro_m = np.random.randint(0, 11, n_mieszkania)
wiek_m = np.random.uniform(0, 50, n_mieszkania)

cena_m = (100000 +
          5000 * powierzchnia_m +
          15000 * liczba_pokoi_m +
          2000 * pietro_m -
          1000 * wiek_m +
          np.random.normal(0, 40000, n_mieszkania))

print("\n1. Dane wygenerowane (50 obserwacji)")

# 2. DataFrame
df_mieszkania = pd.DataFrame({
    'powierzchnia': powierzchnia_m,
    'liczba_pokoi': liczba_pokoi_m,
    'pietro': pietro_m,
    'wiek': wiek_m,
    'cena': cena_m
})

print("\n2. DataFrame:")
print(df_mieszkania.head())

# 3. Analiza eksploracyjna
print("\n3. Statystyki opisowe:")
print(df_mieszkania.describe())

# Macierz korelacji
print("\n   Macierz korelacji z ceną:")
print(df_mieszkania.corr()['cena'].sort_values(ascending=False))

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_mieszkania.corr(), annot=True, cmap='coolwarm', center=0, fmt='.3f')
plt.title('Macierz Korelacji')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/zadanie_3_4_korelacja.png', dpi=300, bbox_inches='tight')
print("\n   Heatmap zapisany")
plt.close()

# 4. Przygotowanie danych
X_mieszkania = df_mieszkania[['powierzchnia', 'liczba_pokoi', 'pietro', 'wiek']]
y_mieszkania = df_mieszkania['cena']

print(f"\n4. Przygotowanie danych:")
print(f"   Shape X: {X_mieszkania.shape}")
print(f"   Shape y: {y_mieszkania.shape}")

# 5. Model regresji wielorakiej
model_mieszkania = LinearRegression()
model_mieszkania.fit(X_mieszkania, y_mieszkania)

print(f"\n5. Współczynniki modelu:")
print(f"   β₀ (intercept): {model_mieszkania.intercept_:,.2f} zł")
for feature, coef in zip(X_mieszkania.columns, model_mieszkania.coef_):
    print(f"   β ({feature}): {coef:,.2f} zł")

print(f"\n   Interpretacja współczynników:")
print(f"   - Każdy m² zwiększa cenę o {model_mieszkania.coef_[0]:,.0f} zł")
print(f"   - Każdy pokój zwiększa cenę o {model_mieszkania.coef_[1]:,.0f} zł")
print(f"   - Każde piętro zwiększa cenę o {model_mieszkania.coef_[2]:,.0f} zł")
print(f"   - Każdy rok wieku zmniejsza cenę o {abs(model_mieszkania.coef_[3]):,.0f} zł")

# 6. Metryki
y_pred_mieszkania = model_mieszkania.predict(X_mieszkania)
r2_mieszkania = r2_score(y_mieszkania, y_pred_mieszkania)
rmse_mieszkania = np.sqrt(mean_squared_error(y_mieszkania, y_pred_mieszkania))

n = len(y_mieszkania)
p = X_mieszkania.shape[1]
r2_adj_mieszkania = 1 - ((1 - r2_mieszkania) * (n - 1) / (n - p - 1))

print(f"\n6. Metryki:")
print(f"   R²:           {r2_mieszkania:.4f}")
print(f"   Adjusted R²:  {r2_adj_mieszkania:.4f}")
print(f"   RMSE:         {rmse_mieszkania:,.2f} zł")

# Model tylko z powierzchnią dla porównania
model_simple = LinearRegression()
model_simple.fit(X_mieszkania[['powierzchnia']], y_mieszkania)
y_pred_simple = model_simple.predict(X_mieszkania[['powierzchnia']])
r2_simple = r2_score(y_mieszkania, y_pred_simple)

print(f"\n   Porównanie z modelem prostym (tylko powierzchnia):")
print(f"   R² (prosty):    {r2_simple:.4f}")
print(f"   R² (wieloraki): {r2_mieszkania:.4f}")
print(f"   Poprawa:        {(r2_mieszkania - r2_simple) * 100:.2f}%")

# 7. Przewidywanie
print(f"\n7. Przewidywanie:")
przyklad = pd.DataFrame({
    'powierzchnia': [60],
    'liczba_pokoi': [3],
    'pietro': [5],
    'wiek': [10]
})
przewidywanie = model_mieszkania.predict(przyklad)
print(f"   Mieszkanie: 60m², 3 pokoje, 5 piętro, 10 lat")
print(f"   Przewidywana cena: {przewidywanie[0]:,.2f} zł")

# 8. Wizualizacja
plt.figure(figsize=(10, 6))
plt.scatter(y_mieszkania, y_pred_mieszkania, alpha=0.5, s=50)
plt.plot([y_mieszkania.min(), y_mieszkania.max()],
         [y_mieszkania.min(), y_mieszkania.max()],
         'r--', linewidth=2, label='Idealne dopasowanie')
plt.xlabel('Cena rzeczywista [zł]')
plt.ylabel('Cena przewidywana [zł]')
plt.title(f'Regresja Wieloraka (R² = {r2_mieszkania:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.text(0.05, 0.95, f'R² = {r2_mieszkania:.4f}\nAdj R² = {r2_adj_mieszkania:.4f}',
         transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.savefig('/mnt/user-data/outputs/zadanie_3_4_wieloraka.png', dpi=300, bbox_inches='tight')
print("\n8. Wizualizacja zapisana")
plt.close()

# ----------------------------------------------------------------------------
# Zadanie 3.5: Podział Train/Test
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("ZADANIE 3.5: Podział Train/Test")
print("-" * 80)

# 1. Podział danych
X_train, X_test, y_train, y_test = train_test_split(
    X_mieszkania, y_mieszkania, test_size=0.2, random_state=42
)

print(f"\n1. Podział danych:")
print(f"   Trening: {len(X_train)} obs ({len(X_train) / len(X_mieszkania) * 100:.0f}%)")
print(f"   Test:    {len(X_test)} obs ({len(X_test) / len(X_mieszkania) * 100:.0f}%)")

# 2. Trenowanie
model_validated = LinearRegression()
model_validated.fit(X_train, y_train)

print(f"\n2. Model wytrenowany na zbiorze treningowym")

# 3. Predykcje
y_train_pred = model_validated.predict(X_train)
y_test_pred = model_validated.predict(X_test)

# 4. Metryki
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\n4. Metryki dla obu zbiorów:")
print(f"   {'Metryka':<15} {'Trening':>15} {'Test':>15}")
print(f"   {'-' * 45}")
print(f"   {'R²':<15} {r2_train:>15.4f} {r2_test:>15.4f}")
print(f"   {'RMSE':<15} {rmse_train:>15,.2f} {rmse_test:>15,.2f}")

# 5. Ocena
diff = abs(r2_train - r2_test)
print(f"\n5. Ocena generalizacji:")
print(f"   Różnica R²: {diff:.4f}")
if diff < 0.05:
    print(f"   Model generalizuje DOBRZE ✓")
elif diff < 0.1:
    print(f"   Model może mieć lekkie przeuczenie ⚠")
else:
    print(f"   Model prawdopodobnie PRZEUCZONY ✗")

# 6. Wizualizacja
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
plt.savefig('/mnt/user-data/outputs/zadanie_3_5_train_test.png', dpi=300, bbox_inches='tight')
print("\n6. Wizualizacja zapisana")
plt.close()

# ----------------------------------------------------------------------------
# Zadanie 3.6: Analiza ze Statsmodels
# ----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("ZADANIE 3.6: Analiza ze Statsmodels")
print("-" * 80)

# 1. Przygotowanie danych
X_mieszkania_const = sm.add_constant(X_mieszkania)
print("\n1. Dodano stałą do X")

# 2. Model OLS
model_ols_mieszkania = sm.OLS(y_mieszkania, X_mieszkania_const)
results_ols_mieszkania = model_ols_mieszkania.fit()

print("\n2. Model OLS utworzony i dopasowany")

# 3. Szczegółowe podsumowanie
print("\n3. Szczegółowe podsumowanie:")
print(results_ols_mieszkania.summary())

# 4. Kluczowe informacje
print("\n4. Kluczowe informacje:")
print(f"   R²:            {results_ols_mieszkania.rsquared:.4f}")
print(f"   Adjusted R²:   {results_ols_mieszkania.rsquared_adj:.4f}")
print(f"   F-statistic:   {results_ols_mieszkania.fvalue:.4f}")
print(f"   Prob(F):       {results_ols_mieszkania.f_pvalue:.6f}")

# 5. Test istotności zmiennych
print("\n5. Test istotności zmiennych:")
for param, pval in zip(results_ols_mieszkania.params.index, results_ols_mieszkania.pvalues):
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "NS"
    print(f"   {param:15s}: p = {pval:.6f} {sig}")
    if pval < 0.05:
        print(f"                  → ISTOTNY ✓")
    else:
        print(f"                  → nieistotny, można usunąć ✗")

# 6. Sprawdzenie założeń
print("\n6. Sprawdzenie założeń:")

# Normalność reszt
from scipy.stats import jarque_bera

jb_stat, jb_p = jarque_bera(results_ols_mieszkania.resid)
print(f"   Test Jarque-Bera (normalność):")
print(f"   Statystyka: {jb_stat:.4f}, p-value: {jb_p:.4f}")
if jb_p > 0.05:
    print(f"   Reszty MAJĄ rozkład normalny ✓")
else:
    print(f"   Reszty NIE MAJĄ rozkładu normalnego ✗")

# Durbin-Watson
dw = sm.stats.stattools.durbin_watson(results_ols_mieszkania.resid)
print(f"\n   Test Durbina-Watsona (autokorelacja):")
print(f"   Statystyka: {dw:.4f}")
if 1.5 <= dw <= 2.5:
    print(f"   Brak autokorelacji ✓")
else:
    print(f"   Wykryto autokorelację ✗")

# Breusch-Pagan
bp_test = het_breuschpagan(results_ols_mieszkania.resid, results_ols_mieszkania.model.exog)
print(f"\n   Test Breusch-Pagan (homoskedastyczność):")
print(f"   Statystyka LM: {bp_test[0]:.4f}, p-value: {bp_test[1]:.4f}")
if bp_test[1] > 0.05:
    print(f"   Wariancja reszt STAŁA (homoskedastyczność) ✓")
else:
    print(f"   Wykryto heteroskedastyczność ✗")

# 7. Wykresy diagnostyczne
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals vs Fitted
axes[0, 0].scatter(results_ols_mieszkania.fittedvalues, results_ols_mieszkania.resid, alpha=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Wartości dopasowane')
axes[0, 0].set_ylabel('Reszty')
axes[0, 0].set_title('Reszty vs Wartości Dopasowane')
axes[0, 0].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(results_ols_mieszkania.resid, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot (Normalność)')
axes[0, 1].grid(True, alpha=0.3)

# Histogram reszt
axes[1, 0].hist(results_ols_mieszkania.resid, bins=20, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Reszty')
axes[1, 0].set_ylabel('Częstość')
axes[1, 0].set_title('Histogram Reszt')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Cook's distance
influence = results_ols_mieszkania.get_influence()
cooks_d = influence.cooks_distance[0]
axes[1, 1].stem(range(len(cooks_d)), cooks_d, markerfmt=',')
axes[1, 1].axhline(y=4 / len(y_mieszkania), color='r', linestyle='--', label='Próg (4/n)')
axes[1, 1].set_xlabel('Indeks obserwacji')
axes[1, 1].set_ylabel("Cook's distance")
axes[1, 1].set_title('Odległość Cooka')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/zadanie_3_6_diagnostyka.png', dpi=300, bbox_inches='tight')
print("\n7. Wykresy diagnostyczne zapisane")
plt.close()

# 8. VIF (współliniowość)
print("\n8. VIF (Variance Inflation Factor):")
vif_data = pd.DataFrame()
vif_data["Zmienna"] = X_mieszkania.columns
vif_data["VIF"] = [variance_inflation_factor(X_mieszkania.values, i)
                   for i in range(X_mieszkania.shape[1])]
print(vif_data)

print("\n   Interpretacja:")
for idx, row in vif_data.iterrows():
    if row['VIF'] < 5:
        status = "brak problemu"
    elif row['VIF'] < 10:
        status = "umiarkowana współliniowość"
    else:
        status = "silna współliniowość"
    print(f"   {row['Zmienna']:15s}: VIF = {row['VIF']:.2f} {status}")

# ============================================================================
# PODSUMOWANIE
# ============================================================================

print("\n" + "=" * 80)
print("WSZYSTKIE ZADANIA ROZWIĄZANE!")
print("=" * 80)

print("\nWygenerowane wizualizacje:")
print("  1. zadanie_1_2_wizualizacje.png - Statystyka opisowa")
print("  2. zadanie_1_5_porownanie.png - Test t dwóch grup")
print("  3. zadanie_2_1_binom.png - Rozkład dwumianowy")
print("  4. zadanie_2_2_egzamin.png - Egzamin (dwumianowy)")
print("  5. zadanie_2_3_poisson.png - Rozkład Poissona")
print("  6. zadanie_2_4_normalny.png - Rozkład normalny (wzrost)")
print("  7. zadanie_2_5_iq_hist.png - IQ histogram")
print("  8. zadanie_2_5_iq_qq.png - IQ Q-Q plot")
print("  9. zadanie_2_6_ctg.png - Centralne Twierdzenie Graniczne")
print(" 10. zadanie_3_1_scatter.png - Scatter plot")
print(" 11. zadanie_3_1_regresja.png - Regresja prosta")
print(" 12. zadanie_3_1_reszty.png - Analiza reszt")
print(" 13. zadanie_3_4_korelacja.png - Macierz korelacji")
print(" 14. zadanie_3_4_wieloraka.png - Regresja wieloraka")
print(" 15. zadanie_3_5_train_test.png - Walidacja modelu")
print(" 16. zadanie_3_6_diagnostyka.png - Diagnostyka (statsmodels)")

print("\n" + "=" * 80)
print("KONIEC ROZWIĄZAŃ")
print("=" * 80)