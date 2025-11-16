import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import comb

# Konfiguracja
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
np.random.seed(42)


print("="*60)
print("ROZKŁAD DWUMIANOWY - B(n=10, p=0.5)")
print("="*60)

# Parametry
n = 10  # liczba prób
p = 0.5  # prawdopodobieństwo sukcesu

# a) P(X = 6)
prob_6 = stats.binom.pmf(6, n, p)
print(f"\na) P(X = 6) = {prob_6:.4f} = {prob_6*100:.2f}%")

# b) Wartość oczekiwana
mean = stats.binom.mean(n, p)
var = stats.binom.var(n, p)
std = stats.binom.std(n, p)
print(f"\nb) Wartość oczekiwana: {mean}")
print(f"   Wariancja: {var}")
print(f"   Odchylenie std: {std:.2f}")

# c) P(X ≥ 8)
prob_geq_8 = 1 - stats.binom.cdf(7, n, p)  # P(X ≥ 8) = 1 - P(X ≤ 7)
print(f"\nc) P(X ≥ 8) = {prob_geq_8:.4f} = {prob_geq_8*100:.2f}%")

# Alternatywnie
prob_geq_8_alt = stats.binom.pmf(8, n, p) + stats.binom.pmf(9, n, p) + stats.binom.pmf(10, n, p)
print(f"   Sprawdzenie: {prob_geq_8_alt:.4f}")

# Wizualizacja
k_values = np.arange(0, n+1)
probabilities = stats.binom.pmf(k_values, n, p)

plt.figure(figsize=(12, 5))

# Wykres słupkowy
plt.subplot(1, 2, 1)
plt.bar(k_values, probabilities, alpha=0.7, edgecolor='black', color='skyblue')
plt.bar(6, stats.binom.pmf(6, n, p), color='red', alpha=0.8, label='P(X=6)')
plt.axvline(mean, color='green', linestyle='--', linewidth=2, label=f'E(X)={mean}')
plt.xlabel('Liczba orłów (k)')
plt.ylabel('Prawdopodobieństwo P(X=k)')
plt.title(f'Rozkład Dwumianowy B(n={n}, p={p})')
plt.xticks(k_values)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Dystrybuanta (CDF)
plt.subplot(1, 2, 2)
cdf_values = stats.binom.cdf(k_values, n, p)
plt.step(k_values, cdf_values, where='post', linewidth=2, color='blue')
plt.xlabel('k')
plt.ylabel('P(X ≤ k)')
plt.title('Dystrybuanta (CDF)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Generowanie próbek
print("\nGENEROWANIE PRÓBEK:")
samples = np.random.binomial(n, p, size=1000)
print(f"Wygenerowano 1000 próbek")
print(f"Średnia z próbek: {np.mean(samples):.2f} (oczekiwane: {mean})")
print(f"Odch. std z próbek: {np.std(samples):.2f} (oczekiwane: {std:.2f})")
