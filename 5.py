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
print("ZADANIE 2.1.")
print("="*60)

# Parametry
n = 20
p = 0.5
print(f"\nParametry: n={n}, p={p}")

# Prawdopodobieństwa
prob_10 = stats.binom.pmf(10, n, p)
prob_15 = stats.binom.pmf(15, n, p)
prob_8 = stats.binom.cdf(8, n, p)
prob_12 = 1 - stats.binom.cdf(11, n, p)
print(f"P(X = 10) = {prob_10:.4f} = {prob_10*100:.2f}%")
print(f"P(X = 15) = {prob_15:.4f} = {prob_15*100:.2f}%")
print(f"P(X ≤ 8) = {prob_8:.4f} = {prob_8*100:.2f}%")
print(f"P(X ≥ 12) = {prob_12:.4f} = {prob_12*100:.2f}%")

# Charakterystyki
mean = stats.binom.mean(n, p)
var = stats.binom.var(n, p)
std = stats.binom.std(n, p)
print(f"Wartość oczekiwana: {mean}")
print(f"Wariancja: {var}")
print(f"Odchylenie std: {std:.2f}")

# Wizualizacja
k_values = np.arange(0, n+1)
probabilities = stats.binom.pmf(k_values, n, p)

plt.figure(figsize=(12, 5))

# Wykres słupkowy
plt.bar(k_values, probabilities, alpha=0.7, edgecolor='black', color='skyblue')
plt.axvline(mean, color='green', linestyle='--', linewidth=2, label=f'E(X)={mean}')
plt.xlabel('Liczba orłów (k)')
plt.ylabel('Prawdopodobieństwo P(X=k)')
plt.title(f'Rozkład Dwumianowy B(n={n}, p={p})')
plt.xticks(k_values)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# Generowanie próbek
print("\nGENEROWANIE PRÓBEK:")
samples = np.random.binomial(n, p, size=1000)
print(f"Wygenerowano 1000 próbek")
print(f"Średnia z próbek: {np.mean(samples):.2f} (oczekiwane: {mean})")
print(f"Odch. std z próbek: {np.std(samples):.2f} (oczekiwane: {std:.2f})")
