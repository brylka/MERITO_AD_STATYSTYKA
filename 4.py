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


print("\n" + "="*60)
print("ROZKŁAD NORMALNY - N(μ=178, σ=7)")
print("="*60)

# Parametry
mu = 178  # średnia
sigma = 7  # odchylenie standardowe

# a) P(171 < X < 185)
prob_a = stats.norm.cdf(185, mu, sigma) - stats.norm.cdf(171, mu, sigma)
print(f"\na) P(171 < X < 185) = {prob_a:.4f} = {prob_a*100:.2f}%")

# b) 95. percentyl (top 5%)
percentile_95 = stats.norm.ppf(0.95, mu, sigma)
print(f"\nb) Wzrost dla top 5%: {percentile_95:.2f} cm")

# c) P(X > 190)
prob_c = 1 - stats.norm.cdf(190, mu, sigma)
print(f"\nc) P(X > 190) = {prob_c:.4f} = {prob_c*100:.2f}%")

# Reguła 68-95-99.7
print("\nREGUŁA 68-95-99.7:")
print(f"68% danych: [{mu-sigma:.1f}, {mu+sigma:.1f}] cm")
print(f"95% danych: [{mu-2*sigma:.1f}, {mu+2*sigma:.1f}] cm")
print(f"99.7% danych: [{mu-3*sigma:.1f}, {mu+3*sigma:.1f}] cm")

# Wizualizacja
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
y = stats.norm.pdf(x, mu, sigma)

plt.figure(figsize=(14, 5))

# PDF
plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', linewidth=2, label='PDF')
plt.fill_between(x, y, where=(x >= 171) & (x <= 185),
                 alpha=0.3, color='green', label='P(171 < X < 185)')
plt.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'μ = {mu}')
plt.axvline(mu-sigma, color='orange', linestyle=':', alpha=0.7, label='μ±σ')
plt.axvline(mu+sigma, color='orange', linestyle=':', alpha=0.7)
plt.axvline(percentile_95, color='purple', linestyle='--',
           label=f'95% percentyl = {percentile_95:.1f}')
plt.xlabel('Wzrost [cm]')
plt.ylabel('Gęstość prawdopodobieństwa')
plt.title(f'Rozkład Normalny N(μ={mu}, σ={sigma})')
plt.legend()
plt.grid(True, alpha=0.3)

# Standaryzacja
plt.subplot(1, 2, 2)
z = np.linspace(-4, 4, 1000)
y_standard = stats.norm.pdf(z, 0, 1)
plt.plot(z, y_standard, 'b-', linewidth=2)
plt.fill_between(z, y_standard, where=(z >= -1) & (z <= 1),
                 alpha=0.3, color='green', label='68% (±1σ)')
plt.fill_between(z, y_standard, where=(z >= -2) & (z <= -1),
                 alpha=0.2, color='yellow')
plt.fill_between(z, y_standard, where=(z >= 1) & (z <= 2),
                 alpha=0.2, color='yellow', label='95% (±2σ)')
plt.xlabel('Z-score')
plt.ylabel('Gęstość')
plt.title('Standardowy Rozkład Normalny N(0,1)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Generowanie próbek
print("\nGENEROWANIE PRÓBEK:")
samples_normal = np.random.normal(mu, sigma, size=1000)
print(f"Średnia z próbek: {np.mean(samples_normal):.2f} (oczekiwane: {mu})")
print(f"Odch. std z próbek: {np.std(samples_normal):.2f} (oczekiwane: {sigma})")

