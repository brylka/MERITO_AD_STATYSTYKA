from scipy import stats
import numpy as np

print("="*50,"TEST T-STUDENTA DLA JEDNEJ PRÓBKI","="*50,sep="\n")

# dane zmierzonych baterii
baterie = np.array([98, 102, 101, 97, 103, 99, 100, 101, 98, 102])
# wartość z hipotezy
mu_0 = 100

# statystyki opisowe
print("STATYSTYKI OPISOWE:")
print("-"*40)
print(f"Liczba obserwacji:          {len(baterie)}")
print(f"Średnia próbki:             {np.mean(baterie):.2f} godz.")
print(f"Odchylenie std próbki:      {np.std(baterie, ddof=1):.2f} godz.")
print(f"Wartość testowaną:          {mu_0} godz.")

# test t-Studenta (1 próbka)
t_statistic, p_value = stats.ttest_1samp(baterie, mu_0)

# wyniki testu
print("WYNIKI TESTU:")
print(f"Statystyka t:               {t_statistic:.2f}")
print(f"P-value:                    {p_value:.2f}")

alpha = 0.05
df = len(baterie) - 1
t_critical = stats.t.ppf(1 -alpha/2, df)
print(f"Wartość krytyczna t_a/2:    {t_critical:.2}")

# decyzja
print("DECYZJA:")

if p_value < alpha:
    print(f"p-value ({p_value:.2f}) < a ({alpha})")
    print("  - ODRZUCAMY H0")
    print("  - Średnia jest istotnie różna od 100 godz.")
else:
    print(f"p-value ({p_value:.2f}) >= a ({alpha})")
    print("  - NIE ODRZUCAMY H0")
    print("  - Brak podstaw do odrzucenia twierdzenia producenta")