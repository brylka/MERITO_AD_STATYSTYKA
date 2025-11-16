import numpy as np
from scipy import stats



# ----------------------------------------------------------------------------
# Zadanie 1.4: Test t-Studenta dla Jednej Próbki
# ----------------------------------------------------------------------------

print("\n" + "-"*80)
print("ZADANIE 1.4: Test t-Studenta dla Jednej Próbki")
print("-"*80)

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
print(f"   H0: u = {mu_0} (twierdzenie producenta)")
print(f"   H1: u != {mu_0} (test dwustronny)")

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
    print(f"   ODRZUCAMY H0")
    print(f"   Wniosek: Średni czas pracy baterii RÓŻNI SIĘ od {mu_0} godz")
else:
    print(f"   p-value ({p_value:.4f}) >= α ({alpha})")
    print(f"   NIE ODRZUCAMY H0")
    print(f"   Wniosek: Brak podstaw do odrzucenia twierdzenia producenta")

# 6. Przedział ufności
print("\n6. Przedział ufności (95%):")
ci = stats.t.interval(0.95, n_baterie-1,
                      loc=mean_baterie,
                      scale=stats.sem(baterie))
print(f"   Przedział: [{ci[0]:.4f}, {ci[1]:.4f}] godz")
if ci[0] <= mu_0 <= ci[1]:
    print(f"   Wartość {mu_0} NALEŻY do przedziału")
else:
    print(f"   Wartość {mu_0} NIE NALEŻY do przedziału")