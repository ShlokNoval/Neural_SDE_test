import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm

# === Load trajectory data ===
df = pd.read_excel("water_X_trajectory_500fps_30kframes_0.02.xlsx", header=None)
x = df[0].values  # Position values

# === True moments from actual data ===
mean_x = np.mean(x)
std_x = np.std(x)
skew_x = skew(x)
kurt_x = kurtosis(x, fisher=False)

# === Define a sample hidden physics function (drift g₁(x)) ===
# Replace this with your trained g₁(x) function if available
def g1(x):
    return 1e8 * (x**2 - 1e-8 * x + 2e-16)  # Just for demo; mimics learned drift behavior

# === Unscented Transform ===
def unscented_transform(F, mean, cov, alpha=1e-3, beta=2, kappa=0):
    n = 1
    lam = alpha**2 * (n + kappa) - n
    gamma = np.sqrt(n + lam)

    sigma_pts = [mean,
                 mean + gamma * np.sqrt(cov),
                 mean - gamma * np.sqrt(cov)]

    Wm = [lam / (n + lam)]
    Wc = [lam / (n + lam) + (1 - alpha**2 + beta)]
    for _ in range(2):
        Wm.append(1 / (2 * (n + lam)))
        Wc.append(1 / (2 * (n + lam)))

    y = [F(pt) for pt in sigma_pts]

    mean_y = np.sum([Wm[i] * y[i] for i in range(3)])
    var_y = np.sum([Wc[i] * (y[i] - mean_y)**2 for i in range(3)])

    return mean_y, var_y

# === Apply UT using g₁(x) ===
ut_mean, ut_var = unscented_transform(g1, mean_x, std_x**2)

# === Plotting: Histogram + UT ===
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# === Histogram of Original Data ===
axs[0].hist(x, bins=100, color='lightblue', edgecolor='gray', alpha=0.7, density=True)
axs[0].axvline(mean_x, color='red', linestyle='--', label=f"Mean = {mean_x:.2e}")
normal_curve = norm.pdf(np.linspace(min(x), max(x), 100), mean_x, std_x)
axs[0].plot(np.linspace(min(x), max(x), 100), normal_curve, 'k--', label="Normal Dist.")
axs[0].legend()
axs[0].set_title("Original Histogram with Moments")
axs[0].set_xlabel("Position (x)")
axs[0].set_ylabel("Density")
axs[0].grid(True)
axs[0].text(0.75, 0.6, '\n'.join((
    f'Mean = {mean_x:.2e}',
    f'Variance = {std_x**2:.2e}',
    f'Skewness = {skew_x:.4f}',
    f'Kurtosis = {kurt_x:.4f}'
)), transform=axs[0].transAxes, bbox=dict(facecolor='white', alpha=0.6), fontsize=9)

# === UT-Based Gaussian Estimate for g₁(x) ===
bins_ut = np.linspace(ut_mean - 3*np.sqrt(ut_var), ut_mean + 3*np.sqrt(ut_var), 200)
ut_pdf = norm.pdf(bins_ut, ut_mean, np.sqrt(ut_var))
axs[1].plot(bins_ut, ut_pdf, color='green', linestyle='-', label="UT Predicted Gaussian")
axs[1].axvline(ut_mean, color='red', linestyle='--', label=f"UT Mean = {ut_mean:.2f}")
axs[1].fill_between(bins_ut, ut_pdf, color='lightgreen', alpha=0.5)
axs[1].set_title("Unscented Transform: g₁(x) - Hidden Physics")
axs[1].set_xlabel("Transformed Position (g₁(x))")
axs[1].set_ylabel("Estimated Density")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
