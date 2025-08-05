import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# Load the trajectory data
df = pd.read_excel("water_X_trajectory_500fps_30kframes_0.02.xlsx", header=None)
x = df[0].values  # Position values

# Calculate moments
mean_x = np.mean(x)
std_x = np.std(x)
skew_x = skew(x)
kurt_x = kurtosis(x, fisher=False)  # Fisher=False gives standard definition (not excess kurtosis)

# Plot histogram
plt.figure(figsize=(10, 5))
count, bins, ignored = plt.hist(x, bins =100, color='lightblue', edgecolor='gray', alpha=0.7, density=True) #−0.75e−7 to +1.00e−7   #Bins = boundaries

# Annotate Mean
plt.axvline(mean_x, color='red', linestyle='--', label=f"Mean = {mean_x:.4f}")  # It’s at around 1.4e−08, which is very close to zero.

# Show a normal curve for comparison using mean & std
from scipy.stats import norm
normal_curve = norm.pdf(bins, mean_x, std_x)     #compare your real data shape to an ideal Gaussian distribution. #pdf = probability density func for perfect gaussian
plt.plot(bins, normal_curve, 'k--', label='Normal Distribution (for reference)')

# Title and labels
plt.title("Histogram of Position Values with Moments")
plt.xlabel("Position (x)")
plt.ylabel("Density")
plt.legend()

# Text box for all moment values
textstr = '\n'.join((
    f'Mean = {mean_x:.4e}',
    f'Variance = {std_x**2:.4e}',
    f'Skewness = {skew_x:.4f}',
    f'Kurtosis = {kurt_x:.4f}'
))
plt.gcf().text(0.75, 0.6, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

plt.grid(True)
plt.tight_layout()
plt.show()
