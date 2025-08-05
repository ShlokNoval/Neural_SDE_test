import pandas as pd
import matplotlib.pyplot as plt

# Load trajectory data (position x and time t)
df = pd.read_excel("water_X_trajectory_500fps_30kframes_0.02.xlsx", header=None)
x = df[0].values  # Position
t = df[1].values  # Time

# Compute velocity: v = dx/dt
v = (x[1:] - x[:-1]) / (t[1:] - t[:-1]) #X[1:] = starting from first to last (elements).
t_v = (t[1:] + t[:-1]) / 2  #Calculates average time between consecutive measurements because velocity represent the rate of change between two points.

# Compute acceleration: a = dv/dt
a = (v[1:] - v[:-1]) / (t_v[1:] - t_v[:-1])
t_a = (t_v[1:] + t_v[:-1]) / 2

# Plot Position (Trajectory)
plt.figure(figsize=(10, 3))
plt.plot(t, x, label="Position x(t)")
plt.title("Position vs Time (Trajectory)") #Position vs Time
plt.xlabel("Time")
plt.ylabel("Position")
plt.grid(True)
plt.legend()

# Plot Velocity
plt.figure(figsize=(10, 3))
plt.plot(t_v, v, label="Velocity v(t)", color="green")
plt.title("Velocity vs Time (Movement)") #Velocity vs Time
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.grid(True)
plt.legend()

# Plot Acceleration
plt.figure(figsize=(10, 3))
plt.plot(t_a, a, label="Acceleration a(t)", color="red")
plt.title("Acceleration vs Time (Change in Movement)") #Acceleration vs Time
plt.xlabel("Time")
plt.ylabel("Acceleration")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
