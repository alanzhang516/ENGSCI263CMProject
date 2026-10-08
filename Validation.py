
# ===============================================================
# ENGSCI 263 - CHECK-IN 2
# VALIDATION: BENCHMARK CONTAINER TEMPERATURE MODEL
# ===============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. File paths
# ---------------------------------------------------------------
DATA_FILE = Path("data/benchmark_container.csv")
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# 2. Benchmark parameters (manufacturer datasheet)
# ---------------------------------------------------------------
mcp = 6.0e8          # Thermal mass [J/K]
UA = 20.0e3          # Heat removal coefficient [W/K]
eta = 0.015          # Fraction of throughput lost as heat
T_hall = 20.0        # Hall temperature [degC]

SEC_PER_DAY = 86400.0
J_PER_MWH = 3.6e9

# Convert parameters for time in days and energy in MWh
alpha = UA * SEC_PER_DAY / mcp
beta = eta * J_PER_MWH / mcp

print(f"Cooling coefficient alpha = {alpha:.4f} /day")
print(f"Heating coefficient beta = {beta:.4f} degC/MWh")

# ---------------------------------------------------------------
# 3. Load benchmark data
# ---------------------------------------------------------------
benchmark = pd.read_csv(DATA_FILE)
benchmark.columns = benchmark.columns.str.strip()

print("\nCSV columns:")
print(benchmark.columns.tolist())

days = benchmark["day"].to_numpy(dtype=float)
throughput = benchmark["throughput_[MWh/day]"].to_numpy(dtype=float)
T_observed = benchmark["cell_temp_[degC]"].to_numpy(dtype=float)

# Check input data
if not np.all(np.isfinite(days)):
    raise ValueError("Day column contains missing values.")

if not np.all(np.isfinite(throughput)):
    raise ValueError("Throughput contains missing values.")

if not np.allclose(np.diff(days), 1.0):
    raise ValueError("Expected one throughput measurement per day.")

# ---------------------------------------------------------------
# 4. Initial condition
# ---------------------------------------------------------------
valid = np.isfinite(T_observed)

if not np.any(valid):
    raise ValueError("No temperature observations found.")

first_obs_idx = np.flatnonzero(valid)[0]

# Begin the model from the first measured temperature
T0 = T_observed[first_obs_idx]
start_day = days[first_obs_idx]

print(f"\nInitial temperature: {T0:.3f} °C")
print(f"Initial day: {start_day:.0f}")

# ---------------------------------------------------------------
# 5. RK4 numerical solver
# ---------------------------------------------------------------
def rk4_step(f, t, y, dt):
    """Advance one step using fourth-order Runge-Kutta."""

    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt*k1/2)
    k3 = f(t + dt/2, y + dt*k2/2)
    k4 = f(t + dt, y + dt*k3)

    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

# ---------------------------------------------------------------
# 6. Primary temperature ODE
# ---------------------------------------------------------------
# dT/dt = beta*E(t) - alpha*(T - T_hall)
#
# E(t) is treated as constant throughout each recorded day.

def temperature_ode(t, T, E):
    return beta * E - alpha * (T - T_hall)

# ---------------------------------------------------------------
# 7. Run temperature simulation
# ---------------------------------------------------------------
steps_per_day = 20
dt = 1.0 / steps_per_day

# Work from the first observation to the last recorded day.
n_days = len(days) - 1 - first_obs_idx
n_steps = n_days * steps_per_day

t_sim = start_day + np.arange(n_steps + 1) * dt

T_sim = np.zeros(n_steps + 1)
T_sim[0] = T0

for i in range(n_steps):

    # Identify the daily throughput value
    day_index = first_obs_idx + i // steps_per_day
    E_day = throughput[day_index]

    # Fixed daily forcing during this RK4 step
    def f(t, T):
        return temperature_ode(t, T, E_day)

    T_sim[i+1] = rk4_step(
        f,
        t_sim[i],
        T_sim[i],
        dt
    )

# ---------------------------------------------------------------
# 8. Compare model with benchmark observations
# ---------------------------------------------------------------
valid_comparison = valid & (days >= start_day)

observed_days = days[valid_comparison]
observed_temp = T_observed[valid_comparison]

# Evaluate model at recorded measurement times
predicted_temp = np.interp(
    observed_days,
    t_sim,
    T_sim
)

residuals = predicted_temp - observed_temp

RMSE = np.sqrt(np.mean(residuals**2))
MAE = np.mean(np.abs(residuals))
MAX_ERROR = np.max(np.abs(residuals))

print("\n===================================")
print("VALIDATION RESULTS")
print("===================================")
print(f"Number of observations: {len(observed_temp)}")
print(f"RMSE: {RMSE:.4f} °C")
print(f"MAE: {MAE:.4f} °C")
print(f"Maximum absolute error: {MAX_ERROR:.4f} °C")

# ---------------------------------------------------------------
# 9. Validation plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    t_sim,
    T_sim,
    color="tab:blue",
    linewidth=2,
    label="RK4 model (benchmark parameters)"
)

ax.scatter(
    observed_days,
    observed_temp,
    color="black",
    s=35,
    zorder=3,
    label="Benchmark measurements"
)

ax.set_xlabel("Day")
ax.set_ylabel("Cell temperature [°C]")
ax.set_title("Validation: Temperature Model vs Benchmark Data")

ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

fig.savefig(
    FIGURE_DIR / "validation_temperature.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


# ===============================================================
# ENGSCI 263 - CAPACITY VALIDATION
# ===============================================================

# ---------------------------------------------------------------
# 1. Manufacturer benchmark capacity parameters
# ---------------------------------------------------------------
k_cal = 0.0175 / 365.0
k_cyc = 0.023 / (365.0 * 400.0)

Ea = 50000.0      # Activation energy [J/mol]
R = 8.314         # Gas constant [J/(mol K)]
T_ref = 25.0      # Reference temperature [degC]

# Read observed benchmark capacities
Q_observed = benchmark["capacity_[fraction]"].to_numpy(dtype=float)
valid_Q = np.isfinite(Q_observed)

# Initial capacity from first available measurement
first_Q_idx = np.flatnonzero(valid_Q)[0]
Q0 = Q_observed[first_Q_idx]
Q_start_day = days[first_Q_idx]

# Require an initial capacity measurement at the start of simulation
if not np.isclose(Q_start_day, t_sim[0]):
    raise ValueError(
        "First capacity measurement does not match the temperature "
        "simulation start. Initial-condition handling is needed."
    )

# ---------------------------------------------------------------
# 2. Capacity degradation ODE
# ---------------------------------------------------------------
def capacity_ode(t, Q, T, E):

    arrhenius = np.exp(
        -(Ea / R) *
        (1.0 / (T + 273.15) - 1.0 / (T_ref + 273.15))
    )

    return -(k_cal + k_cyc * E) * arrhenius


# ---------------------------------------------------------------
# 3. Solve capacity ODE using RK4
# ---------------------------------------------------------------
Q_sim = np.zeros_like(t_sim)
Q_sim[0] = Q0

for i in range(len(t_sim) - 1):

    t = t_sim[i]
    dt_step = t_sim[i+1] - t_sim[i]

    day_index = first_obs_idx + i // steps_per_day
    E_day = throughput[day_index]

    # Temperature at RK4 stages
    T1 = T_sim[i]
    T2 = 0.5 * (T_sim[i] + T_sim[i+1])
    T4 = T_sim[i+1]

    k1 = capacity_ode(t, Q_sim[i], T1, E_day)
    k2 = capacity_ode(t + dt_step/2, Q_sim[i] + dt_step*k1/2, T2, E_day)
    k3 = capacity_ode(t + dt_step/2, Q_sim[i] + dt_step*k2/2, T2, E_day)
    k4 = capacity_ode(t + dt_step, Q_sim[i] + dt_step*k3, T4, E_day)

    Q_sim[i+1] = Q_sim[i] + dt_step/6 * (k1 + 2*k2 + 2*k3 + k4)


# ---------------------------------------------------------------
# 4. Compare against observed capacity
# ---------------------------------------------------------------
valid_capacity_comparison = valid_Q & (days >= t_sim[0])

Q_obs_days = days[valid_capacity_comparison]
Q_obs_values = Q_observed[valid_capacity_comparison]

Q_pred_values = np.interp(
    Q_obs_days,
    t_sim,
    Q_sim
)

Q_residuals = Q_pred_values - Q_obs_values

Q_RMSE = np.sqrt(np.mean(Q_residuals**2))
Q_MAE = np.mean(np.abs(Q_residuals))
Q_MAX_ERROR = np.max(np.abs(Q_residuals))

print("\n===================================")
print("CAPACITY VALIDATION RESULTS")
print("===================================")
print(f"Capacity observations: {len(Q_obs_values)}")
print(f"RMSE: {Q_RMSE:.6f}")
print(f"MAE: {Q_MAE:.6f}")
print(f"Maximum absolute error: {Q_MAX_ERROR:.6f}")


# ---------------------------------------------------------------
# 5. Capacity validation graph
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    t_sim,
    Q_sim,
    color="tab:blue",
    linewidth=2,
    label="RK4 model (benchmark parameters)"
)

ax.scatter(
    Q_obs_days,
    Q_obs_values,
    color="black",
    s=35,
    zorder=3,
    label="Benchmark capacity measurements"
)

ax.set_xlabel("Day")
ax.set_ylabel("Capacity [fraction of nameplate]")
ax.set_title("Validation: Capacity Model vs Benchmark Data")

ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

fig.savefig(
    FIGURE_DIR / "validation_capacity.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
