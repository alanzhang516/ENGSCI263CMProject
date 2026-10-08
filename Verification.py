from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIGURE_DIR = Path("figures")

# Model parameters
alpha = 0.20       # UA / (m * c_p) [1/day]
beta = 0.005       # eta / (m * c_p) [degC / MWh]
T_hall = 20.0      # Hall temperature [degC]
T0 = 26.5          # Initial temperature [degC]

# 1. Zero-Forcing Time Series Simulation (RK4 solver vs Exact)
t_eval = np.linspace(0, 30, 301) # 30 days, dt = 0.1 day
dt = t_eval[1] - t_eval[0]

def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Zero-forcing ODE function
def f_zero(t, T):
    return -alpha * (T - T_hall)

# Exact solution
T_exact_zero = T_hall + (T0 - T_hall) * np.exp(-alpha * t_eval)

# Numerical solution
T_num_zero = np.zeros_like(t_eval)
T_num_zero[0] = T0
for i in range(len(t_eval) - 1):
    T_num_zero[i+1] = rk4_step(f_zero, t_eval[i], T_num_zero[i], dt)

# Convergence testing across different dt values
dt_list = np.logspace(-3, 0, 10)
errors_zero = []
for h in dt_list:
    t_steps = np.arange(0, 30 + h, h)
    T_sim = T0
    for t in t_steps[:-1]:
        T_sim = rk4_step(f_zero, t, T_sim, h)
    exact_end = T_hall + (T0 - T_hall) * np.exp(-alpha * 30.0)
    errors_zero.append(abs(T_sim - exact_end))

# Plotting Zero-Forcing Verification
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Plot A: Time History
ax1.plot(t_eval, T_exact_zero, 'k-', linewidth=2, label='Exact Analytical')
ax1.plot(t_eval, T_num_zero, 'r--', linewidth=1.5, label='Numerical Simulator (RK4)')
ax1.set_xlabel('Day')
ax1.set_ylabel('Cell Temperature [°C]')
ax1.set_title('Zero-Forcing Benchmark: Temperature Decay')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot B: Grid Convergence
ax2.loglog(dt_list, errors_zero, 'o-', color='tab:blue', label='Simulator Global Error')
ax2.loglog(dt_list, 1e-4 * (dt_list**4), 'k:', label='Theoretical $O(\\Delta t^4)$ Slope')
ax2.set_xlabel('Time Step $\\Delta t$ [days]')
ax2.set_ylabel('Absolute Error at $t=30$ days [°C]')
ax2.set_title('Zero-Forcing Benchmark: Grid Convergence')
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
fig1.savefig(
    FIGURE_DIR / "zero_forcing_verif.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)
plt.show()

# Manufactured solution parameters
A_mms, omega_mms, slope_mms = 5.0, 2 * np.pi / 10.0, 0.1

def T_mms(t):
    return T_hall + A_mms * np.sin(omega_mms * t) + slope_mms * t

def dT_mms_dt(t):
    return A_mms * omega_mms * np.cos(omega_mms * t) + slope_mms

# Synthetic forcing function E(t)
def E_func(t):
    return 400.0 + 200.0 * np.cos(2 * np.pi * t / 7.0)

# MMS Source term
def q_mms(t):
    return dT_mms_dt(t) - (beta * E_func(t) - alpha * (T_mms(t) - T_hall))

# Modified ODE executed by numerical simulator
def f_mms(t, T):
    return beta * E_func(t) - alpha * (T - T_hall) + q_mms(t)

# Run MMS Simulation
t_mms_eval = np.linspace(0, 20, 201)
dt_mms = t_mms_eval[1] - t_mms_eval[0]
T_num_mms = np.zeros_like(t_mms_eval)
T_num_mms[0] = T_mms(0)

for i in range(len(t_mms_eval) - 1):
    T_num_mms[i+1] = rk4_step(f_mms, t_mms_eval[i], T_num_mms[i], dt_mms)

# Grid Convergence Study for MMS
dt_mms_list = np.logspace(-3, 0, 10)
rms_errors_mms = []

for h in dt_mms_list:
    t_steps = np.arange(0, 20 + h, h)
    T_sim = T_mms(0)
    sim_results = [T_sim]
    for t in t_steps[:-1]:
        T_sim = rk4_step(f_mms, t, T_sim, h)
        sim_results.append(T_sim)
    exact_results = T_mms(t_steps)
    rms_err = np.sqrt(np.mean((np.array(sim_results) - exact_results)**2))
    rms_errors_mms.append(rms_err)

# Plotting MMS Verification
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4.5))

# Plot A: MMS Time History Overlay
ax3.plot(t_mms_eval, T_mms(t_mms_eval), 'k-', linewidth=2, label='Manufactured Exact $T_{mms}(t)$')
ax3.plot(t_mms_eval, T_num_mms, 'r--', linewidth=1.5, label='Simulator Output $T_{num}(t)$')
ax3.set_xlabel('Day')
ax3.set_ylabel('Cell Temperature [°C]')
ax3.set_title('MMS Benchmark: Forced Trajectory Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot B: MMS Convergence Log-Log
ax4.loglog(dt_mms_list, rms_errors_mms, 's-', color='tab:green', label='Simulator RMS Error')
ax4.loglog(dt_mms_list, 1e-3 * (dt_mms_list**4), 'k:', label='Theoretical $O(\\Delta t^4)$ Slope')
ax4.set_xlabel('Time Step $\\Delta t$ [days]')
ax4.set_ylabel('RMS Temperature Error [°C]')
ax4.set_title('MMS Benchmark: Order-of-Accuracy Verification')
ax4.legend()
ax4.grid(True, which="both", alpha=0.3)

plt.tight_layout()
fig2.savefig(
    FIGURE_DIR / "mms_verif.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)
plt.show()