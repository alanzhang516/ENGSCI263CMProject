from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# Model parameters (benchmark datasheet values)
# ---------------------------------------------------------------
mcp = 6.0e8          # thermal mass [J/K]
UA = 20.0e3          # heat removal [W/K]
eta = 0.015          # fraction of throughput lost as heat [-]
T_hall = 20.0        # hall temperature [degC]
T0 = 26.5            # initial temperature [degC]

SEC_PER_DAY = 86400.0
J_PER_MWH = 3.6e9

alpha = UA * SEC_PER_DAY / mcp     # [1/day]      ~2.88
beta = eta * J_PER_MWH / mcp       # [degC/MWh]   ~0.09


# ---------------------------------------------------------------
# Solver
# ---------------------------------------------------------------
def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve(f, y0, t_end, n_steps):
    """Integrate from 0 to t_end in exactly n_steps RK4 steps."""
    t = np.linspace(0.0, t_end, n_steps + 1)
    dt = t[1] - t[0]
    y = np.zeros_like(t)
    y[0] = y0
    for i in range(n_steps):
        y[i + 1] = rk4_step(f, t[i], y[i], dt)
    return t, y


# ---------------------------------------------------------------
# 1. Zero-forcing benchmark
# ---------------------------------------------------------------
def f_zero(t, T):
    return -alpha * (T - T_hall)


def T_zero_exact(t):
    return T_hall + (T0 - T_hall) * np.exp(-alpha * t)


t_zero, T_num_zero = solve(f_zero, T0, t_end=5.0, n_steps=100)   # dt = 0.05 day

fig1, ax1 = plt.subplots(figsize=(6, 4.5))
ax1.plot(t_zero, T_zero_exact(t_zero), 'k-', linewidth=2, label='Exact analytical')
ax1.plot(t_zero, T_num_zero, 'r--', linewidth=1.5, label='Numerical simulator (RK4)')
ax1.set_xlabel('Day')
ax1.set_ylabel('Cell temperature [°C]')
ax1.set_title('Zero-forcing benchmark: temperature decay')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
fig1.savefig(FIGURE_DIR / "zero_forcing_verif.png", dpi=300,
             bbox_inches="tight", facecolor="white")
plt.show()


# ---------------------------------------------------------------
# 2. Method of manufactured solutions (MMS) benchmark
# ---------------------------------------------------------------
A_mms, omega_mms, slope_mms = 5.0, 2 * np.pi / 10.0, 0.1


def T_mms(t):
    return T_hall + A_mms * np.sin(omega_mms * t) + slope_mms * t


def dT_mms_dt(t):
    return A_mms * omega_mms * np.cos(omega_mms * t) + slope_mms


def E_func(t):
    """Synthetic throughput forcing [MWh/day]."""
    return 400.0 + 200.0 * np.cos(2 * np.pi * t / 7.0)


def q_mms(t):
    """Source term that makes T_mms an exact solution."""
    return dT_mms_dt(t) - (beta * E_func(t) - alpha * (T_mms(t) - T_hall))


def f_mms(t, T):
    return beta * E_func(t) - alpha * (T - T_hall) + q_mms(t)


T_END_MMS = 20.0
t_mms, T_num_mms = solve(f_mms, T_mms(0.0), T_END_MMS, n_steps=200)   # dt = 0.1 day

# Convergence study: step counts chosen so every run ends exactly at t = 20
n_list = np.array([40, 80, 160, 320, 640, 1280])
dt_list = T_END_MMS / n_list
rms_errors = []
for n in n_list:
    t, T_num = solve(f_mms, T_mms(0.0), T_END_MMS, n)
    rms_errors.append(np.sqrt(np.mean((T_num - T_mms(t)) ** 2)))
rms_errors = np.array(rms_errors)

# Observed order of accuracy (slope of log-log fit)
order = np.polyfit(np.log(dt_list), np.log(rms_errors), 1)[0]

fig2, ax3 = plt.subplots(figsize=(6, 4.5))
ax3.plot(t_mms, T_mms(t_mms), 'k-', linewidth=2, label='Manufactured exact $T_{mms}(t)$')
ax3.plot(t_mms, T_num_mms, 'r--', linewidth=1.5, label='Simulator output $T_{num}(t)$')
ax3.set_xlabel('Day')
ax3.set_ylabel('Cell temperature [°C]')
ax3.set_title('MMS benchmark: forced trajectory comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig(FIGURE_DIR / "mms_verif.png", dpi=300,
             bbox_inches="tight", facecolor="white")
plt.show()

print(f"alpha = {alpha:.3f} /day, beta = {beta:.4f} degC/MWh, observed order = {order:.2f}")