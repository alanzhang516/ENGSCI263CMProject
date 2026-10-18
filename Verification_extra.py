# Capacity ODE verification: numerical solution vs exact solution at constant temperature.
# If the cells start at their steady-state temperature under constant throughput, T stays
# constant, so the capacity ODE has the exact solution
#     Q(t) = Q0 - (k_cal + k_cyc*E) * F(T_ss) * t      (a straight line)

import numpy as np
import matplotlib.pyplot as plt

# Parameters (calibrated values, Check-in 2)
ETA, T_AIR, EA, R, T_REF = 0.015, 20.0, 50e3, 8.314, 298.15
C_TH, H = 5.9055e9, 25027.0
K_CAL, K_CYC, Q0 = 2.6735e-5, 2.3802e-7, 0.99961
W_PER_MWH_DAY = 3.6e9 / 86400       # 1 MWh/day in watts


def F(T):
    """Arrhenius factor relative to 25 degC (T in degC)."""
    return np.exp(-(EA / R) * (1.0 / (T + 273.15) - 1.0 / T_REF))


def solve(E, T0, n_sub=10):
    """Exact-daily solver: exact temperature within each day, capacity by the
    trapezoid rule on n_sub sub-steps per day. Returns T, Q at the start of each day."""
    a = H * 86400.0 / C_TH
    s = np.linspace(0, 1, n_sub + 1)
    T, Q = T0, Q0
    Ts, Qs = [T], [Q]
    for e in E:
        T_ss = T_AIR + ETA * e * W_PER_MWH_DAY / H
        T_fine = T_ss + (T - T_ss) * np.exp(-a * s)
        g = (K_CAL + K_CYC * e) * F(T_fine)
        Q -= np.sum((g[1:] + g[:-1]) / 2) / n_sub      # trapezoid rule over one day
        T = T_fine[-1]
        Ts.append(T)
        Qs.append(Q)
    return np.array(Ts), np.array(Qs)


# Test: one year at a constant 400 MWh/day, starting at steady state
E_CONST, DAYS = 400.0, 365
T_ss = T_AIR + ETA * E_CONST * W_PER_MWH_DAY / H
t = np.arange(DAYS + 1)

_, Q_num = solve(np.full(DAYS, E_CONST), T0=T_ss)
Q_exact = Q0 - (K_CAL + K_CYC * E_CONST) * F(T_ss) * t

print(f'Steady-state temperature: {T_ss:.2f} degC')
print(f'Max |numerical - exact|:  {np.abs(Q_num - Q_exact).max():.1e}')

# Sub-step check on the real record: does using 10 sub-steps per day matter?
import pandas as pd
E_data = pd.read_csv('data/grid_battery_data.csv')['throughput_[MWh/day]'].to_numpy(float)[:-1]
_, Q10 = solve(E_data, T0=26.5, n_sub=10)
_, Q1000 = solve(E_data, T0=26.5, n_sub=1000)
print(f'Real record, 10 vs 1000 sub-steps per day: max |difference| = {np.abs(Q10 - Q1000).max():.1e}')

# Plot
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(t, Q_exact, 'k-', lw=2, label='Analytic (straight line)')
ax.plot(t[::15], Q_num[::15], 'o', color='tab:red', ms=4, mfc='none', label='Numerical')
ax.set(xlabel='Day', ylabel='Capacity [fraction of nameplate]',
       title=f'Capacity ODE at constant T = {T_ss:.1f} °C')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('verify_capacity.png', dpi=200)
plt.show()