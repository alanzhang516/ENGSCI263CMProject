from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, lsq_linear

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------
eta = 0.015            # resistive heat fraction [-] (datasheet, fixed)
T_hall = 20.0          # [degC]
Ea = 50.0e3            # [J/mol]
R = 8.314              # [J/(mol K)]
T_ref = 25.0           # [degC]

SEC_PER_DAY = 86400.0
J_PER_MWH = 3.6e9
DT = 0.1               # RK4 step [day]
N_SUB = int(round(1.0 / DT))

# Initial guesses (revised starting values)
mcp_0, UA_0 = 6.0e8, 20.0e3
kcal_0, kcyc_0, Q0_0 = 0.0175 / 365.0, 0.023 / (365.0 * 400.0), 1.0

# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
df = pd.read_csv("data/grid_battery_data.csv")
days = df["day"].to_numpy()
E = df["throughput_[MWh/day]"].to_numpy(dtype=float)
T_obs = df["cell_temp_[degC]"].to_numpy()
Q_obs = df["capacity_[fraction]"].to_numpy()
mT = ~np.isnan(T_obs)      # temperature readings (sensor gap excluded)
mQ = ~np.isnan(Q_obs)      # capacity readings
T0 = T_obs[0]


# ---------------------------------------------------------------
# Temperature model
# Throughput is constant within each day, so the ODE has an exact solution
# over each sub-step:  T(t+h) = T_ss + (T(t) - T_ss) exp(-a h),
# with T_ss = T_hall + b E / a.  This is unconditionally stable (unlike RK4
# at large UA/mcp) and is used for calibration. The verified RK4 solver is
# kept as a cross-check (solve_T_rk4).
# ---------------------------------------------------------------
def solve_T(mcp, UA, dt=DT):
    """Returns (t_fine, T_fine, T_day). T_day[d] is the state at the START of
    day d, which is compared with the reading taken on day d."""
    a = UA * SEC_PER_DAY / mcp          # [1/day]
    b = eta * J_PER_MWH / mcp           # [degC per MWh]
    decay = np.exp(-a * dt)
    T = T0
    T_fine, T_day = [T0], [T0]
    for d in days[:-1]:
        T_ss = T_hall + b * E[d] / a
        for _ in range(N_SUB):
            T = T_ss + (T - T_ss) * decay
            T_fine.append(T)
        T_day.append(T)
    t_fine = np.arange(len(T_fine)) * dt
    return t_fine, np.array(T_fine), np.array(T_day)


def solve_T_rk4(mcp, UA, dt=DT):
    """Verified RK4 solver, used only to cross-check the calibrated result."""
    a = UA * SEC_PER_DAY / mcp
    b = eta * J_PER_MWH / mcp
    f = lambda T, e: b * e - a * (T - T_hall)
    T = T0
    T_day = [T0]
    for d in days[:-1]:
        e = E[d]
        for _ in range(N_SUB):
            k1 = f(T, e)
            k2 = f(T + dt / 2 * k1, e)
            k3 = f(T + dt / 2 * k2, e)
            k4 = f(T + dt * k3, e)
            T = T + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        T_day.append(T)
    return np.array(T_day)


def temp_residuals(p):
    """p = [log10(mcp), log10(UA)] -- log scaling handles the very
    different magnitudes of the two parameters."""
    _, _, T_day = solve_T(10 ** p[0], 10 ** p[1])
    return T_day[mT] - T_obs[mT]


def rmse(r):
    return np.sqrt(np.mean(r ** 2))


# ---------------------------------------------------------------
# Stage 1: temperature calibration (multi-start)
# ---------------------------------------------------------------
bounds_T = ([8.0, 3.0], [11.0, 6.0])     # mcp 1e8-1e11 J/K, UA 1e3-1e6 W/K
starts = [(6e8, 2e4), (1e9, 1e4), (2e9, 2e4), (4e9, 1.5e4), (6e9, 3e4),
          (1e10, 5e4), (3e8, 5e3), (2e10, 1e5)]

results = []
print("Multi-start temperature calibration")
for m0, u0 in starts:
    sol = least_squares(temp_residuals, [np.log10(m0), np.log10(u0)],
                        bounds=bounds_T, x_scale="jac")
    mcp_s, UA_s = 10 ** sol.x
    at_bound = np.any(np.isclose(sol.x, bounds_T[0]) | np.isclose(sol.x, bounds_T[1]))
    ok = sol.success and np.all(np.isfinite(sol.fun)) and not at_bound
    if ok:
        results.append((rmse(sol.fun), mcp_s, UA_s))
    print(f"  start mcp={m0:.1e}, UA={u0:.1e}  ->  mcp={mcp_s:.3e}, "
          f"UA={UA_s:.3e}, RMSE={rmse(sol.fun):.3f} degC, "
          f"success={sol.success}, at_bound={at_bound}")

rmse_T, mcp_c, UA_c = min(results)
rmse_T0 = rmse(temp_residuals([np.log10(mcp_0), np.log10(UA_0)]))
tau_c = mcp_c / UA_c / SEC_PER_DAY
print(f"\nCalibrated: mcp = {mcp_c:.3e} J/K, UA = {UA_c:.3e} W/K, "
      f"tau = {tau_c:.2f} days, RMSE = {rmse_T:.3f} degC "
      f"(initial guess RMSE = {rmse_T0:.3f} degC)")

# Comparison: mcp fixed at datasheet value, UA only
sol_fix = least_squares(lambda p: temp_residuals([np.log10(mcp_0), p[0]]),
                        [np.log10(UA_0)], bounds=([3.0], [6.0]))
print(f"Max |exact - RK4| at calibrated values: "
      f"{np.max(np.abs(solve_T(mcp_c, UA_c)[2] - solve_T_rk4(mcp_c, UA_c))):.2e} degC")

print(f"mcp fixed at datasheet: UA = {10**sol_fix.x[0]:.3e} W/K, "
      f"RMSE = {rmse(sol_fix.fun):.3f} degC")

# ---------------------------------------------------------------
# Stage 2: capacity calibration (linear least squares)
# With T(t) fixed, Q(t) = Q0 - k_cal*I_cal(t) - k_cyc*I_cyc(t), which is
# linear in (Q0, k_cal, k_cyc), so lsq_linear finds the global minimum.
# ---------------------------------------------------------------
def arrhenius(T):
    return np.exp(-Ea / R * (1.0 / (T + 273.15) - 1.0 / (T_ref + 273.15)))


def fade_integrals(T_fine):
    """Integrate f(T) and E*f(T) over each day on the fine RK4 grid.
    Returns cumulative integrals at the start of each day."""
    fT = arrhenius(T_fine)
    I_cal, I_cyc = [0.0], [0.0]
    for i, d in enumerate(days[:-1]):
        seg = fT[i * N_SUB:(i + 1) * N_SUB + 1]
        day_int = DT * (seg.sum() - 0.5 * (seg[0] + seg[-1]))   # trapezoid rule
        I_cal.append(I_cal[-1] + day_int)
        I_cyc.append(I_cyc[-1] + E[d] * day_int)
    return np.array(I_cal), np.array(I_cyc)


def capacity(Q0, kcal, kcyc, I_cal, I_cyc):
    return Q0 - kcal * I_cal - kcyc * I_cyc


t_fine, T_fine_c, T_day_c = solve_T(mcp_c, UA_c)
I_cal, I_cyc = fade_integrals(T_fine_c)

# Scale columns so the three unknowns are of similar size
s = np.array([1.0, 1e-5, 1e-7])
A = np.column_stack([np.ones(mQ.sum()), -I_cal[mQ], -I_cyc[mQ]]) * s
sol_Q = lsq_linear(A, Q_obs[mQ], bounds=([0.9, 0.0, 0.0], [1.1, np.inf, np.inf]))
Q0_c, kcal_c, kcyc_c = sol_Q.x * s
Q_day_c = capacity(Q0_c, kcal_c, kcyc_c, I_cal, I_cyc)
rmse_Q = rmse(Q_day_c[mQ] - Q_obs[mQ])

# Initial-guess models for comparison
_, T_fine_0, T_day_0 = solve_T(mcp_0, UA_0)
I_cal0, I_cyc0 = fade_integrals(T_fine_0)
Q_day_0 = capacity(Q0_0, kcal_0, kcyc_0, I_cal0, I_cyc0)
rmse_Q0 = rmse(Q_day_0[mQ] - Q_obs[mQ])

print(f"\nCalibrated: Q0 = {Q0_c:.4f}, k_cal = {kcal_c:.3e} /day "
      f"({kcal_c*365:.4f} /yr), k_cyc = {kcyc_c:.3e} /MWh "
      f"({kcyc_c*400:.3e} /cycle), RMSE = {rmse_Q:.5f} "
      f"(initial guess RMSE = {rmse_Q0:.5f})")

# Continuous capacity curves on the fine grid, using the same trapezoid
# rule as the fitted integrals so plotted and fitted values agree
def capacity_fine(Q0, kcal, kcyc, T_fine):
    fT = arrhenius(T_fine)
    E_fine = np.repeat(E[:-1], N_SUB)
    dQ = (kcal + kcyc * E_fine) * 0.5 * (fT[:-1] + fT[1:]) * DT
    return Q0 - np.concatenate([[0.0], np.cumsum(dQ)])


Q_fine_c = capacity_fine(Q0_c, kcal_c, kcyc_c, T_fine_c)
Q_fine_0 = capacity_fine(Q0_0, kcal_0, kcyc_0, T_fine_0)
assert np.allclose(Q_fine_c[::N_SUB], Q_day_c)

print("\nParameter   Initial guess   Calibrated   Change")
for name, p0, pc in [("mcp [J/K]", mcp_0, mcp_c), ("UA [W/K]", UA_0, UA_c),
                     ("k_cal [/day]", kcal_0, kcal_c), ("k_cyc [/MWh]", kcyc_0, kcyc_c),
                     ("Q0 [-]", Q0_0, Q0_c)]:
    print(f"{name:13s} {p0:11.3e}   {pc:11.3e}   {100*(pc/p0-1):+6.0f}%")

# ---------------------------------------------------------------
# Figure 1: calibrated temperature and capacity
# ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

ax1.plot(t_fine, T_fine_0, "-", color="0.6", lw=0.8, label="Initial guess")
ax1.plot(t_fine, T_fine_c, "-", color="tab:blue", lw=1.5, label="Calibrated model")
ax1.plot(days[mT], T_obs[mT], "ko", ms=4, label="Data")
ax1.set_ylabel("Cell temperature [°C]")
ax1.set_title(f"Calibrated temperature model (RMSE = {rmse_T:.2f} °C)")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.plot(t_fine, Q_fine_0, "-", color="0.6", lw=0.8, label="Initial guess")
ax2.plot(t_fine, Q_fine_c, "-", color="tab:blue", lw=1.5, label="Calibrated model")
ax2.plot(days[mQ], Q_obs[mQ], "ko", ms=4, label="Data")
ax2.set_xlabel("Day (from 1 July 2024)")
ax2.set_ylabel("Capacity [fraction of nameplate]")
ax2.set_title(f"Calibrated capacity model (RMSE = {rmse_Q:.4f})")
ax2.legend(loc="lower left", fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, days[-1])

plt.tight_layout()
fig.savefig(FIGURE_DIR / "calibration.png", dpi=300, bbox_inches="tight",
            facecolor="white")

# ---------------------------------------------------------------
# Figure 2: temperature objective surface (uniqueness + identifiability)
# ---------------------------------------------------------------
lm = np.linspace(8.3, 10.5, 45)       # log10 mcp
lu = np.linspace(3.5, 5.5, 41)        # log10 UA
with np.errstate(all="ignore"):
    Z = np.array([[rmse(temp_residuals([x, y])) for x in lm] for y in lu])

fig2, ax = plt.subplots(figsize=(6, 4.5))
cs = ax.contourf(lm, lu, np.log10(Z), levels=25, cmap="viridis")
cb = fig2.colorbar(cs, ax=ax)
cb.set_label("log$_{10}$(RMSE [°C])")
ax.plot([np.log10(m) for m, u in starts], [np.log10(u) for m, u in starts],
        "wx", ms=7, mew=1.5, label="Starting points")
ax.plot(np.log10(mcp_c), np.log10(UA_c), "r*", ms=14, label="Calibrated")
ax.plot(np.log10(mcp_0), np.log10(UA_0), "wo", mfc="none", ms=8, label="Initial guess")
ax.set_xlabel(r"log$_{10}$ $mc_p$ [J/K]")
ax.set_ylabel(r"log$_{10}$ $UA$ [W/K]")
ax.set_title("Temperature misfit surface ($\\eta$ = 0.015 fixed)")
ax.legend(loc="lower right", fontsize=8, facecolor="0.75", framealpha=0.9)
plt.tight_layout()
fig2.savefig(FIGURE_DIR / "misfit_surface.png", dpi=300, bbox_inches="tight",
             facecolor="white")
plt.show()


# ---------------------------------------------------------------
# Parameter table: initial guesses vs calibrated values
# ---------------------------------------------------------------
def sci(x, sig=3):
    """Format a number as a x 10^n using mathtext."""
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    if -2 <= exp <= 4:
        return f"{x:,.{max(sig - 1 - exp, 0)}f}"
    mant = x / 10 ** exp
    return rf"${mant:.{sig - 1}f} \times 10^{{{exp}}}$"


rows = [
    # name, units, initial, calibrated (None = fixed), basis
    (r"Thermal mass, $mc_p$", "J/K", mcp_0, mcp_c, "Calibrated"),
    (r"Heat removal, $UA$", "W/K", UA_0, UA_c, "Calibrated"),
    (r"Calendar fade, $k_{cal}$", "1/day", kcal_0, kcal_c, "Calibrated"),
    (r"Cycle fade, $k_{cyc}$", "1/MWh", kcyc_0, kcyc_c, "Calibrated"),
    (r"Initial capacity, $Q_0$", "–", Q0_0, Q0_c, "Calibrated"),
    (r"Heat loss fraction, $\eta$", "–", eta, None, "Fixed (datasheet)"),
    (r"Activation energy, $E_a$", "J/mol", Ea, None, "Fixed (datasheet)"),
    (r"Hall temperature, $T_{hall}$", "°C", T_hall, None, "Fixed (given)"),
    (r"Reference temperature, $T_{ref}$", "°C", T_ref, None, "Fixed (definition)"),
    (r"Initial temperature, $T_0$", "°C", T0, None, "Fixed (first reading)"),
]

cell_text = []
for name, unit, p0, pc, basis in rows:
    if pc is None:
        cell_text.append([name, unit, sci(p0), "–", "–", basis])
    else:
        ratio = pc / p0
        if ratio > 3:
            change = f"×{ratio:.1f}"
        elif abs(ratio - 1) < 0.01:
            change = f"{100*(ratio-1):+.2f}%"
        else:
            change = f"{100*(ratio-1):+.0f}%"
        fmt = (lambda v: f"{v:.4f}") if "Q_0" in name else sci
        cell_text.append([name, unit, fmt(p0), fmt(pc), change.replace("-", "−"), basis])

headers = ["Parameter", "Units", "Initial guess", "Calibrated", "Change", "Status"]

fig3, ax = plt.subplots(figsize=(10, 0.36 * (len(cell_text) + 1) + 0.2))
ax.axis("off")
tbl = ax.table(cellText=cell_text, colLabels=headers, loc="center",
               cellLoc="center", colLoc="center",
               colWidths=[0.29, 0.08, 0.15, 0.15, 0.09, 0.20])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.45)
n_cal = sum(r[3] is not None for r in rows)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("0.75")
    if c == 0:
        cell.set_text_props(ha="left")
        cell.PAD = 0.03
    if r == 0:
        cell.set_facecolor("0.85")
        cell.set_text_props(weight="bold")
    elif r <= n_cal:
        cell.set_facecolor("#eaf2fb")          # calibrated / derived
plt.tight_layout()
fig3.savefig(FIGURE_DIR / "parameter_table.png", dpi=300, bbox_inches="tight",
             facecolor="white")

# Same table as CSV (plain numbers) for copying into the report
pd.DataFrame(
    [[r[0].replace("$", "").replace("\\", ""), r[1], r[2],
      r[3] if r[3] is not None else "", r[4]] for r in rows],
    columns=["parameter", "units", "initial_guess", "calibrated", "status"],
).to_csv(FIGURE_DIR / "parameter_table.csv", index=False)
plt.show()