# # ENGSCI 263 Grid Battery: Uncertainty Analysis and Scenario Forecasts
#
# Based on the team's uncertainty-lab script, rewritten so one file does everything:
#   1  Joint calibration of all five parameters -> best fit + covariance (Gaussian posterior)
#   2  Parameter uncertainty in physical units, correlation matrix, residuals
#   3  Is the Gaussian good enough? (profile misfit contours + importance-sampling check)
#   4  Infeasible samples (negative fade rates) rejected, as in the course notes
#   5  Ten-year forecasts for the three options, with 90 % prediction intervals
#   6  Decision metrics: years to end of life, capacity after 10 years, temperatures
#   7  Sensitivity to fixed assumptions: activation energy and chiller effectiveness
#
# Everything is saved in a figures/ folder next to this script:
#   fig1-fig4  uncertainty section (Figures 1-4 of the report section)
#   figA       Gaussian vs true posterior samples (appendix)
#   fc1-fc4    forecasts and sensitivity
#   forecast_summary.csv  every forecast number
# Run top to bottom (VS Code / Jupyter treat each "# %%" as a cell).

# %% Imports and settings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares
from scipy.stats import multivariate_normal

SEED = 263
rng = np.random.default_rng(SEED)
# All paths are relative to this script's folder, so figures always land in <script folder>/figures
# no matter where the script is run from. (In a notebook, __file__ doesn't exist: use the working folder.)
try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path.cwd()
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

_candidates = [BASE / "data" / "grid_battery_data.csv", BASE / "grid_battery_data.csv"]
DATA_FILE = next((p for p in _candidates if p.exists()), None)
if DATA_FILE is None:
    raise FileNotFoundError("Put grid_battery_data.csv in this folder or in a data/ subfolder")

# Fixed physical parameters
ETA, T_HALL, EA, R, T_REF = 0.015, 20.0, 50.0e3, 8.314, 25.0
SEC_PER_DAY, J_PER_MWH = 86400.0, 3.6e9
Q_CYCLE = 400.0                      # MWh of throughput per full cycle

# Numerical settings (verified: 10 sub-steps changes capacity by < 1e-7)
N_SUB = 10

# Forecast settings
FORECAST_YEARS = 10
N_FUT = 365 * FORECAST_YEARS
EOLS = [0.80, 0.70]                  # end-of-life thresholds to report (cite your choice!)
MAINT_BLOCKS = [(90, 7), (270, 7)]   # (start day of each forecast year, length): zero throughput
RESAMPLE_THROUGHPUT = True           # True: daily throughput resampled from the record
CHILL = 2.0                          # chiller multiplies H by this
N_FORECAST = 1000                    # posterior samples used for the prediction intervals
N_SHOW = 50                          # samples drawn on the spaghetti plots
PI = (5, 95)                         # 90 % prediction interval

# Sensitivity settings (replace with the ranges you cite)
EA_VALUES = [40e3, 50e3, 60e3]
CHILL_VALUES = [1.5, 2.0, 2.5]
N_SENS = 300

# Profile-misfit landscape (slowest part, ~1 min). Set False while iterating.
RUN_PROFILE = True
N_GRID = 20

# %% Data
df = pd.read_csv(DATA_FILE)
days = df["day"].to_numpy()
E = df["throughput_[MWh/day]"].to_numpy(float)
T_obs = df["cell_temp_[degC]"].to_numpy()
Q_obs = df["capacity_[fraction]"].to_numpy()
mT, mQ = ~np.isnan(T_obs), ~np.isnan(Q_obs)       # the sensor gap is simply left out
nT, nQ = mT.sum(), mQ.sum()
T0 = T_obs[0]
E_FIT = E[:-1]                                     # throughput on days 0..181 drives days 1..182

POOL_WARRANTY = E[(days <= 90) & (E > 0)]          # one cycle/day, maintenance excluded
POOL_AGGRESSIVE = E[(days >= 92) & (E > 0)]        # two cycles/day, maintenance excluded

# %% Parameters and solver
# theta = [log10 C_th, log10 H, Q0, k_cal*1e5, k_cyc*1e7]  (scaled so entries are O(1))
LABELS = [r"$\log_{10} C_{th}$", r"$\log_{10} H$", r"$Q_0$",
          r"$k_{cal}$ [$10^{-5}$/day]", r"$k_{cyc}$ [$10^{-7}$/MWh]"]
SHORT = ["C_th", "H", "Q0", "k_cal", "k_cyc"]
UNITS = ["J/K", "W/K", "-", "1/day", "1/MWh"]
LOWER = np.array([8.0, 3.0, 0.9, 0.0, 0.0])
UPPER = np.array([11.0, 6.0, 1.1, np.inf, np.inf])
X_SCALE = np.array([0.05, 0.01, 1e-3, 1.0, 1.0])
TH0 = np.array([np.log10(6e8), np.log10(2e4), 1.0, 4.79, 1.575])   # datasheet starting point

S_GRID = np.linspace(0.0, 1.0, N_SUB + 1)
W_TRAP = np.full(N_SUB + 1, 1.0 / N_SUB)
W_TRAP[[0, -1]] = 0.5 / N_SUB                      # trapezoid weights over one day


def to_physical(th):
    th = np.atleast_2d(th)
    return 10 ** th[:, 0], 10 ** th[:, 1], th[:, 2], th[:, 3] * 1e-5, th[:, 4] * 1e-7


def arrhenius(T, Ea=EA):
    return np.exp(-Ea / R * (1.0 / (T + 273.15) - 1.0 / (T_REF + 273.15)))


def simulate(th, E_days, T_start=None, Q_start=None, Ea=EA, H_mult=1.0):
    """Exact-daily solver, vectorised over parameter sets.

    th      : (5,) or (N, 5) parameter vector(s)
    E_days  : (D,) or (N, D) throughput for each day [MWh/day]
    Returns T, Q of shape (N, D+1) at the start of each day.
    Temperature is solved exactly within each day (throughput is constant within a day);
    capacity integrates F(T) over each day with the trapezoid rule on N_SUB sub-steps.
    """
    Cth, H, Q0, kcal, kcyc = to_physical(th)
    H = H * H_mult
    N = len(Cth)
    E_days = np.broadcast_to(np.atleast_2d(E_days), (N, np.shape(E_days)[-1]))
    D = E_days.shape[1]
    a = H * SEC_PER_DAY / Cth                                   # 1/day
    T_ss = T_HALL + ETA * (J_PER_MWH / SEC_PER_DAY) * E_days / H[:, None]
    decay = np.exp(-a)

    T = np.empty((N, D + 1))
    T[:, 0] = T0 if T_start is None else T_start
    for d in range(D):
        T[:, d + 1] = T_ss[:, d] + (T[:, d] - T_ss[:, d]) * decay

    within = np.exp(-a[:, None] * S_GRID[None, :])              # (N, N_SUB+1)
    day_int = np.empty((N, D))
    chunk = max(1, int(2e6 // (N * (N_SUB + 1))))               # keep memory modest
    for s in range(0, D, chunk):
        sl = slice(s, min(s + chunk, D))
        T_fine = T_ss[:, sl, None] + (T[:, sl][:, :, None] - T_ss[:, sl, None]) * within[:, None, :]
        day_int[:, sl] = arrhenius(T_fine, Ea) @ W_TRAP
    zeros = np.zeros((N, 1))
    I_cal = np.hstack([zeros, np.cumsum(day_int, axis=1)])
    I_cyc = np.hstack([zeros, np.cumsum(E_days * day_int, axis=1)])
    Q_init = Q0 if Q_start is None else Q_start
    Q = Q_init[:, None] - kcal[:, None] * I_cal - kcyc[:, None] * I_cyc
    return T, Q


def scaled_residuals(th, sig, Ea=EA):
    T, Q = simulate(th, E_FIT, Ea=Ea)
    return np.concatenate([(T[0, mT] - T_obs[mT]) / sig[0], (Q[0, mQ] - Q_obs[mQ]) / sig[1]])


def misfit_many(ths, sig, Ea=EA):
    """Psi for many parameter sets at once (infeasible sets get +inf)."""
    ths = np.atleast_2d(ths)
    ok = np.all(ths[:, :3] > LOWER[:3], 1) & np.all(ths[:, :3] < UPPER[:3], 1) & (ths[:, 3] > 0) & (ths[:, 4] > 0)
    psi = np.full(len(ths), np.inf)
    if ok.any():
        T, Q = simulate(ths[ok], E_FIT, Ea=Ea)
        psi[ok] = (((T[:, mT] - T_obs[mT]) / sig[0]) ** 2).sum(1) + (((Q[:, mQ] - Q_obs[mQ]) / sig[1]) ** 2).sum(1)
    return psi


def calibrate(Ea=EA, x0=TH0):
    """Joint least squares with noise re-estimated from the residuals (3 passes).
    Returns best fit, covariance (J^T J)^-1 of the noise-scaled residuals, noise sds."""
    sig = np.array([1.0, 1e-3])
    for _ in range(3):
        sol = least_squares(scaled_residuals, x0, args=(sig, Ea), bounds=(LOWER, UPPER), x_scale=X_SCALE)
        T, Q = simulate(sol.x, E_FIT, Ea=Ea)
        sig = np.array([np.sqrt(np.sum((T[0, mT] - T_obs[mT]) ** 2) / (nT - 2)),
                        np.sqrt(np.sum((Q[0, mQ] - Q_obs[mQ]) ** 2) / (nQ - 3))])
        x0 = sol.x
    return sol.x, np.linalg.inv(sol.jac.T @ sol.jac), sig


def draw_feasible(mean, cov, n, rng):
    """Gaussian samples with infeasible ones (negative fade rates, out of bounds) rejected."""
    out, n_drawn = [], 0
    while sum(len(o) for o in out) < n:
        raw = rng.multivariate_normal(mean, cov, 2 * n)
        n_drawn += len(raw)
        ok = np.all(raw[:, :3] > LOWER[:3], 1) & np.all(raw[:, :3] < UPPER[:3], 1) & (raw[:, 3] > 0) & (raw[:, 4] > 0)
        out.append(raw[ok])
    samples = np.vstack(out)[:n]
    return samples, 1 - (np.vstack(out).shape[0] / n_drawn)


# %% 1 - Calibration: best fit and covariance
opt, cov, sig = calibrate()
PSI_MIN = float(np.sum(scaled_residuals(opt, sig) ** 2))
sd = np.sqrt(np.diag(cov))
corr = cov / np.outer(sd, sd)

phys_opt = np.array([p[0] for p in to_physical(opt)])
phys_sd = np.array([np.log(10) * phys_opt[0] * sd[0], np.log(10) * phys_opt[1] * sd[1],
                    sd[2], sd[3] * 1e-5, sd[4] * 1e-7])          # delta method for the log10 params
rel_sd = phys_sd / np.abs(phys_opt)

print(f"Noise estimates: sigma_T = {sig[0]:.3f} degC, sigma_Q = {sig[1]:.5f}")
print(f"Minimum misfit Psi = {PSI_MIN:.1f} from {nT + nQ} data points\n")
print("CALIBRATED PARAMETERS (physical units, +/- one standard deviation)")
for k in range(5):
    print(f"  {SHORT[k]:6s} = {phys_opt[k]:.4e} {UNITS[k]:6s} +/- {phys_sd[k]:.2e}  ({100 * rel_sd[k]:.2f} %)")
print(f"Most uncertain: {SHORT[int(np.argmax(rel_sd))]};  best constrained: {SHORT[int(np.argmin(rel_sd))]}")
print("\nCorrelation matrix")
print(pd.DataFrame(corr, index=SHORT, columns=SHORT).round(2).to_string())
print("Pairs with |correlation| > 0.5:")
for i in range(5):
    for j in range(i + 1, 5):
        if abs(corr[i, j]) > 0.5:
            print(f"  {SHORT[i]} - {SHORT[j]}: rho = {corr[i, j]:+.2f}")

# %% Figure: correlation matrix
fig, ax = plt.subplots(figsize=(5.6, 4.8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(5), SHORT)
ax.set_yticks(range(5), SHORT)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{corr[i, j]:+.2f}", ha="center", va="center",
                color="white" if abs(corr[i, j]) > 0.6 else "black", fontsize=9)
fig.colorbar(im, ax=ax, label="Correlation coefficient")
ax.set_title("Posterior parameter correlations")
fig.tight_layout()
fig.savefig(FIG / "fig2_correlation_matrix.png", dpi=300, facecolor="white")

# %% 2 - Fit to the data with posterior samples, and residuals
fit_samples, _ = draw_feasible(opt, cov, N_SHOW, rng)
T_best, Q_best = simulate(opt, E_FIT)
T_s, Q_s = simulate(fit_samples, E_FIT)

fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
for k in range(N_SHOW):
    axs[0, 0].plot(days, T_s[k], color="gray", alpha=0.25, lw=1)
    axs[0, 1].plot(days, Q_s[k], color="gray", alpha=0.25, lw=1)
axs[0, 0].plot(days, T_best[0], "r-", lw=2, label="Best fit")
axs[0, 0].plot(days[mT], T_obs[mT], "ko", ms=4, label="Data")
axs[0, 0].plot([], [], color="gray", label=f"{N_SHOW} posterior samples")
axs[0, 1].plot(days, Q_best[0], "r-", lw=2)
axs[0, 1].plot(days[mQ], Q_obs[mQ], "ko", ms=4)
axs[0, 0].set_ylabel("Cell temperature [°C]")
axs[0, 1].set_ylabel("Capacity [fraction of nameplate]")
axs[0, 0].legend(fontsize=8)
res_T = T_obs[mT] - T_best[0, mT]
res_Q = Q_obs[mQ] - Q_best[0, mQ]
axs[1, 0].axhline(0, color="k", lw=0.8)
axs[1, 0].axhspan(-sig[0], sig[0], color="tab:blue", alpha=0.12, label=r"$\pm\sigma$")
axs[1, 0].plot(days[mT], res_T, "ko", ms=4)
axs[1, 1].axhline(0, color="k", lw=0.8)
axs[1, 1].axhspan(-sig[1], sig[1], color="tab:blue", alpha=0.12)
axs[1, 1].plot(days[mQ], res_Q, "ko", ms=4)
axs[1, 0].set_ylabel("Residual [°C]")
axs[1, 1].set_ylabel("Residual [–]")
axs[1, 0].legend(fontsize=8)
for ax in axs[1]:
    ax.set_xlabel("Day (from 1 July 2024)")
    ax.axvline(91.5, color="gray", ls=":", lw=1)
for ax in axs.flat:
    ax.grid(alpha=0.3)
fig.suptitle("Calibrated model, posterior samples and residuals")
fig.tight_layout()
fig.savefig(FIG / "fig1_fit_and_residuals.png", dpi=300, facecolor="white")
for name, r, mask in [("temperature", res_T, days[mT] >= 92), ("capacity", res_Q, days[mQ] >= 92)]:
    print(f"Mean {name} residual: before regime change {r[~mask].mean():+.3g}, after {r[mask].mean():+.3g}")

# %% 3a - Misfit landscape: profile misfit vs Gaussian contours
PAIRS = [(0, 1), (3, 4), (2, 3)]             # (C_th, H), (k_cal, k_cyc), (Q0, k_cal)
LEVELS = [2.30, 6.18, 11.83]                 # Delta-Psi for 68 / 95 / 99.7 % with 2 parameters


def gaussian_delta_psi(i, j, XI, XJ):
    C = cov[np.ix_([i, j], [i, j])]
    d = np.stack([XI - opt[i], XJ - opt[j]], -1)
    return np.einsum("...a,ab,...b->...", d, np.linalg.inv(C), d)


def profile_misfit(i, j, xi, xj):
    """Delta-Psi on a grid; the other three parameters are re-optimised at each point."""
    others = [k for k in range(5) if k not in (i, j)]
    Z = np.full((len(xj), len(xi)), np.nan)
    start = opt[others].copy()
    for b, vb in enumerate(xj):
        cols = range(len(xi)) if b % 2 == 0 else range(len(xi) - 1, -1, -1)   # warm-start snake
        for a in cols:
            def f(po):
                th = opt.copy()
                th[i], th[j], th[others] = xi[a], vb, po
                return scaled_residuals(th, sig)
            s = least_squares(f, start, bounds=(LOWER[others], UPPER[others]), x_scale=X_SCALE[others])
            Z[b, a] = np.sum(s.fun ** 2) - PSI_MIN
            start = s.x
    return Z


grids = {}
for (i, j) in PAIRS:
    xi = np.linspace(max(opt[i] - 4 * sd[i], LOWER[i]), opt[i] + 4 * sd[i], N_GRID)
    xj = np.linspace(max(opt[j] - 4 * sd[j], LOWER[j]), opt[j] + 4 * sd[j], N_GRID)
    XI, XJ = np.meshgrid(xi, xj)
    Z = profile_misfit(i, j, xi, xj) if RUN_PROFILE else None
    grids[(i, j)] = (xi, xj, Z, gaussian_delta_psi(i, j, XI, XJ))

if RUN_PROFILE:
    fig, axs = plt.subplots(1, len(PAIRS), figsize=(5.2 * len(PAIRS), 4.6))
    for ax, (i, j) in zip(axs, PAIRS):
        xi, xj, Z, ZG = grids[(i, j)]
        XI, XJ = np.meshgrid(xi, xj)
        cf = ax.contourf(XI, XJ, np.log10(np.clip(Z, 1e-2, None)), levels=20, cmap="viridis")
        ax.contour(XI, XJ, Z, levels=LEVELS, colors="white", linewidths=1.4)
        ax.contour(XI, XJ, ZG, levels=LEVELS, colors="red", linewidths=1.2, linestyles="--")
        ax.plot(opt[i], opt[j], "r*", ms=13)
        ax.set_xlabel(LABELS[i])
        ax.set_ylabel(LABELS[j])
        fig.colorbar(cf, ax=ax).set_label(r"$\log_{10}(\Psi - \Psi_{min})$")
        inside = Z < LEVELS[-1]
        print(f"{SHORT[i]}-{SHORT[j]}: largest |profile - Gaussian| Delta-Psi inside 99.7 % region = "
              f"{np.nanmax(np.abs(Z[inside] - ZG[inside])):.2f}")
    axs[0].legend(handles=[Patch(fc="none", ec="white", label="Profile misfit"),
                           Patch(fc="none", ec="red", ls="--", label="Gaussian approximation")],
                  loc="upper left", fontsize=8, facecolor="0.4", labelcolor="white")
    fig.suptitle("Misfit landscape (contours at 68 / 95 / 99.7 %)")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_misfit_landscape.png", dpi=300, facecolor="white")

# %% 3b - Gaussian vs true posterior (importance sampling of exp(-Psi/2))
N_IS = 20000
raw = rng.multivariate_normal(opt, cov, N_IS)
logw = -0.5 * misfit_many(raw, sig) - multivariate_normal(opt, cov).logpdf(raw)
logw -= logw[np.isfinite(logw)].max()
w = np.where(np.isfinite(logw), np.exp(logw), 0.0)
w /= w.sum()
ess = 1.0 / np.sum(w ** 2)
true_post = raw[rng.choice(N_IS, 10000, p=w)]
feasible = (raw[:, 3] > 0) & (raw[:, 4] > 0)
gauss_post = raw[feasible]
print(f"\nImportance-sampling check: ESS/N = {ess / N_IS:.2f}  (1 = Gaussian exactly right)")
print(f"Gaussian draws with a negative fade rate (rejected): {100 * np.mean(~feasible):.1f} %")

fig, axs = plt.subplots(1, len(PAIRS), figsize=(5.2 * len(PAIRS), 4.4))
for ax, (i, j) in zip(axs, PAIRS):
    xi, xj, _, ZG = grids[(i, j)]
    XI, XJ = np.meshgrid(xi, xj)
    ax.plot(true_post[::4, i], true_post[::4, j], ".", ms=2, color="tab:orange", alpha=0.4)
    ax.plot(gauss_post[::6, i], gauss_post[::6, j], ".", ms=2, color="tab:blue", alpha=0.25)
    ax.contour(XI, XJ, ZG, levels=LEVELS, colors="k", linewidths=1, linestyles="--")
    ax.plot(opt[i], opt[j], "r*", ms=13)
    ax.set_xlabel(LABELS[i])
    ax.set_ylabel(LABELS[j])
axs[0].legend(handles=[Patch(fc="tab:orange", label="True posterior (resampled)"),
                       Patch(fc="tab:blue", label="Gaussian samples"),
                       Patch(fc="none", ec="k", ls="--", label="Gaussian 68/95/99.7 %")],
              fontsize=8, loc="upper left")
fig.suptitle(f"Gaussian approximation vs true posterior (ESS/N = {ess / N_IS:.2f})")
fig.tight_layout()
fig.savefig(FIG / "figA_posterior_comparison.png", dpi=300, facecolor="white")

# %% 4 - The calendar-fade marginal sits close to zero: infeasible samples
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(raw[:, 3].min(), raw[:, 3].max(), 60)
ax.hist(raw[:, 3], bins=bins, density=True, alpha=0.45, color="tab:blue", label="Gaussian approximation")
ax.hist(true_post[:, 3], bins=bins, density=True, histtype="step", lw=2, color="tab:orange",
        label="True posterior (resampled)")
ax.axvspan(bins[0], 0, color="red", alpha=0.12, label=f"Infeasible: $k_{{cal}}$ < 0 ({100 * np.mean(raw[:, 3] <= 0):.1f} % of samples)")
ax.axvline(opt[3], color="k", ls="--", lw=1, label="Best fit")
ax.set_xlabel(LABELS[3])
ax.set_ylabel("Probability density")
ax.set_title("Calendar fade rate: posterior and infeasible region")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig4_kcal_marginal.png", dpi=300, facecolor="white")

# %% 5 - Ten-year forecasts for the three options
SCENARIOS = {
    "Warranty (1 cycle/day)": dict(pool=POOL_WARRANTY, h_mult=1.0, colour="tab:green"),
    "Aggressive (2 cycles/day)": dict(pool=POOL_AGGRESSIVE, h_mult=1.0, colour="tab:red"),
    "Aggressive + chiller": dict(pool=POOL_AGGRESSIVE, h_mult=CHILL, colour="tab:blue"),
}
_u = np.random.default_rng(SEED + 1).random((N_FORECAST, N_FUT))     # common random numbers


def future_throughput(pool, n):
    """Daily throughput for the forecast: resampled from the record (or its mean), with
    the same maintenance weeks every year. Same random numbers for every scenario."""
    if RESAMPLE_THROUGHPUT:
        Ef = pool[(_u[:n] * len(pool)).astype(int)]
    else:
        Ef = np.full((n, N_FUT), pool.mean())
    for yr in range(FORECAST_YEARS):
        for start, length in MAINT_BLOCKS:
            Ef[:, yr * 365 + start: yr * 365 + start + length] = 0.0
    return Ef


def forecast(ths, pool, h_mult, Ea=EA):
    """Record period with the actual throughput, then the forecast from each sample's own
    end-of-record state. The chiller (H x h_mult) applies only from the forecast start."""
    T_d, Q_d = simulate(ths, E_FIT, Ea=Ea)
    Ef = future_throughput(pool, len(ths))
    T_f, Q_f = simulate(ths, Ef, T_start=T_d[:, -1], Q_start=Q_d[:, -1], Ea=Ea, H_mult=h_mult)
    return T_d, Q_d, T_f, Q_f, Ef


def years_to(Q_f, thr):
    below = Q_f < thr
    return np.where(below.any(1), np.argmax(below, 1) / 365.0, np.inf)


def summarise(T_f, Q_f, Ef):
    operating = Ef > 0
    out = {
        "Mean operating temperature [°C]": np.nanmean(np.where(operating, T_f[:, 1:], np.nan), 1),
        "Max temperature [°C]": T_f[:, 30:].max(1),            # skip the cool-down from the trial
        f"Capacity after {FORECAST_YEARS} yr [-]": np.clip(Q_f[:, N_FUT], 0, None),
        "Fade per full cycle, year 1 [%]": 100 * (Q_f[:, 0] - Q_f[:, 365]) / (Ef[:, :365].sum(1) / Q_CYCLE),
    }
    for thr in EOLS:
        out[f"Years to {int(100 * thr)} % capacity"] = years_to(Q_f, thr)
    return out


def interval(v):
    f = v[np.isfinite(v)]
    if len(f) == 0:
        return np.nan, np.nan, np.nan, 1.0
    lo, med, hi = np.percentile(f, [PI[0], 50, PI[1]])
    return med, lo, hi, 1 - len(f) / len(v)


fc_samples, frac_rejected = draw_feasible(opt, cov, N_FORECAST, rng)
results, summaries, rows = {}, {}, []
for name, sc in SCENARIOS.items():
    results[name] = forecast(fc_samples, sc["pool"], sc["h_mult"])
    summaries[name] = summarise(results[name][2], results[name][3], results[name][4])
    for metric, v in summaries[name].items():
        med, lo, hi, never = interval(v)
        rows.append(dict(scenario=name, metric=metric, median=med, pi_low=lo, pi_high=hi,
                         frac_not_reached=never))
summary = pd.DataFrame(rows)
summary.to_csv(FIG / "forecast_summary.csv", index=False)

print(f"\nFORECASTS: {N_FORECAST} posterior samples, {PI[1] - PI[0]} % prediction intervals, "
      f"years counted from the end of the record (30 Dec 2024)")
for name in SCENARIOS:
    print(f"\n{name}")
    for _, r in summary[summary.scenario == name].iterrows():
        extra = f"  ({100 * r.frac_not_reached:.0f} % of samples not reached)" if r.frac_not_reached > 0 else ""
        print(f"  {r.metric:34s} {r['median']:8.3g}   [{r.pi_low:.3g}, {r.pi_high:.3g}]{extra}")

print("\nIn the course notes' wording:")
for name in SCENARIOS:
    med, lo, hi, _ = interval(summaries[name][f"Years to {int(100 * EOLS[0])} % capacity"])
    print(f"  {name}: we are {PI[1] - PI[0]} % confident capacity falls to {int(100 * EOLS[0])} % "
          f"between {lo:.1f} and {hi:.1f} years after the end of the record.")

# %% Figure: coursebook-style futures, one panel per option
t_data = days / 365.25
t_fut = (days[-1] + np.arange(N_FUT + 1)) / 365.25
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for ax, (name, sc) in zip(axs, SCENARIOS.items()):
    T_d, Q_d, T_f, Q_f, _ = results[name]
    lo, hi = np.percentile(np.clip(Q_f, 0, None), PI, axis=0)
    ax.fill_between(t_fut, lo, hi, color=sc["colour"], alpha=0.2, label=f"{PI[1] - PI[0]} % prediction interval")
    for k in range(N_SHOW):
        ax.plot(np.r_[t_data, t_fut], np.clip(np.r_[Q_d[k], Q_f[k]], 0, None), color="gray", alpha=0.2, lw=0.8)
    bT_d, bQ_d = simulate(opt, E_FIT)
    E_mean = np.full(N_FUT, sc["pool"].mean())
    for yr in range(FORECAST_YEARS):
        for start, length in MAINT_BLOCKS:
            E_mean[yr * 365 + start: yr * 365 + start + length] = 0.0
    _, bQ_f = simulate(opt, E_mean, T_start=bT_d[:, -1], Q_start=bQ_d[:, -1], H_mult=sc["h_mult"])
    ax.plot(np.r_[t_data, t_fut], np.clip(np.r_[bQ_d[0], bQ_f[0]], 0, None), "r-", lw=2, label="Best fit")
    ax.plot(days[mQ] / 365.25, Q_obs[mQ], "ko", ms=3, label="Data")
    for thr in EOLS:
        ax.axhline(thr, color="k", ls="--", lw=0.8)
    ax.axvline(t_data[-1], color="gray", ls=":", lw=1)
    ax.set_title(name)
    ax.set_xlabel("Years since 1 July 2024")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
axs[0].set_ylabel("Capacity [fraction of nameplate]")
axs[0].plot([], [], color="gray", label=f"{N_SHOW} posterior samples")
axs[0].legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "fc1_capacity_by_option.png", dpi=300, facecolor="white")

# %% Figure: the three options compared (capacity and temperature)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
for name, sc in SCENARIOS.items():
    T_d, Q_d, T_f, Q_f, _ = results[name]
    q = np.percentile(np.clip(Q_f, 0, None), [PI[0], 50, PI[1]], axis=0)
    a1.fill_between(t_fut, q[0], q[2], color=sc["colour"], alpha=0.25)
    a1.plot(t_fut, q[1], color=sc["colour"], lw=2, label=name)
    Tw = uniform_filter1d(T_f, size=7, axis=1, mode="nearest")
    tq = np.percentile(Tw, [PI[0], 50, PI[1]], axis=0)
    a2.fill_between(t_fut, tq[0], tq[2], color=sc["colour"], alpha=0.25)
    a2.plot(t_fut, tq[1], color=sc["colour"], lw=1.4, label=name)
a1.plot(days[mQ] / 365.25, Q_obs[mQ], "ko", ms=3, label="Data")
a2.plot(days[mT] / 365.25, T_obs[mT], "ko", ms=3, label="Data")
for thr in EOLS:
    a1.axhline(thr, color="k", ls="--", lw=0.8)
    a1.text(t_fut[-1], thr + 0.01, f"{int(100 * thr)} %", ha="right", fontsize=8)
for ax in (a1, a2):
    ax.axvline(t_data[-1], color="gray", ls=":", lw=1)
    ax.grid(alpha=0.3)
a1.set_ylim(0, 1.02)
a1.set_ylabel("Capacity [fraction of nameplate]")
a1.set_title(f"Capacity: median and {PI[1] - PI[0]} % prediction interval")
a1.legend(fontsize=8, loc="lower left")
a2.set_ylabel("Cell temperature [°C] (7-day mean)")
a2.set_xlabel("Years since 1 July 2024")
a2.set_title(f"Cell temperature: median and {PI[1] - PI[0]} % prediction interval")
a2.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "fc2_options_compared.png", dpi=300, facecolor="white")

# %% Figure: years to end of life
fig, axs = plt.subplots(1, len(EOLS), figsize=(6 * len(EOLS), 4.2))
for ax, thr in zip(np.atleast_1d(axs), EOLS):
    key = f"Years to {int(100 * thr)} % capacity"
    for name, sc in SCENARIOS.items():
        v = summaries[name][key]
        ax.hist(v[np.isfinite(v)], bins=40, alpha=0.55, color=sc["colour"], label=name)
    ax.set_xlabel(f"Years after end of record until capacity < {int(100 * thr)} %")
    ax.set_ylabel("Number of posterior samples")
    ax.grid(alpha=0.3)
np.atleast_1d(axs)[0].legend(fontsize=8)
fig.suptitle("Time to end of life")
fig.tight_layout()
fig.savefig(FIG / "fc3_years_to_eol.png", dpi=300, facecolor="white")

# %% 6 - Sensitivity to fixed assumptions
KEY = f"Years to {int(100 * EOLS[0])} % capacity"

# (a) Activation energy: recalibrate at each value so every case still fits the data
ea_rows = []
for Ea in EA_VALUES:
    o, c, s = calibrate(Ea, x0=opt)
    th, _ = draw_feasible(o, c, N_SENS, np.random.default_rng(SEED + 2))
    kc, ky = to_physical(o)[3][0], to_physical(o)[4][0]
    print(f"\nEa = {Ea / 1e3:.0f} kJ/mol: k_cal = {kc:.2e}/day, k_cyc = {ky:.2e}/MWh, "
          f"misfit = {np.sum(scaled_residuals(o, sig, Ea) ** 2):.1f}")
    for name, sc in SCENARIOS.items():
        _, _, T_f, Q_f, Ef = forecast(th, sc["pool"], sc["h_mult"], Ea=Ea)
        med, lo, hi, _ = interval(years_to(Q_f, EOLS[0]))
        ea_rows.append(dict(Ea=Ea / 1e3, scenario=name, median=med, lo=lo, hi=hi))
        print(f"  {name:28s} {KEY}: {med:.2f} [{lo:.2f}, {hi:.2f}]")
ea_df = pd.DataFrame(ea_rows)

# (b) Chiller effectiveness
ch_rows = []
th = fc_samples[:N_SENS]
print()
for cf in CHILL_VALUES:
    _, _, T_f, Q_f, Ef = forecast(th, POOL_AGGRESSIVE, cf)
    med, lo, hi, _ = interval(years_to(Q_f, EOLS[0]))
    mT_op = np.median(np.nanmean(np.where(Ef > 0, T_f[:, 1:], np.nan), 1))
    ch_rows.append(dict(factor=cf, median=med, lo=lo, hi=hi, temp=mT_op))
    print(f"Chiller H x {cf:.1f}: operating T {mT_op:.1f} degC, {KEY}: {med:.2f} [{lo:.2f}, {hi:.2f}]")
ch_df = pd.DataFrame(ch_rows)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
offsets = {n: o for n, o in zip(SCENARIOS, [-1.2, 0, 1.2])}
for name, sc in SCENARIOS.items():
    d = ea_df[ea_df.scenario == name]
    a1.errorbar(d.Ea + offsets[name], d["median"], yerr=[d["median"] - d.lo, d.hi - d["median"]],
                fmt="o", color=sc["colour"], capsize=4, label=name)
a1.set_xticks(ea_df.Ea.unique())
a1.set_xlabel("Activation energy [kJ/mol]")
a1.set_ylabel(f"Years to {int(100 * EOLS[0])} % capacity")
a1.set_title("Sensitivity to activation energy (recalibrated each time)")
a1.legend(fontsize=8)
a2.errorbar(ch_df.factor, ch_df["median"], yerr=[ch_df["median"] - ch_df.lo, ch_df.hi - ch_df["median"]],
            fmt="o-", color="tab:blue", capsize=4, label="Aggressive + chiller")
for _, r in ch_df.iterrows():
    a2.annotate(f"{r.temp:.1f} °C", (r.factor, r["median"]), textcoords="offset points", xytext=(8, -12), fontsize=8)
for name in ["Warranty (1 cycle/day)", "Aggressive (2 cycles/day)"]:
    med = interval(summaries[name][KEY])[0]
    a2.axhline(med, color=SCENARIOS[name]["colour"], ls="--", lw=1, label=f"{name} (median)")
a2.set_xticks(CHILL_VALUES)
a2.set_xlabel("Chiller multiplier on heat removal H")
a2.set_ylabel(f"Years to {int(100 * EOLS[0])} % capacity")
a2.set_title("Sensitivity to chiller effectiveness")
a2.legend(fontsize=8)
for ax in (a1, a2):
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fc4_sensitivity.png", dpi=300, facecolor="white")

print(f"\nFigures and forecast_summary.csv written to {FIG}")
plt.show()