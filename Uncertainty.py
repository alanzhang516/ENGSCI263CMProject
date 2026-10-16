# # ENGSCI 263 - Uncertainty analysis for the battery model
#
# Answers the same questions as the Uncertainty Lab, for the battery which is to say:
#   Task 1  covariance matrix, marginal standard deviations, correlations
#   Task 2  misfit landscape, Gaussian posterior, comparison with the "true" posterior
#           (+ optional interactive click-through plot)
#   Task 3  best model + 50 posterior samples, extended ten years into the future

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import least_squares
from scipy.stats import multivariate_normal

rng = np.random.default_rng(263)
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)

eta, T_hall, Ea, R, T_ref = 0.015, 20.0, 50.0e3, 8.314, 25.0
SEC_PER_DAY, J_PER_MWH = 86400.0, 3.6e9
DT, N_SUB = 0.1, 10
Q_CYCLE, EOL = 400.0, 0.80
SCENARIOS = [("A: warranty (1 cycle/day)", 1 * Q_CYCLE, 1.0),
             ("B: aggressive (2 cycles/day)", 2 * Q_CYCLE, 1.0),
             ("C: aggressive + chilled", 2 * Q_CYCLE, 2.0)]
FORECAST_YEARS = 10

df = pd.read_csv("data/grid_battery_data.csv")
days = df["day"].to_numpy()
E = df["throughput_[MWh/day]"].to_numpy(float)
T_obs = df["cell_temp_[degC]"].to_numpy()
Q_obs = df["capacity_[fraction]"].to_numpy()
mT, mQ = ~np.isnan(T_obs), ~np.isnan(Q_obs)
nT, nQ = mT.sum(), mQ.sum()
T0, nd, Ed = T_obs[0], len(days) - 1, E[:-1]
s_grid = np.arange(N_SUB + 1) * DT


def arrhenius(T):
    return np.exp(-Ea / R * (1.0 / (T + 273.15) - 1.0 / (T_ref + 273.15)))


# theta = [log10 C_th, log10 H, Q0, k_cal*1e5, k_cyc*1e7]  (scaled so entries are O(1))
LABELS = [r"$\log_{10} C_{th}$", r"$\log_{10} H$", r"$Q_0$",
          r"$k_{cal}$ [$10^{-5}$/day]", r"$k_{cyc}$ [$10^{-7}$/MWh]"]
SHORT = ["C_th", "H", "Q0", "k_cal", "k_cyc"]
LOWER = np.array([8.0, 3.0, 0.9, 0.0, 0.0])
UPPER = np.array([11.0, 6.0, 1.1, np.inf, np.inf])
X_SCALE = np.array([0.05, 0.01, 1e-3, 1.0, 1.0])


def simulate(th):
    """Temperature and capacity at the start of each day."""
    Cth, H, Q0, kcal, kcyc = 10 ** th[0], 10 ** th[1], th[2], th[3] * 1e-5, th[4] * 1e-7
    a = H * SEC_PER_DAY / Cth
    b = eta * J_PER_MWH / Cth
    T_ss = T_hall + b * Ed / a
    decay = np.exp(-a)
    T_day = np.empty(nd + 1)
    T_day[0] = T0
    for d in range(nd):                      # exact solution of the ODE over each day
        T_day[d + 1] = T_ss[d] + (T_day[d] - T_ss[d]) * decay
    T_fine = T_ss[:, None] + (T_day[:-1, None] - T_ss[:, None]) * np.exp(-a * s_grid)[None, :]
    fT = arrhenius(T_fine)
    day_int = DT * (fT.sum(1) - 0.5 * (fT[:, 0] + fT[:, -1]))
    I_cal = np.concatenate([[0.0], np.cumsum(day_int)])
    I_cyc = np.concatenate([[0.0], np.cumsum(Ed * day_int)])
    return T_day, Q0 - kcal * I_cal - kcyc * I_cyc


def scaled_residuals(th, sig):
    T_day, Q_day = simulate(th)
    return np.concatenate([(T_day[mT] - T_obs[mT]) / sig[0], (Q_day[mQ] - Q_obs[mQ]) / sig[1]])


def misfit(th, sig):
    """Psi(theta): sum of squared, noise-scaled residuals.  Posterior ~ exp(-Psi/2)."""
    return float(np.sum(scaled_residuals(th, sig) ** 2))


# %% Calibration -> best fit and covariance matrix (what curve_fit returns as opt, cov)
th0 = np.array([np.log10(6e8), np.log10(2e4), 1.0, 4.79, 1.575])
sig, x0 = np.array([1.0, 1e-3]), th0
for _ in range(3):                           # fit, estimate noise from residuals, refit
    sol = least_squares(scaled_residuals, x0, args=(sig,), bounds=(LOWER, UPPER), x_scale=X_SCALE)
    T_day, Q_day = simulate(sol.x)
    sig = np.array([np.sqrt(np.sum((T_day[mT] - T_obs[mT]) ** 2) / (nT - 2)),
                    np.sqrt(np.sum((Q_day[mQ] - Q_obs[mQ]) ** 2) / (nQ - 3))])
    x0 = sol.x
opt = sol.x
cov = np.linalg.inv(sol.jac.T @ sol.jac)     # Gaussian posterior covariance
PSI_MIN = misfit(opt, sig)
print(f"Noise: sigma_T = {sig[0]:.3f} degC, sigma_Q = {sig[1]:.5f};  minimum misfit Psi = {PSI_MIN:.1f} "
      f"({nT + nQ} data points)")

# %% Task 1 - covariance, standard deviations, correlations
sd = np.sqrt(np.diag(cov))
corr = cov / np.outer(sd, sd)
print("\nCovariance matrix (theta in the scaled units of LABELS):")
print(pd.DataFrame(cov, index=SHORT, columns=SHORT).to_string(float_format=lambda v: f"{v: .3e}"))
print("\nMarginal standard deviations")
print(f"{'param':8s}{'best fit':>12s}{'variance':>12s}{'std dev':>12s}{'relative sd':>13s}")
for k in range(5):
    print(f"{SHORT[k]:8s}{opt[k]:12.5g}{cov[k, k]:12.3e}{sd[k]:12.3e}{100 * sd[k] / abs(opt[k]):12.2f}%")
print("\nCorrelation matrix  rho_ij = C_ij / (sd_i sd_j)")
print(pd.DataFrame(corr, index=SHORT, columns=SHORT).round(2).to_string())

# physical-unit standard deviations (delta method for the log10 parameters)
Cth_hat, H_hat = 10 ** opt[0], 10 ** opt[1]
print(f"\nIn physical units:  C_th = {Cth_hat:.3e} J/K (+/- {np.log(10) * Cth_hat * sd[0]:.2e}),  "
      f"H = {H_hat:.0f} W/K (+/- {np.log(10) * H_hat * sd[1]:.0f}),  "
      f"k_cal = {opt[3] * 1e-5:.2e}/day (+/- {sd[3] * 1e-5:.1e}),  "
      f"k_cyc = {opt[4] * 1e-7:.2e}/MWh (+/- {sd[4] * 1e-7:.1e})")

rel = sd / np.abs(opt)
print(f"\nMost uncertain parameter in relative terms: {SHORT[int(np.argmax(rel))]} "
      f"({100 * rel.max():.0f}%); best constrained: {SHORT[int(np.argmin(rel))]} ({100 * rel.min():.2f}%).")
print("Pairs with |correlation| > 0.5:")
for i in range(5):
    for j in range(i + 1, 5):
        if abs(corr[i, j]) > 0.5:
            kind = "positive: both tend to be high or low together" if corr[i, j] > 0 else \
                   "negative: one can be higher only if the other is lower (a trade-off)"
            print(f"  {SHORT[i]} - {SHORT[j]}: rho = {corr[i, j]:+.2f}  ({kind})")

# %% Task 2a - misfit landscape (profile) vs Gaussian approximation
PAIRS = [(0, 1), (3, 4), (2, 3)]             # (C_th, H), (k_cal, k_cyc), (Q0, k_cal)
LEVELS = [2.30, 6.18, 11.83]                 # Delta-Psi for 68 %, 95 %, 99.7 % (2 degrees of freedom)
N_GRID = 25


def profile_misfit(i, j, xi, xj):
    """Delta-Psi on a grid: the other three parameters are re-optimised at every grid point."""
    others = [k for k in range(5) if k not in (i, j)]
    Z = np.full((len(xj), len(xi)), np.nan)
    start = opt[others].copy()
    for b, vb in enumerate(xj):
        cols = range(len(xi)) if b % 2 == 0 else range(len(xi) - 1, -1, -1)   # serpentine warm start
        for a in cols:
            def f(po):
                th = opt.copy()
                th[i], th[j], th[others] = xi[a], vb, po
                return scaled_residuals(th, sig)
            s = least_squares(f, start, bounds=(LOWER[others], UPPER[others]), x_scale=X_SCALE[others])
            Z[b, a] = np.sum(s.fun ** 2) - PSI_MIN
            start = s.x
    return Z


def gaussian_delta_psi(i, j, XI, XJ):
    C = cov[np.ix_([i, j], [i, j])]
    d = np.stack([XI - opt[i], XJ - opt[j]], -1)
    return np.einsum("...a,ab,...b->...", d, np.linalg.inv(C), d)


fig, axs = plt.subplots(1, len(PAIRS), figsize=(5.2 * len(PAIRS), 4.6))
grids = {}
for ax, (i, j) in zip(axs, PAIRS):
    xi = np.linspace(max(opt[i] - 4 * sd[i], LOWER[i]), opt[i] + 4 * sd[i], N_GRID)
    xj = np.linspace(max(opt[j] - 4 * sd[j], LOWER[j]), opt[j] + 4 * sd[j], N_GRID)
    XI, XJ = np.meshgrid(xi, xj)
    Z = profile_misfit(i, j, xi, xj)
    ZG = gaussian_delta_psi(i, j, XI, XJ)
    grids[(i, j)] = (xi, xj, Z, ZG)
    cf = ax.contourf(XI, XJ, np.log10(np.clip(Z, 1e-2, None)), levels=20, cmap="viridis")
    ax.contour(XI, XJ, Z, levels=LEVELS, colors="white", linewidths=1.4)
    ax.contour(XI, XJ, ZG, levels=LEVELS, colors="red", linewidths=1.2, linestyles="--")
    ax.plot(opt[i], opt[j], "r*", ms=13)
    ax.set_xlabel(LABELS[i])
    ax.set_ylabel(LABELS[j])
    fig.colorbar(cf, ax=ax).set_label(r"$\log_{10}(\Psi - \Psi_{min})$")
    inside = Z < LEVELS[-1]
    dev = np.nanmax(np.abs(Z[inside] - ZG[inside])) if inside.any() else np.nan
    print(f"{SHORT[i]}-{SHORT[j]}: largest |profile - Gaussian| misfit inside the 99.7% region = {dev:.2f}")
axs[0].legend(handles=[Patch(fc="none", ec="white", label="Profile misfit (true)"),
                       Patch(fc="none", ec="red", ls="--", label="Gaussian approx.")],
              loc="upper left", fontsize=8, facecolor="0.4", labelcolor="white")
fig.suptitle("Misfit landscape (contours at 68 / 95 / 99.7 %)")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "lab_misfit_landscape.png", dpi=300, facecolor="white")

# %% Task 2b - Gaussian vs the true posterior (importance sampling of exp(-Psi/2))
N_IS = 20000
raw = rng.multivariate_normal(opt, cov, N_IS)


def log_target(th):
    if not (np.all(th[:3] > LOWER[:3]) and np.all(th[:3] < UPPER[:3]) and th[3] > 0 and th[4] > 0):
        return -np.inf
    return -0.5 * misfit(th, sig)


logw = np.array([log_target(t) for t in raw]) - multivariate_normal(opt, cov).logpdf(raw)
logw -= logw[np.isfinite(logw)].max()
w = np.where(np.isfinite(logw), np.exp(logw), 0.0)
w /= w.sum()
ess = 1.0 / np.sum(w ** 2)
true_post = raw[rng.choice(N_IS, 10000, p=w)]
gauss_post = raw[(raw[:, 3] > 0) & (raw[:, 4] > 0)]
print(f"\nImportance-sampling check: ESS/N = {ess / N_IS:.2f}   (1 = Gaussian exactly right)")
print(f"Gaussian draws with a negative fade rate: {100 * np.mean(~((raw[:, 3] > 0) & (raw[:, 4] > 0))):.1f}%")

fig, axs = plt.subplots(1, len(PAIRS), figsize=(5.2 * len(PAIRS), 4.4))
for ax, (i, j) in zip(axs, PAIRS):
    xi, xj, Z, ZG = grids[(i, j)]
    XI, XJ = np.meshgrid(xi, xj)
    ax.plot(true_post[::4, i], true_post[::4, j], ".", ms=2, color="tab:orange", alpha=0.4)
    ax.plot(gauss_post[::6, i], gauss_post[::6, j], ".", ms=2, color="tab:blue", alpha=0.25)
    ax.contour(XI, XJ, ZG, levels=LEVELS, colors="k", linewidths=1, linestyles="--")
    ax.plot(opt[i], opt[j], "r*", ms=13)
    ax.set_xlabel(LABELS[i])
    ax.set_ylabel(LABELS[j])
axs[0].legend(handles=[Patch(fc="tab:orange", label="True posterior (resampled)"),
                       Patch(fc="tab:blue", label="Gaussian samples"),
                       Patch(fc="none", ec="k", ls="--", label="Gaussian 68/95/99.7%")],
              fontsize=8, loc="upper left")
fig.suptitle("Gaussian approximation vs true posterior")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "lab_posterior_comparison.png", dpi=300, facecolor="white")


# %% Task 2c - interactive click-through (needs an interactive backend, e.g. %matplotlib widget)
# IMPORTANT NOTE HERE: My laptop actually can't run interactive plots for some reason, if this doesn't work
# for you guys just delete this stuff
def plot_interactive(i=3, j=4):

    xi, xj, Z, ZG = grids[(i, j)]
    XI, XJ = np.meshgrid(xi, xj)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    axR.contourf(XI, XJ, ZG, levels=np.linspace(0, 40, 21), cmap="viridis_r")
    axR.plot(opt[i], opt[j], "r*", ms=12)
    axR.set_xlabel(LABELS[i])
    axR.set_ylabel(LABELS[j])
    axR.set_title("Click a parameter pair")
    axL.plot(days[mQ], Q_obs[mQ], "ko", ms=4, label="Capacity data")
    axL.plot(days, simulate(opt)[1], "r-", lw=2, label="Best fit")
    (line,) = axL.plot([], [], "b-", lw=1.5, label="Selected")
    title = axL.set_title("")
    axL.set_xlabel("Day")
    axL.set_ylabel("Capacity")
    axL.legend()

    def onclick(ev):
        if ev.inaxes is not axR:
            return
        th = opt.copy()
        th[i], th[j] = ev.xdata, ev.ydata
        line.set_data(days, simulate(th)[1])
        title.set_text(f"Psi = {misfit(th, sig):.1f}  (Psi_min = {PSI_MIN:.1f})")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)
    return fig



# %% Task 3 - best model + 50 posterior samples, extended ten years into the future
samples = gauss_post[:50]
t_end = days[-1]
t_fut = np.linspace(t_end, t_end + FORECAST_YEARS * 365.25, 200)


def future_capacity(th, Esc, mult):
    """Capacity after the end of the data.  The cell time constant is only ~3 days, so under a
    constant regime it sits at its steady-state temperature and Q falls linearly in time."""
    Cth, H, kcal, kcyc = 10 ** th[0], 10 ** th[1], th[3] * 1e-5, th[4] * 1e-7
    a = mult * H * SEC_PER_DAY / Cth
    T_ss = T_hall + (eta * J_PER_MWH / Cth) * Esc / a
    rate = (kcal + kcyc * Esc) * arrhenius(T_ss)
    return simulate(th)[1][-1] - rate * (t_fut - t_end)


# (a) data period: temperature and capacity
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
for k, th in enumerate(samples):
    Ts, Qs = simulate(th)
    a1.plot(days, Ts, color="gray", alpha=0.3, lw=1, label="Posterior samples" if k == 0 else None)
    a2.plot(days, Qs, color="gray", alpha=0.3, lw=1)
Tb, Qb = simulate(opt)
a1.plot(days, Tb, "r-", lw=2, label="Best model")
a2.plot(days, Qb, "r-", lw=2)
a1.plot(days[mT], T_obs[mT], "ko", ms=4, label="Data")
a2.plot(days[mQ], Q_obs[mQ], "ko", ms=4)
a1.set_ylabel("Cell temperature [°C]")
a2.set_ylabel("Capacity [fraction]")
a2.set_xlabel("Day (from 1 July 2024)")
a1.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "lab_ensemble_data.png", dpi=300, facecolor="white")

# (b) the three options over the next ten years
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, (lab, Esc, mult) in zip(axs, SCENARIOS):
    for k, th in enumerate(samples):
        ax.plot(np.r_[days, t_fut] / 365.25, np.r_[simulate(th)[1], future_capacity(th, Esc, mult)],
                color="gray", alpha=0.3, lw=1, label="Posterior samples" if k == 0 else None)
    ax.plot(np.r_[days, t_fut] / 365.25, np.r_[simulate(opt)[1], future_capacity(opt, Esc, mult)],
            "r-", lw=2, label="Best model")
    ax.plot(days[mQ] / 365.25, Q_obs[mQ], "ko", ms=3, label="Data")
    ax.axhline(EOL, color="k", ls="--", lw=1)
    ax.set_title(lab)
    ax.set_xlabel("Years since 1 July 2024")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
axs[0].set_ylabel("Capacity [fraction]")
axs[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "lab_ensemble_forecast.png", dpi=300, facecolor="white")
print("\nFigures written: figures/lab_misfit_landscape.png, lab_posterior_comparison.png, "
      "lab_ensemble_data.png, lab_ensemble_forecast.png")