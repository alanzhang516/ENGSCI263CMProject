from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# File locations
DATA_PATH = Path("data/grid_battery_data.csv")
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

DAY = "day"
THROUGHPUT = "throughput_[MWh/day]"
TEMPERATURE = "cell_temp_[degC]"
CAPACITY = "capacity_[fraction]"

# Aggressive regime began on day 92
REGIME_CHANGE_DAY = 92


# 1. Energy throughput graph
fig1, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(
    df[DAY],
    df[THROUGHPUT],
    color="tab:blue",
    linewidth=1.8
)

ax1.axvline(
    REGIME_CHANGE_DAY,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Aggressive regime begins"
)

ax1.set_title("Daily Battery Energy Throughput")
ax1.set_xlabel("Day")
ax1.set_ylabel("Energy throughput (MWh/day)")
ax1.grid(False)
ax1.legend()

fig1.tight_layout()
fig1.savefig(
    FIGURE_DIR / "energy_throughput.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)


# 2. Cell-temperature graph
temperature_mask = df[TEMPERATURE].notna()

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.scatter(
    df.loc[temperature_mask, DAY],
    df.loc[temperature_mask, TEMPERATURE],
    color="tab:red",
    s=45,
    label="Observed cell temperature"
)

ax2.axhline(
    20,
    color="grey",
    linestyle="--",
    linewidth=1.2,
    label="Hall temperature"
)

ax2.axvline(
    REGIME_CHANGE_DAY,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Aggressive regime begins"
)

ax2.set_title("Observed Battery Cell Temperature")
ax2.set_xlabel("Day")
ax2.set_ylabel("Cell temperature (°C)")
ax2.grid(False)
ax2.legend()

fig2.tight_layout()
fig2.savefig(
    FIGURE_DIR / "cell_temperature.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)


# 3. Capacity graph
capacity_mask = df[CAPACITY].notna()

fig3, ax3 = plt.subplots(figsize=(10, 5))

ax3.scatter(
    df.loc[capacity_mask, DAY],
    df.loc[capacity_mask, CAPACITY],
    color="tab:green",
    s=45,
    label="Observed capacity"
)

ax3.axvline(
    REGIME_CHANGE_DAY,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Aggressive regime begins"
)

ax3.set_title("Observed Battery Capacity")
ax3.set_xlabel("Day")
ax3.set_ylabel("Capacity (fraction of nameplate)")
ax3.grid(False)
ax3.legend()

fig3.tight_layout()
fig3.savefig(
    FIGURE_DIR / "battery_capacity.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()