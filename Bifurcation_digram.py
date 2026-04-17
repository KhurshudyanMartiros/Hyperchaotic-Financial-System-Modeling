import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from numba import njit

# =====================================================
# Global Plot Style (Publication Quality)
# =====================================================
plt.rcParams.update({
    "text.usetex": False,           # Set True if LaTeX is installed
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.0,
})

# =====================================================
# Continuous Financial System
# =====================================================

@njit
def finance_rhs(state, a, b, c, d, k):
    x, y, z, u = state

    dx = z + (y - a) * x + u
    dy = 1.0 - b * y - x * x
    dz = -x - c * z
    du = -d * x * y - k * u

    return np.array([dx, dy, dz, du])


# =====================================================
# RK4 Integrator
# =====================================================

@njit
def rk4_step(state, dt, a, b, c, d, k):
    k1 = finance_rhs(state, a, b, c, d, k)
    k2 = finance_rhs(state + 0.5 * dt * k1, a, b, c, d, k)
    k3 = finance_rhs(state + 0.5 * dt * k2, a, b, c, d, k)
    k4 = finance_rhs(state + dt * k3, a, b, c, d, k)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# =====================================================
# Parameters
# =====================================================

a = 0.9
b = 0.2
d = 0.2
k = 0.17

c_values = np.linspace(0.0, 2.0, 800)

dt = 0.01
total_steps = 50000
transient_steps = 12500

initial_state = np.array([1.0, 2.0, 0.5, 0.5])


# =====================================================
# Bifurcation Computation (Maxima-based)
# =====================================================

bifurcation_c = []
bifurcation_xmax = []

for c in c_values:

    state = initial_state.copy()

    x_series = np.zeros(total_steps)

    # Integrate
    for i in range(total_steps):
        state = rk4_step(state, dt, a, b, c, d, k)
        x_series[i] = state[0]

    # Remove transient
    x_series = x_series[transient_steps:]

    # Detect local maxima
    peaks, _ = find_peaks(x_series)

    maxima = x_series[peaks]

    # Store results
    bifurcation_xmax.extend(maxima)
    bifurcation_c.extend([c] * len(maxima))


# =====================================================
# Create Figure
# =====================================================
fig, ax = plt.subplots(figsize=(7, 5))

# Scatter plot
ax.scatter(
    bifurcation_c,
    bifurcation_xmax,
    s=0.15,               # very small for dense bifurcation diagrams
    color="black",
    marker=".",
    rasterized=True       # keeps vector PDF small
)

# Axis limits
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-2, 3)

# Labels (LaTeX-style math)
ax.set_xlabel(r"$c$")
ax.set_ylabel(r"$x_{\max}$")

# Clean journal-style look
#ax.spines["top"].set_visible(False)
#ax.spines["right"].set_visible(False)

ax.tick_params(direction="in", length=6, width=1)

# Tight layout
fig.tight_layout()

# =====================================================
# Save Figure (High Quality)
# =====================================================
fig.savefig("bifurcation_diagram.pdf", dpi=600, bbox_inches="tight")
fig.savefig("bifurcation_diagram.png", dpi=600, bbox_inches="tight")

plt.show()
