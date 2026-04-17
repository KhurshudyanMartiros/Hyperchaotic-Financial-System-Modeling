import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =====================================================
# Parameters
# =====================================================
a  = 0.9
b  = 0.2
c  = 1.5
v  = 0.2
k  = 0.17
k1 = 9.5


# =====================================================
# System Definition
# =====================================================
def financial_system(t, state):
    X, Y, Z, U = state
    
    # dX/dt
    dX = U + X * (-a + 1/b + Y) + Z
    
    # dY/dt
    dY = -X**2 - b * Y
    
    # dZ/dt
    dZ = -X - c * Z
    
    # dU/dt  (substituting dX)
    dU = -k * U - v * X * (1/b + Y) - k1 * dX
    
    return [dX, dY, dZ, dU]

# =====================================================
# Time Span
# =====================================================
t_span = (0, 100)
t_eval = np.linspace(0, 100, 1000)

# Initial conditions
y0 = [-1.5, -1, 0.5, 2.5]

# Solve
sol = solve_ivp(
    financial_system,
    t_span,
    y0,
    t_eval=t_eval,
    method='RK45'
)

# =====================================================
# Plot (Publication Style)
# =====================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.linewidth": 1.0,
})

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(sol.t, sol.y[0], label=r"$X(t)$")
ax.plot(sol.t, sol.y[1], label=r"$Y(t)$")
ax.plot(sol.t, sol.y[2], label=r"$Z(t)$")
ax.plot(sol.t, sol.y[3], label=r"$U(t)$")

ax.set_xlabel(r"$t$")
ax.set_ylabel("State variables")

ax.legend(frameon=False)
#ax.spines["top"].set_visible(False)
#ax.spines["right"].set_visible(False)
ax.tick_params(direction="in", length=5)

fig.tight_layout()

fig.savefig("financial_system_time_series.pdf", dpi=600, bbox_inches="tight")
plt.show()
