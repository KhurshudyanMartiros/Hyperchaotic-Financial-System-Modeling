from pynamicalsys import ContinuousDynamicalSystem as cds

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pynamicalsys import PlotStyler

from numba import njit

# =====================================================
# Function to create and save single projection
# =====================================================
def single_projection(X, Y, xlabel, ylabel, filename):
    # Apply the plot style
    ps = PlotStyler(fontsize=12, linewidth=0.5)
    ps.apply_style()
    fig = plt.figure(figsize=(6, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X, Y, "k-")#linewidth=0.5, color='blue')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close(fig)
    
# =====================================================
# Function to create and save 3D trajectories
# =====================================================
def trajectory_3D(X, Y, Z, xlabel, ylabel, zlabel, filename):
    # Set the plot style
    ps = PlotStyler(fontsize=12, linewidth=0.5)
    ps.apply_style()
    fig = plt.figure(figsize=(6, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Plot the trajectory
    ax.plot(X, Y, Z, "k-")

    # Set the labels and view angle
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.grid(False)
    plt.tight_layout()
    ax.view_init(elev=30, azim=-140)
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close(fig)

def lypunov_exponents_plot(lyapunov_exponents, filename):
    # =====================================================
    # Lyapunov Exponents – Convergence + Boxed Spectrum
    # =====================================================

    ps = PlotStyler(fontsize=12, linewidth=0.5)
    ps.apply_style()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    colors = ["royalblue", "navy", "black", "midnightblue"]

    # Plot convergence curves
    for i in range(4):
        ax.plot(
            lyapunov_exponents[:, 0],        # time
            lyapunov_exponents[:, i + 1],    # λ_i
            color=colors[i],
            linewidth=0.5
        )

    # Zero reference line
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Axis labels
    ax.set_xlabel("t")
    ax.set_ylabel("Lyapunov exponents")

    ax.set_xlim(transient_time, total_time)

    # -----------------------------------------------------
    # Compute final converged values
    # -----------------------------------------------------
    lambda_final = lyapunov_exponents[-1, 1:5]

    # Build legend text manually
    legend_labels = [
        rf"$\lambda_1={lambda_final[0]:.6f}$",
        rf"$\lambda_2={lambda_final[1]:.6f}$",
        rf"$\lambda_3={lambda_final[2]:.6f}$",
        rf"$\lambda_4={lambda_final[3]:.4f}$",
    ]

    # Create dummy lines for legend
    for i in range(4):
        ax.plot([], [], color=colors[i], label=legend_labels[i])

    # Boxed legend (like your image)
    leg = ax.legend(
        loc="lower left",
        frameon=True,
        fontsize=10,
        handlelength=0.5,
    )

    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close(fig)

    
# Define your 4D, 4-parameter system
@njit
def chaotic_4d_system(time, state, params):
    a, b, c, d, k = params
    x, y, z, u = state
    # Replace these with your actual governing equations
    dxdt = z + (y - a) * x + u
    dydt = 1.0 - b * y - x * x
    dzdt = -x - c * z
    dudt = -d * x * y - k * u
    return np.array([dxdt, dydt, dzdt, dudt])

@njit
def jacobian(time, state, params):
    a, b, c, d, k = params
    x, y, z, u = state

    return np.array([
        [y - a,      x,     1.0,  1.0],
        [-2.0*x,    -b,     0.0,  0.0],
        [-1.0,       0.0,  -c,    0.0],
        [-d*y,     -d*x,    0.0, -k]
    ])
# Initialize the system
# params: list of the 4 parameters [a, b, c, d]
# dim: dimension of the system (4)

a, b, c, d, k = 0.9, 0.2, 1.5, 0.2, 0.17
parameters=[a, b, c, d, k]
ds = cds(equations_of_motion=chaotic_4d_system, jacobian=jacobian, system_dimension=4, parameters=parameters)

ds.integrator("rk4", time_step=0.001)

# Initial condition [x0, y0, z0, w0]
initial_state = [1.0, 2.0, 0.5, 0.5]

total_time = 1000
transient_time = 0

trajectory = ds.trajectory(initial_state, total_time, transient_time=transient_time)
print(trajectory.shape)


# =====================================================
# Generate and save all projections
# =====================================================

single_projection(trajectory[:, 1], trajectory[:, 3], r"$x$", r"$z$", "projection_xz_pynamicalsys.pdf")
single_projection(trajectory[:, 2], trajectory[:, 3], r"$y$", r"$z$", "projection_yz_pynamicalsys.pdf")
single_projection(trajectory[:, 1], trajectory[:, 2], r"$x$", r"$y$", "projection_xy_pynamicalsys.pdf")
single_projection(trajectory[:, 3], trajectory[:, 4], r"$z$", r"$u$", "projection_zu_pynamicalsys.pdf")
single_projection(trajectory[:, 2], trajectory[:, 4], r"$y$", r"$u$", "projection_yu_pynamicalsys.pdf")
single_projection(trajectory[:, 1], trajectory[:, 4], r"$x$", r"$u$", "projection_xu_pynamicalsys.pdf")

print("All projection plots generated and saved successfully.")


trajectory_3D(trajectory[:, 1], trajectory[:, 2], trajectory[:, 3], r"$x$", r"$y$", r"$z$", "trajectory_xyz_pynamicalsys.pdf")

print("All 3D trajectory generated and saved successfully.")


lyapunov_exponents = ds.lyapunov(
    initial_state,
    total_time,
    transient_time=transient_time,
    return_history=True,
    #log_base=2,
)

lypunov_exponents_plot(lyapunov_exponents, "Lypunov_exponents_pynamicalsys.pdf")


#print(lyapunov_exponents)

