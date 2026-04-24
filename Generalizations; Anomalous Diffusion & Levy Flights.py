#%% Importing Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import levy_stable
from cycler import cycler


#%% Tokyo Night Storm Theme
PALETTE = {
    "bg":     "#1a1b26",
    "panel":  "#24283b",
    "fg":     "#c0caf5",
    "muted":  "#a9b1d6",
    "subtle": "#565f89",
    "blue":   "#7aa2f7",
    "cyan":   "#7dcfff",
    "purple": "#bb9af7",
    "red":    "#f7768e",
    "green":  "#9ece6a",
    "yellow": "#e0af68",
    "orange": "#ff9e64",
}
CYCLE = [PALETTE[k] for k in ("blue", "cyan", "purple", "red", "green", "yellow", "orange")]

def applyTokyoNight():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor":   PALETTE["bg"],
        "axes.facecolor":     PALETTE["bg"],
        "savefig.facecolor":  PALETTE["bg"],
        "axes.edgecolor":     PALETTE["subtle"],
        "axes.labelcolor":    PALETTE["fg"],
        "axes.titlecolor":    PALETTE["fg"],
        "xtick.color":        PALETTE["muted"],
        "ytick.color":        PALETTE["muted"],
        "text.color":         PALETTE["fg"],
        "grid.color":         PALETTE["subtle"],
        "grid.linestyle":     "--",
        "grid.alpha":         0.4,
        "axes.prop_cycle":    cycler(color=CYCLE),
        "legend.facecolor":   PALETTE["panel"],
        "legend.edgecolor":   PALETTE["subtle"],
        "legend.labelcolor":  PALETTE["fg"],
        "font.family":        "sans-serif",
        "font.size":          10,
    })

applyTokyoNight()


#%% Physical Setup & Simulation Parameters
rng = np.random.default_rng(42)

nSteps   = 5000   # number of steps per trajectory
nWalks   = 200    # number of independent walkers for ensemble averaging
dt       = 1.0    # time step

hurstExponents = [0.3, 0.5, 0.7, 0.9]   # H < 0.5: subdiffusion, H = 0.5: normal, H > 0.5: superdiffusion
levyAlpha      = [1.2, 1.6, 2.0]         # Levy stability index: alpha=2 -> Gaussian, alpha<2 -> heavy-tailed
ouMean         = 0.0                     # Ornstein-Uhlenbeck long-term mean
ouTheta        = 0.05                    # OU mean-reversion rate
ouSigma        = 1.0                     # OU noise amplitude


#%% Helper: Fractional Gaussian Noise via Davies-Harte Method
def fractionalGaussianNoise(n, H, seed=None):
    """
    Generate fractional Gaussian noise with Hurst exponent H using the
    Davies-Harte (exact spectral) method. Returns an array of length n.
    H = 0.5 recovers standard Gaussian white noise.
    """
    localRng = np.random.default_rng(seed)
    # Build autocovariance of fGn
    k = np.arange(n)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    # Embed in circulant of size 2n
    row = np.concatenate([gamma, gamma[-2:0:-1]])
    eigVals = np.real(np.fft.fft(row))
    eigVals = np.maximum(eigVals, 0)  # numerical floor
    phi = localRng.standard_normal(2 * n) + 1j * localRng.standard_normal(2 * n)
    fgnFull = np.real(np.fft.ifft(np.sqrt(eigVals) * phi))
    return fgnFull[:n]


#%% Generating Fractional Brownian Motion Trajectories
print("Computing fBm trajectories...")
fBmTrajectories = {}   # H -> array (nWalks, nSteps+1)

for H in hurstExponents:
    traj = np.zeros((nWalks, nSteps + 1))
    for walkIdx in range(nWalks):
        noise = fractionalGaussianNoise(nSteps, H, seed=walkIdx)
        traj[walkIdx, 1:] = np.cumsum(noise)
    fBmTrajectories[H] = traj


#%% Computing MSD for fBm
print("Computing MSD...")
maxLag = 500
lagTimes = np.arange(1, maxLag + 1)
fBmMSD = {}   # H -> msd array

for H in hurstExponents:
    traj = fBmTrajectories[H]
    msdArr = np.array([
        np.mean((traj[:, lag:] - traj[:, :-lag]) ** 2)
        for lag in lagTimes
    ])
    fBmMSD[H] = msdArr


#%% Generating Levy Flight Trajectories (2D)
print("Computing Levy flight trajectories...")
levyTrajectories = {}   # alpha -> array (nWalks, nSteps+1, 2)

for alpha in levyAlpha:
    traj2D = np.zeros((nWalks, nSteps + 1, 2))
    for walkIdx in range(nWalks):
        if alpha < 2.0:
            # Levy-stable steps: isotropic 2D drawn via angle + magnitude
            angles = rng.uniform(0, 2 * np.pi, nSteps)
            magnitudes = levy_stable.rvs(alpha=alpha, beta=0, size=nSteps,
                                         random_state=int(walkIdx * 13 + 7))
            magnitudes = np.abs(magnitudes)  # isotropic
            stepsX = magnitudes * np.cos(angles)
            stepsY = magnitudes * np.sin(angles)
        else:
            # alpha=2: standard Gaussian
            stepsX = rng.standard_normal(nSteps)
            stepsY = rng.standard_normal(nSteps)
        traj2D[walkIdx, 1:, 0] = np.cumsum(stepsX)
        traj2D[walkIdx, 1:, 1] = np.cumsum(stepsY)
    levyTrajectories[alpha] = traj2D


#%% Generating Ornstein-Uhlenbeck Process
print("Computing Ornstein-Uhlenbeck process...")
# Exact Euler-Maruyama discretisation: dX = -theta*(X - mu)*dt + sigma*dW
ouTrajectories = np.zeros((nWalks, nSteps + 1))
for walkIdx in range(nWalks):
    x = 0.0
    traj = [x]
    noise = rng.standard_normal(nSteps) * np.sqrt(dt)
    for i in range(nSteps):
        x = x - ouTheta * (x - ouMean) * dt + ouSigma * noise[i]
        traj.append(x)
    ouTrajectories[walkIdx] = traj

# OU long-time variance (analytical): sigma^2 / (2*theta)
ouVarianceAnalytic = ouSigma ** 2 / (2 * ouTheta)


#%% Fitting Anomalous Diffusion Exponents  alpha_diff from MSD ~ t^alpha_diff
from scipy.stats import linregress

diffusionExponents = {}
for H in hurstExponents:
    logT = np.log(lagTimes)
    logMSD = np.log(fBmMSD[H])
    slope, intercept, _, _, _ = linregress(logT, logMSD)
    diffusionExponents[H] = slope   # expect ~ 2H


#%% Plotting: Figure 1 - fBm Trajectories & MSD Scaling
print("Plotting Figure 1: fBm and anomalous diffusion...")
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle("Fractional Brownian Motion: Anomalous Diffusion", fontsize=15, color=PALETTE["fg"])

# Left: sample trajectories for each H
ax = axes1[0]
ax.set_title("Trajectories for Different Hurst Exponents", color=PALETTE["fg"])
tPlot = np.arange(nSteps + 1)
colors = [PALETTE["blue"], PALETTE["green"], PALETTE["yellow"], PALETTE["red"]]
labels = [f"H = {H:.1f}  (α_diff ≈ {diffusionExponents[H]:.2f})" for H in hurstExponents]
for H, col, lbl in zip(hurstExponents, colors, labels):
    ax.plot(tPlot, fBmTrajectories[H][0], color=col, linewidth=0.8, label=lbl)
ax.set_xlabel("Time Step", color=PALETTE["fg"])
ax.set_ylabel("Position", color=PALETTE["fg"])
ax.legend(fontsize=9)
ax.grid(True)

# Right: MSD scaling
ax = axes1[1]
ax.set_title("Mean Squared Displacement Scaling  (MSD ~ t^{2H})", color=PALETTE["fg"])
for H, col in zip(hurstExponents, colors):
    ax.loglog(lagTimes, fBmMSD[H], color=col, linewidth=2, label=f"H = {H:.1f}")
    # Theoretical line MSD ~ t^(2H)
    prefac = fBmMSD[H][0] / lagTimes[0] ** (2 * H)
    ax.loglog(lagTimes, prefac * lagTimes ** (2 * H), color=col, linewidth=1,
              linestyle="--", alpha=0.5)
ax.set_xlabel("Lag Time τ", color=PALETTE["fg"])
ax.set_ylabel("MSD  ⟨Δx²(τ)⟩", color=PALETTE["fg"])
ax.legend(fontsize=9)
ax.grid(True, which="both")

fig1.tight_layout()
fig1.savefig("Plots/fBm_anomalous_diffusion.jpg", bbox_inches="tight", dpi=300)
plt.show()


#%% Plotting: Figure 2 - Levy Flights in 2D
print("Plotting Figure 2: Levy flights...")
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle("Lévy Flights in 2D: Heavy-Tailed Step Distributions", fontsize=15, color=PALETTE["fg"])

levyColors = [PALETTE["cyan"], PALETTE["purple"], PALETTE["orange"]]
levyLabels = [f"α = {a:.1f}" for a in levyAlpha]

for ax, alpha, col, lbl in zip(axes2, levyAlpha, levyColors, levyLabels):
    # Plot 5 sample trajectories
    traj = levyTrajectories[alpha]
    for w in range(5):
        x = traj[w, :, 0]
        y = traj[w, :, 1]
        # Color trajectory by step progression
        norm = Normalize(vmin=0, vmax=nSteps)
        for seg in range(0, nSteps, 50):
            segColor = plt.cm.plasma(norm(seg))
            ax.plot(x[seg:seg+51], y[seg:seg+51], color=segColor, linewidth=0.6, alpha=0.8)
        ax.plot(x[0], y[0], "o", color=PALETTE["green"], markersize=5, zorder=5)
        ax.plot(x[-1], y[-1], "x", color=PALETTE["red"], markersize=7, zorder=5)
    ax.set_title(lbl, color=PALETTE["fg"], fontsize=13)
    ax.set_xlabel("x", color=PALETTE["fg"])
    ax.set_ylabel("y", color=PALETTE["fg"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

# Shared colorbar for step progression
sm = ScalarMappable(cmap="plasma", norm=Normalize(0, nSteps))
sm.set_array([])
cbar = fig2.colorbar(sm, ax=axes2.ravel().tolist(), fraction=0.015, pad=0.04)
cbar.set_label("Step", color=PALETTE["fg"])
cbar.ax.yaxis.set_tick_params(color=PALETTE["muted"])

fig2.tight_layout()
fig2.savefig("Plots/levy_flights_2d.jpg", bbox_inches="tight", dpi=300)
plt.show()


#%% Plotting: Figure 3 - Ornstein-Uhlenbeck: Bounded Diffusion & Stationary Distribution
print("Plotting Figure 3: Ornstein-Uhlenbeck process...")
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
fig3.suptitle("Ornstein-Uhlenbeck Process: Mean-Reverting Diffusion", fontsize=15, color=PALETTE["fg"])

# Left: sample trajectories
ax = axes3[0]
ax.set_title("Sample Trajectories", color=PALETTE["fg"])
tArr = np.arange(nSteps + 1) * dt
for w in range(8):
    ax.plot(tArr, ouTrajectories[w], linewidth=0.7, alpha=0.8)
ax.axhline(ouMean, color=PALETTE["red"], linewidth=1.5, linestyle="--", label=f"μ = {ouMean}")
ax.fill_between(tArr,
                ouMean - np.sqrt(ouVarianceAnalytic),
                ouMean + np.sqrt(ouVarianceAnalytic),
                alpha=0.15, color=PALETTE["yellow"], label="±σ∞")
ax.set_xlabel("Time", color=PALETTE["fg"])
ax.set_ylabel("X(t)", color=PALETTE["fg"])
ax.legend(fontsize=9)
ax.grid(True)

# Middle: stationary distribution vs. theoretical Gaussian
ax = axes3[1]
ax.set_title("Stationary Distribution", color=PALETTE["fg"])
finalVals = ouTrajectories[:, -1]
ax.hist(finalVals, bins=30, density=True, color=PALETTE["blue"], alpha=0.7, label="Simulated")
xGauss = np.linspace(finalVals.min() - 0.5, finalVals.max() + 0.5, 300)
from scipy.stats import norm as scipy_norm
ax.plot(xGauss, scipy_norm.pdf(xGauss, ouMean, np.sqrt(ouVarianceAnalytic)),
        color=PALETTE["orange"], linewidth=2.5, label=f"N(μ, σ²={ouVarianceAnalytic:.1f})")
ax.set_xlabel("X", color=PALETTE["fg"])
ax.set_ylabel("Density", color=PALETTE["fg"])
ax.legend(fontsize=9)
ax.grid(True)

# Right: OU variance growth vs time (transient)
ax = axes3[2]
ax.set_title("Variance Relaxation to Stationary Value", color=PALETTE["fg"])
varOverTime = np.var(ouTrajectories, axis=0)
ax.plot(tArr, varOverTime, color=PALETTE["cyan"], linewidth=2, label="Simulated Var(t)")
# Analytical: Var(t) = sigma^2/(2*theta) * (1 - exp(-2*theta*t))
varAnalyticTime = (ouSigma ** 2 / (2 * ouTheta)) * (1 - np.exp(-2 * ouTheta * tArr))
ax.plot(tArr, varAnalyticTime, color=PALETTE["yellow"], linewidth=2,
        linestyle="--", label="Analytical Var(t)")
ax.axhline(ouVarianceAnalytic, color=PALETTE["red"], linewidth=1, linestyle=":",
           label=f"σ²∞ = {ouVarianceAnalytic:.1f}")
ax.set_xlabel("Time", color=PALETTE["fg"])
ax.set_ylabel("Variance", color=PALETTE["fg"])
ax.legend(fontsize=9)
ax.grid(True)

fig3.tight_layout()
fig3.savefig("Plots/ornstein_uhlenbeck.jpg", bbox_inches="tight", dpi=300)
plt.show()


#%% Plotting: Figure 4 - Comparison: Normal vs Anomalous vs Levy MSD
print("Plotting Figure 4: MSD comparison...")
fig4, ax4 = plt.subplots(figsize=(10, 6))
fig4.suptitle("Mean Squared Displacement: Diffusion Regimes", fontsize=15, color=PALETTE["fg"])

ax4.set_title("MSD ~ t^α  (log-log)", color=PALETTE["fg"])

# fBm curves
for H, col, lbl in zip(hurstExponents, colors, labels):
    ax4.loglog(lagTimes, fBmMSD[H], color=col, linewidth=2.5, label=f"fBm H={H:.1f} (α={2*H:.1f})")

# Levy MSD (only alpha=2 has finite MSD; lighter alphas have infinite variance)
levyMSD_gaussian = np.array([
    np.mean((levyTrajectories[2.0][:, lag, 0] - levyTrajectories[2.0][:, 0, 0]) ** 2)
    for lag in lagTimes
])
ax4.loglog(lagTimes, levyMSD_gaussian, color=PALETTE["orange"], linewidth=2,
           linestyle="-.", label="Lévy α=2.0 (Gaussian)")

# OU MSD
ouMSD = np.array([
    np.mean((ouTrajectories[:, lag] - ouTrajectories[:, 0]) ** 2)
    for lag in lagTimes
])
ax4.loglog(lagTimes, ouMSD, color=PALETTE["purple"], linewidth=2,
           linestyle="--", label="Ornstein-Uhlenbeck (bounded)")

# Reference slopes
refT = lagTimes.astype(float)
ax4.loglog(refT, 2.0 * refT, color=PALETTE["subtle"], linewidth=1.2, linestyle=":",
           alpha=0.6, label="Normal diffusion (slope=1)")

ax4.set_xlabel("Lag Time τ", color=PALETTE["fg"])
ax4.set_ylabel("MSD  ⟨Δx²⟩", color=PALETTE["fg"])
ax4.legend(fontsize=9, loc="upper left")
ax4.grid(True, which="both")

fig4.tight_layout()
fig4.savefig("Plots/msd_comparison.jpg", bbox_inches="tight", dpi=300)
plt.show()

print("Done.")
