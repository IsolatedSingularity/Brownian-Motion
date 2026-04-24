# %% Importing Modules
import os

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import levy_stable, linregress
from scipy.stats import norm as scipy_norm

# %% Tokyo Night Storm Theme
PALETTE = {
    "bg": "#1a1b26",
    "panel": "#24283b",
    "fg": "#c0caf5",
    "muted": "#a9b1d6",
    "subtle": "#565f89",
    "blue": "#7aa2f7",
    "cyan": "#7dcfff",
    "purple": "#bb9af7",
    "red": "#f7768e",
    "green": "#9ece6a",
    "yellow": "#e0af68",
    "orange": "#ff9e64",
}
CYCLE = [PALETTE[k] for k in ("blue", "cyan", "purple", "red", "green", "yellow", "orange")]


def applyTokyoNight():
    """Apply Tokyo Night Storm dark theme to all subsequent matplotlib figures."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["bg"],
            "savefig.facecolor": PALETTE["bg"],
            "axes.edgecolor": PALETTE["subtle"],
            "axes.labelcolor": PALETTE["fg"],
            "axes.titlecolor": PALETTE["fg"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "text.color": PALETTE["fg"],
            "grid.color": PALETTE["subtle"],
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "axes.prop_cycle": cycler(color=CYCLE),
            "legend.facecolor": PALETTE["panel"],
            "legend.edgecolor": PALETTE["subtle"],
            "legend.labelcolor": PALETTE["fg"],
            "font.family": "sans-serif",
            "font.size": 10,
        }
    )


# â”€â”€ Fractional Brownian Motion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class FractionalBrownianMotion:
    """
    Exact simulation of fractional Brownian motion (fBm) for a range of
    Hurst exponents H via the Davies-Harte spectral method.

    The fBm increment process (fractional Gaussian noise) has autocovariance:

        gamma(k) = 1/2 * (|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})

    Embedding gamma in a circulant matrix and diagonalizing via FFT gives an
    exact O(n log n) sampler. The resulting trajectory satisfies:

        E[|B_H(t) - B_H(s)|^2] = |t - s|^{2H}

    so the MSD scales as <Delta x^2(tau)> ~ tau^{2H}.
    H < 0.5  =>  subdiffusion (anti-persistent increments)
    H = 0.5  =>  standard Brownian motion
    H > 0.5  =>  superdiffusion (persistent increments)

    Parameters
    ----------
    nSteps         : number of time steps per trajectory
    nWalks         : number of independent walkers for ensemble averaging
    hurstExponents : list of H values to simulate
    rng            : numpy Generator (optional); defaults to seed-42 RNG
    """

    def __init__(self, nSteps=5000, nWalks=200, hurstExponents=None, rng=None):
        self.nSteps = nSteps
        self.nWalks = nWalks
        self.hurstExponents = hurstExponents if hurstExponents is not None else [0.3, 0.5, 0.7, 0.9]
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.trajectories = {}  # H -> ndarray (nWalks, nSteps+1)
        self.msd = {}  # H -> ndarray (maxLag,)
        self.lagTimes = None
        self.diffusionExponents = {}  # H -> float, fitted slope of log MSD vs log tau

    def _fractionalGaussianNoise(self, n, H, seed):
        """
        Davies-Harte method: embed fGn autocovariance in a circulant of size 2n,
        diagonalize via FFT, and draw the exact stationary Gaussian process.
        Returns an array of length n drawn from the fGn with Hurst exponent H.
        """
        localRng = np.random.default_rng(seed)
        k = np.arange(n)
        gamma = 0.5 * (
            np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H)
        )
        # Circulant embedding of size 2n:  [gamma_0, gamma_1, ..., gamma_{n-1}, gamma_{n-1}, ..., gamma_1]
        row = np.concatenate([gamma, gamma[n - 1 : 0 : -1]])
        m = len(row)  # should be 2n
        eigVals = np.real(np.fft.fft(row))
        eigVals = np.maximum(eigVals, 0.0)  # numerical floor
        phi = localRng.standard_normal(m) + 1j * localRng.standard_normal(m)
        fgnFull = np.real(np.fft.ifft(np.sqrt(eigVals) * phi)) * np.sqrt(m)
        return fgnFull[:n]

    def generate(self):
        """Generate fBm trajectories for all configured Hurst exponents."""
        print("Computing fBm trajectories...")
        for H in self.hurstExponents:
            traj = np.zeros((self.nWalks, self.nSteps + 1))
            for walkIdx in range(self.nWalks):
                noise = self._fractionalGaussianNoise(self.nSteps, H, seed=walkIdx)
                traj[walkIdx, 1:] = np.cumsum(noise)
            self.trajectories[H] = traj
        return self

    def computeMSD(self, maxLag=500):
        """
        Compute the ensemble-averaged mean squared displacement for each H:

            MSD(tau) = < [x(t + tau) - x(t)]^2 >_{t, walkers}

        Expected scaling: MSD(tau) ~ tau^{2H}.
        """
        print("Computing MSD...")
        self.lagTimes = np.arange(1, maxLag + 1)
        for H in self.hurstExponents:
            traj = self.trajectories[H]
            self.msd[H] = np.array(
                [np.mean((traj[:, lag:] - traj[:, :-lag]) ** 2) for lag in self.lagTimes]
            )
        return self

    def fitExponents(self):
        """
        Fit MSD ~ tau^alpha_diff by log-linear regression.
        Theoretical prediction: alpha_diff = 2H.
        """
        for H in self.hurstExponents:
            slope, _, _, _, _ = linregress(np.log(self.lagTimes), np.log(self.msd[H]))
            self.diffusionExponents[H] = slope
        return self


# â”€â”€ Levy Flight â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class LevyFlight:
    """
    2D isotropic Levy flights with stability index alpha.

    Step sizes are drawn from a symmetric alpha-stable distribution:

        P(|dx|) ~ |dx|^{-(1+alpha)},  alpha in (0, 2)

    For alpha = 2 the distribution is Gaussian (normal diffusion). For alpha < 2
    the variance diverges: the characteristic function is phi(k) = exp(-|k|^alpha),
    and the central limit theorem is replaced by the generalized stable-law limit.
    The trajectory exhibits long ballistic bursts separated by local clustering.

    Parameters
    ----------
    nSteps     : number of steps per trajectory
    nWalks     : number of independent walkers
    alphaValues: list of stability indices to simulate
    rng        : numpy Generator (optional)
    """

    def __init__(self, nSteps=5000, nWalks=200, alphaValues=None, rng=None):
        self.nSteps = nSteps
        self.nWalks = nWalks
        self.alphaValues = alphaValues if alphaValues is not None else [1.2, 1.6, 2.0]
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.trajectories = {}  # alpha -> ndarray (nWalks, nSteps+1, 2)

    def generate(self):
        """Generate 2D Levy flight trajectories for all configured alpha values."""
        print("Computing Levy flight trajectories...")
        for alpha in self.alphaValues:
            traj2D = np.zeros((self.nWalks, self.nSteps + 1, 2))
            for walkIdx in range(self.nWalks):
                if alpha < 2.0:
                    # Isotropic 2D: random direction + Levy-stable magnitude
                    angles = self.rng.uniform(0, 2 * np.pi, self.nSteps)
                    magnitudes = np.abs(
                        levy_stable.rvs(
                            alpha=alpha,
                            beta=0,
                            size=self.nSteps,
                            random_state=int(walkIdx * 13 + 7),
                        )
                    )
                    stepsX = magnitudes * np.cos(angles)
                    stepsY = magnitudes * np.sin(angles)
                else:
                    stepsX = self.rng.standard_normal(self.nSteps)
                    stepsY = self.rng.standard_normal(self.nSteps)
                traj2D[walkIdx, 1:, 0] = np.cumsum(stepsX)
                traj2D[walkIdx, 1:, 1] = np.cumsum(stepsY)
            self.trajectories[alpha] = traj2D
        return self


# â”€â”€ Ornstein-Uhlenbeck Process â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck (OU) mean-reverting diffusion process.

    Solves the Langevin SDE via Euler-Maruyama discretization:

        dX = -theta * (X - mu) * dt + sigma * dW

    The stationary distribution is Gaussian:

        X_infty ~ N(mu, sigma^2 / (2 theta))

    The transient variance relaxes as:

        Var(t) = (sigma^2 / 2*theta) * (1 - exp(-2*theta*t))

    Unlike free Brownian motion the MSD saturates, making OU the canonical
    model for mean-reverting fluctuations (velocity in a viscous fluid,
    interest rates, thermal noise in a harmonic trap).

    Parameters
    ----------
    nSteps : number of time steps per trajectory
    nWalks : number of independent walkers
    mu     : long-term mean (drift target)
    theta  : mean-reversion rate (>0)
    sigma  : noise amplitude
    dt     : time step
    rng    : numpy Generator (optional)
    """

    def __init__(self, nSteps=5000, nWalks=200, mu=0.0, theta=0.05, sigma=1.0, dt=1.0, rng=None):
        self.nSteps = nSteps
        self.nWalks = nWalks
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.trajectories = None  # ndarray (nWalks, nSteps+1)

    def stationaryVariance(self):
        """Analytical long-time variance: sigma^2 / (2 * theta)."""
        return self.sigma**2 / (2 * self.theta)

    def analyticalVariance(self, t):
        """
        Transient variance from time 0:
            Var(t) = (sigma^2 / 2*theta) * (1 - exp(-2*theta*t))
        """
        return self.stationaryVariance() * (1.0 - np.exp(-2 * self.theta * t))

    def generate(self):
        """Simulate OU trajectories for all walkers via Euler-Maruyama."""
        print("Computing Ornstein-Uhlenbeck process...")
        traj = np.zeros((self.nWalks, self.nSteps + 1))
        sqrtDt = np.sqrt(self.dt)
        for walkIdx in range(self.nWalks):
            x = 0.0
            path = [x]
            noise = self.rng.standard_normal(self.nSteps) * sqrtDt
            for i in range(self.nSteps):
                x = x - self.theta * (x - self.mu) * self.dt + self.sigma * noise[i]
                path.append(x)
            traj[walkIdx] = path
        self.trajectories = traj
        return self


# â”€â”€ Visualizer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class DiffusionVisualizer:
    """
    Produces all four diagnostic figures for the anomalous diffusion suite.

    Figures
    -------
    1. fBm trajectories + MSD power-law scaling
    2. 2D Levy flight paths (three alpha values)
    3. OU sample paths, stationary distribution, variance relaxation
    4. Unified log-log MSD comparison across regimes
    """

    def __init__(self, palette=None, outputDir="Plots"):
        self.palette = palette if palette is not None else PALETTE
        self.outputDir = outputDir
        os.makedirs(outputDir, exist_ok=True)

    def _savePath(self, filename):
        return os.path.join(self.outputDir, filename)

    def plotFBm(self, fbm):
        """Figure 1: fBm sample trajectories and MSD power-law scaling."""
        p = self.palette
        colors = [p["blue"], p["green"], p["yellow"], p["red"]]
        tPlot = np.arange(fbm.nSteps + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Fractional Brownian Motion: Anomalous Diffusion", fontsize=15)

        ax = axes[0]
        ax.set_title("Trajectories for Different Hurst Exponents")
        for H, col in zip(fbm.hurstExponents, colors):
            alpha_d = fbm.diffusionExponents.get(H, 2 * H)
            ax.plot(
                tPlot,
                fbm.trajectories[H][0],
                color=col,
                linewidth=0.8,
                label=f"H = {H:.1f}  ($\\alpha_{{\\rm diff}} \\approx {alpha_d:.2f}$)",
            )
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Position")
        ax.legend(fontsize=9)
        ax.grid(True)

        ax = axes[1]
        ax.set_title(r"MSD Scaling  ($\langle\Delta x^2\rangle \sim \tau^{2H}$, log-log)")
        for H, col in zip(fbm.hurstExponents, colors):
            ax.loglog(fbm.lagTimes, fbm.msd[H], color=col, linewidth=2, label=f"H = {H:.1f}")
            prefac = fbm.msd[H][0] / fbm.lagTimes[0] ** (2 * H)
            ax.loglog(
                fbm.lagTimes,
                prefac * fbm.lagTimes ** (2 * H),
                color=col,
                linewidth=1,
                linestyle="--",
                alpha=0.5,
            )
        ax.set_xlabel(r"Lag Time $\tau$")
        ax.set_ylabel(r"MSD $\langle\Delta x^2(\tau)\rangle$")
        ax.legend(fontsize=9)
        ax.grid(True, which="both")

        fig.tight_layout()
        path = self._savePath("fBm_anomalous_diffusion.jpg")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  saved {path}")
        plt.close(fig)

    def plotLevy(self, levy):
        """Figure 2: 2D Levy flight paths coloured by step progression."""
        p = self.palette
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        fig.suptitle(r"L\'evy Flights in 2D: Heavy-Tailed Step Distributions", fontsize=15)
        levyColors = [p["cyan"], p["purple"], p["orange"]]

        for ax, alpha, col in zip(axes, levy.alphaValues, levyColors):
            traj = levy.trajectories[alpha]
            norm = Normalize(vmin=0, vmax=levy.nSteps)
            for w in range(5):
                x, y = traj[w, :, 0], traj[w, :, 1]
                for seg in range(0, levy.nSteps, 50):
                    segColor = plt.cm.plasma(norm(seg))
                    ax.plot(
                        x[seg : seg + 51],
                        y[seg : seg + 51],
                        color=segColor,
                        linewidth=0.6,
                        alpha=0.8,
                    )
                ax.plot(x[0], y[0], "o", color=p["green"], markersize=5, zorder=5)
                ax.plot(x[-1], y[-1], "x", color=p["red"], markersize=7, zorder=5)
            ax.set_title(f"$\\alpha = {alpha:.1f}$", fontsize=13)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True)

        sm = ScalarMappable(cmap="plasma", norm=Normalize(0, levy.nSteps))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.04)
        cbar.set_label("Step")
        cbar.ax.yaxis.set_tick_params(color=p["muted"])

        path = self._savePath("levy_flights_2d.jpg")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  saved {path}")

        # Hero: full 3-panel figure without colorbar
        cbar.remove()
        heroPath = self._savePath("levy_flights_hero.jpg")
        fig.savefig(heroPath, bbox_inches="tight", dpi=300)
        print(f"  saved {heroPath}")

        plt.close(fig)

    def plotOU(self, ou):
        """Figure 3: OU sample paths, stationary distribution, variance relaxation."""
        p = self.palette
        tArr = np.arange(ou.nSteps + 1) * ou.dt
        varInf = ou.stationaryVariance()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Ornstein-Uhlenbeck Process: Mean-Reverting Diffusion", fontsize=15)

        ax = axes[0]
        ax.set_title("Sample Trajectories")
        for w in range(8):
            ax.plot(tArr, ou.trajectories[w], linewidth=0.7, alpha=0.8)
        ax.axhline(ou.mu, color=p["red"], linewidth=1.5, linestyle="--", label=f"$\\mu = {ou.mu}$")
        ax.fill_between(
            tArr,
            ou.mu - varInf**0.5,
            ou.mu + varInf**0.5,
            alpha=0.15,
            color=p["yellow"],
            label=r"$\pm\sigma_\infty$",
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("X(t)")
        ax.legend(fontsize=9)
        ax.grid(True)

        ax = axes[1]
        ax.set_title("Stationary Distribution")
        finalVals = ou.trajectories[:, -1]
        ax.hist(finalVals, bins=30, density=True, color=p["blue"], alpha=0.7, label="Simulated")
        xG = np.linspace(finalVals.min() - 0.5, finalVals.max() + 0.5, 300)
        ax.plot(
            xG,
            scipy_norm.pdf(xG, ou.mu, varInf**0.5),
            color=p["orange"],
            linewidth=2.5,
            label=f"$\\mathcal{{N}}(\\mu,\\,\\sigma^2={varInf:.1f})$",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True)

        ax = axes[2]
        ax.set_title("Variance Relaxation to Stationary Value")
        varSim = np.var(ou.trajectories, axis=0)
        ax.plot(tArr, varSim, color=p["cyan"], linewidth=2, label="Simulated Var(t)")
        ax.plot(
            tArr,
            ou.analyticalVariance(tArr),
            color=p["yellow"],
            linewidth=2,
            linestyle="--",
            label="Analytical Var(t)",
        )
        ax.axhline(
            varInf,
            color=p["red"],
            linewidth=1,
            linestyle=":",
            label=f"$\\sigma^2_\\infty = {varInf:.1f}$",
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Variance")
        ax.legend(fontsize=9)
        ax.grid(True)

        fig.tight_layout()
        path = self._savePath("ornstein_uhlenbeck.jpg")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  saved {path}")
        plt.close(fig)

    def plotMSDComparison(self, fbm, levy, ou):
        """Figure 4: Unified log-log MSD comparison across all diffusion regimes."""
        p = self.palette
        colors = [p["blue"], p["green"], p["yellow"], p["red"]]
        lagTimes = fbm.lagTimes

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Mean Squared Displacement: Diffusion Regimes", fontsize=15)

        for H, col in zip(fbm.hurstExponents, colors):
            ax.loglog(
                lagTimes,
                fbm.msd[H],
                color=col,
                linewidth=2.5,
                label=f"fBm H={H:.1f} ($\\alpha={2 * H:.1f}$)",
            )

        levyMSD = np.array(
            [
                np.mean((levy.trajectories[2.0][:, lag, 0] - levy.trajectories[2.0][:, 0, 0]) ** 2)
                for lag in lagTimes
            ]
        )
        ax.loglog(
            lagTimes,
            levyMSD,
            color=p["orange"],
            linewidth=2,
            linestyle="-.",
            label=r"L\'evy $\alpha=2.0$ (Gaussian)",
        )

        ouMSD = np.array(
            [np.mean((ou.trajectories[:, lag] - ou.trajectories[:, 0]) ** 2) for lag in lagTimes]
        )
        ax.loglog(
            lagTimes,
            ouMSD,
            color=p["purple"],
            linewidth=2,
            linestyle="--",
            label="Ornstein-Uhlenbeck (bounded)",
        )

        ax.loglog(
            lagTimes.astype(float),
            2.0 * lagTimes.astype(float),
            color=p["subtle"],
            linewidth=1.2,
            linestyle=":",
            alpha=0.6,
            label="Normal diffusion (slope = 1)",
        )

        ax.set_xlabel(r"Lag Time $\tau$")
        ax.set_ylabel(r"MSD $\langle\Delta x^2\rangle$")
        ax.set_title(r"MSD $\sim t^\alpha$  (log-log)", fontsize=12)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, which="both")

        fig.tight_layout()
        path = self._savePath("msd_comparison.jpg")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  saved {path}")
        plt.close(fig)


# â”€â”€ Entry Point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    applyTokyoNight()

    scriptDir = os.path.dirname(os.path.abspath(__file__))
    plotsDir = os.path.join(scriptDir, "Plots")

    fbm = FractionalBrownianMotion(
        nSteps=5000,
        nWalks=200,
        hurstExponents=[0.3, 0.5, 0.7, 0.9],
        rng=np.random.default_rng(42),
    )
    fbm.generate().computeMSD(maxLag=500).fitExponents()

    levy = LevyFlight(
        nSteps=5000,
        nWalks=200,
        alphaValues=[1.2, 1.6, 2.0],
        rng=np.random.default_rng(43),
    )
    levy.generate()

    ou = OrnsteinUhlenbeck(
        nSteps=5000,
        nWalks=200,
        mu=0.0,
        theta=0.05,
        sigma=1.0,
        dt=1.0,
        rng=np.random.default_rng(44),
    )
    ou.generate()

    viz = DiffusionVisualizer(palette=PALETTE, outputDir=plotsDir)
    print("Plotting Figure 1: fBm and anomalous diffusion...")
    viz.plotFBm(fbm)
    print("Plotting Figure 2: Levy flights...")
    viz.plotLevy(levy)
    print("Plotting Figure 3: Ornstein-Uhlenbeck process...")
    viz.plotOU(ou)
    print("Plotting Figure 4: MSD comparison...")
    viz.plotMSDComparison(fbm, levy, ou)

    print("Done.")
