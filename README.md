# Brownian Motion

[![CI](https://img.shields.io/github/actions/workflow/status/IsolatedSingularity/Brownian-Motion/ci.yml?branch=main&label=CI&logo=github)](https://github.com/IsolatedSingularity/Brownian-Motion/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-darkblue.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/scipy-8CAAE6.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-11557C.svg)](https://matplotlib.org/)

<p align="center">
  <img src="Plots/levy_flights_hero.jpg" alt="Lévy flight in 2D" width="700">
</p>

Two complementary scripts exploring random walks and their generalizations, from the textbook 1D random walk to the menagerie of anomalous diffusion processes found in physics, biology, and finance.

---

## Files

### `Brownian Motion; Random Walks.py`

A 1D random walk with two step-size classes: 7 unit steps and 1 jump of size 3 per macro-step, repeated for 10,000 macro-steps. The simulation tracks the walker's position $x(t)$ and the cumulative mean-squared displacement, then fits two theoretical curves to $x_\text{RMS}(t)$:

- **Angela fit**: $x_\text{RMS} = \sqrt{2(D + \varepsilon)\, t}$, a diffusion constant shifted by $\varepsilon$
- **Donald fit**: $x_\text{RMS} = \sqrt{2D}\, t^{1/2 + \delta}$, an anomalous-diffusion power law

Goodness-of-fit is assessed with a $\chi^2$ test against each model. The resulting plot overlays the raw trajectory, the RMS curve, and both fits on a single panel.

### `Generalizations; Anomalous Diffusion & Levy Flights.py`

Extends the random-walk picture to four distinct diffusion regimes, all visualised with the Tokyo Night Storm dark palette:

| Process | Key parameter | Diffusion regime |
|---------|--------------|-----------------|
| Fractional Brownian motion (fBm) | Hurst exponent $H \in (0, 1)$ | Sub-diffusive ($H < 0.5$), normal ($H = 0.5$), super-diffusive ($H > 0.5$) |
| Lévy flights | Stability index $\alpha \in (0, 2]$ | Ballistic jumps, infinite variance for $\alpha < 2$ |
| Ornstein-Uhlenbeck | Mean-reversion rate $\theta$, noise $\sigma$ | Bounded; variance saturates at $\sigma^2 / 2\theta$ |
| MSD comparison | All of the above | Log-log MSD scaling $\langle \Delta x^2 \rangle \sim t^\alpha$ |

fBm trajectories are generated via the exact Davies-Harte spectral method. Lévy steps are drawn from a symmetric stable distribution using `scipy.stats.levy_stable`. The OU process uses exact Euler-Maruyama discretisation with analytical variance $\text{Var}(t) = \frac{\sigma^2}{2\theta}(1 - e^{-2\theta t})$ plotted alongside the simulated ensemble.

Output figures (saved to `Plots/`):

- `fBm_anomalous_diffusion.jpg`: trajectories + MSD power-law scaling for $H \in \{0.3, 0.5, 0.7, 0.9\}$
- `levy_flights_2d.jpg`: 2D Lévy flight paths colored by step progression, three stability indices
- `ornstein_uhlenbeck.jpg`: OU sample paths, stationary Gaussian distribution, variance relaxation
- `msd_comparison.jpg`: unified log-log MSD comparison across all regimes

<p align="center">
  <img src="Plots/fBm_anomalous_diffusion.jpg" alt="fBm trajectories and MSD scaling" width="900">
</p>

<p align="center">
  <img src="Plots/levy_flights_2d.jpg" alt="Lévy flights in 2D" width="900">
</p>

<p align="center">
  <img src="Plots/ornstein_uhlenbeck.jpg" alt="Ornstein-Uhlenbeck process" width="900">
</p>

<p align="center">
  <img src="Plots/msd_comparison.jpg" alt="MSD comparison across diffusion regimes" width="700">
</p>

---

## Theory

### From discrete random walks to the Wiener process

Consider a walker on the real line taking steps $\{\xi_i\}_{i=1}^{N}$ drawn i.i.d. from a distribution with mean zero, finite variance $\sigma^2$, and arbitrary higher moments. The displacement after $N$ steps is

$$X_N \;=\; \sum_{i=1}^{N} \xi_i,$$

with mean $\langle X_N \rangle = 0$ and variance $\langle X_N^2 \rangle = N\sigma^2$. The classical central limit theorem guarantees that the rescaled sum $X_N / \sqrt{N}$ converges weakly to a Gaussian, so in the diffusive scaling $t = N\Delta t,\; x = X_N\sqrt{\Delta t}$ the walker's position becomes a Wiener process $W(t)$ in the limit $N \to \infty,\, \Delta t \to 0$ with $D \equiv \sigma^2/(2\Delta t)$ held fixed. The probability density $\rho(x, t)$ satisfies the diffusion equation

$$\partial_t \rho \;=\; D\,\partial_x^2 \rho, \qquad \rho(x,0) = \delta(x),$$

whose fundamental solution is the heat kernel $\rho(x,t) = (4\pi D t)^{-1/2}\exp(-x^2/4Dt)$, yielding the canonical scaling $\langle x^2(t)\rangle = 2Dt$. The first script in this repository fits this law (and a power-law generalisation) to the simulated trajectory of a mixed-step walker; deviations of the fitted exponent from $1/2$ flag departures from the Gaussian universality class.

### Anomalous diffusion and fractional Brownian motion

The Gaussian universality class is fragile. Whenever step correlations decay too slowly, or step variance diverges, the linear-in-time MSD law breaks. We summarise the resulting regimes by the scaling

$$\bigl\langle [x(t) - x(0)]^2 \bigr\rangle \;\sim\; t^{\alpha}, \qquad \alpha \in (0, 2],$$

with $\alpha < 1$ subdiffusion (trapping, viscoelastic media), $\alpha = 1$ ordinary diffusion, and $\alpha > 1$ superdiffusion (long flights, active transport). A clean parametric family realising every $\alpha \in (0, 2)$ is fractional Brownian motion $B_H(t)$, the unique zero-mean Gaussian process with stationary increments and the covariance

$$\bigl\langle B_H(t)\, B_H(s)\bigr\rangle \;=\; \tfrac{1}{2}\bigl(|t|^{2H} + |s|^{2H} - |t - s|^{2H}\bigr), \qquad H \in (0, 1),$$

so that $\langle B_H(t)^2\rangle = t^{2H}$ and consequently $\alpha = 2H$. Differencing on a unit lattice produces fractional Gaussian noise with autocovariance

$$\gamma(k) \;=\; \tfrac{1}{2}\bigl(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H}\bigr),$$

negative for $H < 1/2$ (anti-persistent, locally reversing), zero for $H = 1/2$ (independent Wiener increments), and positive with the slow $k^{2H-2}$ tail for $H > 1/2$ (long-range dependence). We synthesise exact sample paths via the Davies-Harte circulant embedding: the symmetric Toeplitz covariance matrix is diagonalised by the discrete Fourier transform, allowing $\mathcal{O}(N \log N)$ generation of trajectories of length $N$ with the prescribed spectrum.

### Heavy-tailed steps and the generalised central limit theorem

Even when increments are independent, finite variance is essential to the Gaussian limit; relaxing it leads to the second universality class. If step sizes are drawn from a symmetric distribution with the asymptotic tail

$$P(|\xi| > x) \;\sim\; C\, x^{-\alpha}, \qquad 0 < \alpha < 2,$$

the variance diverges and the classical central limit theorem fails. The Gnedenko-Kolmogorov generalised central limit theorem guarantees instead that suitably normalised sums $X_N / N^{1/\alpha}$ converge to a symmetric $\alpha$-stable random variable $S_\alpha$ with characteristic function

$$\mathbb{E}\bigl[e^{ikS_\alpha}\bigr] \;=\; \exp(-c|k|^{\alpha}),$$

reducing to the Gaussian case at $\alpha = 2$ and to the Cauchy distribution at $\alpha = 1$. The corresponding continuous-time process is a Lévy flight: trajectories are dominated by rare large jumps, the spatial density develops algebraic tails $\rho(x, t) \sim t / |x|^{1+\alpha}$, and the MSD is formally infinite for $\alpha < 2$ even though pseudo-MSDs computed from finite samples appear to scale superdiffusively as $\sim t^{2/\alpha}$. The hero figure shows three trajectories at $\alpha = 2.0$, the boundary case where the stable law collapses back onto the Wiener process; smaller $\alpha$ values produce the characteristic clusters-and-jumps morphology that distinguishes Lévy transport from ordinary diffusion in foraging ecology, plasma turbulence, and disordered solids.

### Confined diffusion: the Ornstein-Uhlenbeck process

The two preceding regimes share unbounded growth of the position; many physical settings instead constrain the walker through a restoring force. The minimal model is the overdamped Langevin equation

$$dX_t \;=\; -\theta\,(X_t - \mu)\,dt \;+\; \sigma\, dW_t,$$

with mean-reversion rate $\theta > 0$, equilibrium $\mu$, and diffusion amplitude $\sigma$. The associated Fokker-Planck equation

$$\partial_t \rho \;=\; \theta\,\partial_x\bigl[(x - \mu)\rho\bigr] \;+\; \tfrac{\sigma^2}{2}\,\partial_x^2 \rho$$

admits the closed-form transition density (Mehler kernel)

$$\rho(x, t \,|\, x_0, 0) \;=\; \mathcal{N}\!\left(\mu + (x_0 - \mu) e^{-\theta t},\;\; \frac{\sigma^2}{2\theta}\bigl(1 - e^{-2\theta t}\bigr)\right),$$

whose stationary limit is the Boltzmann-Gibbs distribution $\mathcal{N}(\mu, \sigma^2/2\theta)$ for a harmonic potential $V(x) = \tfrac{1}{2}\theta(x - \mu)^2$ at temperature $k_B T = \sigma^2/2$ (fluctuation-dissipation). On timescales $t \ll \theta^{-1}$ the process is indistinguishable from free Brownian motion with diffusion constant $D = \sigma^2/2$; on timescales $t \gg \theta^{-1}$ the variance saturates at $\sigma^2/2\theta$ and the autocovariance decays as $\langle X_t X_0\rangle - \mu^2 = (\sigma^2/2\theta)\, e^{-\theta t}$. We integrate the SDE with the exact discretisation that samples directly from the Mehler kernel rather than the first-order Euler-Maruyama scheme, eliminating discretisation bias regardless of step size.

### Synthesis

The four processes simulated here trace the boundary of the Wiener universality class in two orthogonal directions: temporal correlations (fBm) and step-size tails (Lévy flights), with the Ornstein-Uhlenbeck process furnishing the bounded counterpoint. The unified MSD comparison plot shows all three departures from the linear law on a single log-log axis, making the geometric origin of $\alpha = 2H$, $\alpha = 2/\alpha_\text{stable}$, and the OU plateau visible at a glance.

---

## Setup

```bash
pip install numpy scipy matplotlib cycler
python "Brownian Motion; Random Walks.py"
python "Generalizations; Anomalous Diffusion & Levy Flights.py"
```

Plots are written to `Plots/` (created automatically if absent).

