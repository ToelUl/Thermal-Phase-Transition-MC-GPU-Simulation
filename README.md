# Thermal Phase Transition MC GPU Simulation

This project implements Monte Carlo simulations for studying thermal phase transitions using PyTorch. It leverages CUDA acceleration, a checkerboard (alternating) Metropolis update scheme, and parallel tempering to improve sampling efficiency. Models implemented include the XY model, Ising model, and q-state Potts model.

---

## 1. Key Features

- **CUDA Acceleration:**  
  Utilize GPU computing to significantly speed up the simulation process.

- **Checkerboard Metropolis Update:**  
  The simulation adopts a checkerboard update pattern. This allows for efficient and parallel updates of the lattice by splitting it into two interleaved sub-lattices.

- **Parallel Tempering:**  
  Enhance sampling efficiency by exchanging configurations between different temperature chains. This is particularly useful near critical points and low temperature region to overcome local minima and achieve better convergence.

- **Automatic Mixed Precision (AMP):**  
  Enable automatic mixed precision to leverage Tensor Cores on compatible GPUs for faster computation.

---

## 2. Background and Theory

Monte Carlo simulations in statistical physics help explore phase transitions and critical phenomena. The simulation uses the Metropolis algorithm with the acceptance probability

\[
P_{\text{acc}} = \min\left(1, \exp\left(-\frac{\Delta E}{T}\right)\right)
\]

where \(\Delta E\) is the energy change and \(T\) is the temperature. The use of parallel tempering further improves exploration of the energy landscape by periodically swapping configurations between temperature chains.

### Models Included

- **XY Model:**  
  Spins are continuous, represented by an angle \(\theta \in [0, 2\pi)\).  
  Energy:  
  \[
  E = -J \sum_{\langle i,j \rangle} \cos(\theta_i - \theta_j)
  \]
  *Adaptive updates* can be applied to adjust the angular perturbation dynamically.

- **Ising Model:**  
  Spins take values \(\pm1\).  
  Energy:  
  \[
  E = -J \sum_{\langle i,j \rangle} s_i s_j
  \]

- **Potts Model:**  
  Spins take integer values in \(\{0, 1, \dots, q-1\}\).  
  Energy is computed using the Kronecker delta:
  \[
  E = -J \sum_{\langle i,j \rangle} \delta(s_i, s_j)
  \]

---

## 3. System Requirements

- Python 3.7 or higher
- [PyTorch](https://pytorch.org/) with CUDA support (if using GPU acceleration)
- Other dependencies as listed in the provided `requirements.txt`

## 4. Example Usage

### XY Model Simulation
```bash
import time
import torch

# Set lattice size and temperature range
L = 32
T = torch.linspace(0.6, 1.5, steps=32)

# Initialize the XY model sampler with CUDA acceleration, AMP enabled, and parallel tempering
sampler_xy = XYModel(
    L=L,
    T=T,
    n_chains=30, # Number of temperature chains for parallel tempering
    adaptive=False,  
    target_acceptance=0.5,
    adapt_rate=0.1,
    device=torch.device("cuda"),
    use_amp=True,
    pt_enabled=True  # Parallel tempering enabled for better sampling
)

start = time.time()
# Run simulation: 10000 sweeps for thermalization and 3000 sweeps for production,
# with samples recorded every 10 sweeps and parallel tempering every 10 sweeps.
samples_xy = sampler_xy(n_sweeps=3000, n_therm=10000, decorrelate=10, pt_interval=10)
end = time.time()
print(f"Elapsed time: {end - start:.2f} s")
print(f"Samples shape: {samples_xy.shape}")

# Update spins with the collected samples and compute physical observables.
sampler_xy.spins = samples_xy
energy_xy = sampler_xy.compute_energy().mean(dim=1).cpu().numpy() / L**2
capacity_xy = sampler_xy.compute_heat_capacity().cpu().numpy()
stiffness_xy = sampler_xy.compute_spin_stiffness().cpu().numpy()

```

### Ising Model Simulation
```bash
import time
import torch

L = 32
T = torch.linspace(1.0, 3.5, steps=32)

# Initialize the Ising model sampler with CUDA acceleration and AMP enabled
sampler_ising = IsingModel(
    L=L,
    T=T,
    n_chains=30,
    device=torch.device("cuda"),
    use_amp=True,
    pt_enabled=False  
)

start = time.time()
# Run simulation: 3000 sweeps for thermalization and 1000 sweeps for production,
# with samples recorded every 10 sweeps.
samples_ising = sampler_ising(n_sweeps=1000, n_therm=3000, decorrelate=10)
end = time.time()
print(f"Elapsed time: {end - start:.2f} s")
print(f"Samples shape: {samples_ising.shape}")

sampler_ising.spins = samples_ising
energy_ising = sampler_ising.compute_energy().mean(dim=1).cpu().numpy() / L**2
capacity_ising = sampler_ising.compute_heat_capacity().cpu().numpy()
magnetization_ising = sampler_ising.compute_magnetization().cpu().numpy()

```

### Potts Model Simulation
```bash
import time
import torch

L = 32
T = torch.linspace(0.5, 1.5, steps=32)

# Initialize the Potts model sampler with q=3, CUDA acceleration, and AMP enabled
sampler_potts = PottsModel(
    L=L,
    T=T,
    q=3,
    n_chains=30,
    device=torch.device("cuda"),
    use_amp=True,
    pt_enabled=False  
)

start = time.time()
# Run simulation: 5000 sweeps for thermalization and 1000 sweeps for production,
# with samples recorded every 10 sweeps.
samples_potts = sampler_potts(n_sweeps=1000, n_therm=5000, decorrelate=10)
end = time.time()
print(f"Elapsed time: {end - start:.2f} s")
print(f"Samples shape: {samples_potts.shape}")

sampler_potts.spins = samples_potts
energy_potts = sampler_potts.compute_energy().mean(dim=1).cpu().numpy() / L**2
capacity_potts = sampler_potts.compute_heat_capacity().cpu().numpy()

```
