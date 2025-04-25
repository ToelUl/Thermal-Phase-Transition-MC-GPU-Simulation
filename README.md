# Thermal Phase Transition MC GPU Simulation

This project implements Monte Carlo simulations for studying thermal phase transitions using PyTorch. It leverages CUDA acceleration, a checkerboard (alternating) Metropolis update scheme, and parallel tempering to improve sampling efficiency. Models implemented include the XY model, Ising model, and q-state Potts model.

## Run Notebooks on Google Colab

- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ToelUl/Thermal-Phase-Transition-MC-GPU-Simulation/blob/main/Ising_model.ipynb) **Ising model Demo**
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ToelUl/Thermal-Phase-Transition-MC-GPU-Simulation/blob/main/Potts_model.ipynb) **Potts model Demo**
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ToelUl/Thermal-Phase-Transition-MC-GPU-Simulation/blob/main/XY_model.ipynb) **XY model Demo**


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

try:
    from gpu_mc import XYModel
except ModuleNotFoundError:
    !git clone https://github.com/ToelUl/Thermal-Phase-Transition-MC-GPU-Simulation.git
    !cp -r Thermal-Phase-Transition-MC-GPU-Simulation/gpu_mc ./
    from gpu_mc import XYModel

import torch

L = 32 # Lattice size
T_start = 0.1 # Start temperature
T_end = 2.0 # End temperature
precision = 0.05 # Temperature precision
device = "cuda:0"
T = torch.linspace(T_start, T_end, int((T_end-T_start)//precision)+1, device=device)
ensemble_number = 3000 # Number of samples
n_chains = 30 # Number of parallel chains. Suggested: 10~50
pt_interval = 2 # Parallel tempering interval. Suggested: 1~5
pt_prob = 0.1 # Parallel tempering probability 0.1~0.5
tau_pt = pt_interval / pt_prob # Autocorrelation time for parallel tempering
factor_therm = 15 # 10~50
factor_decorrelate = 2 # 1~10
tau = L**2 # Autocorrelation time
tau_eff = (tau_pt * tau) / (tau_pt + tau) # Effective Autocorrelation time
n_therm =  int(factor_therm * tau) # Number of thermalization sweeps
decorrelate = int(factor_decorrelate * tau_eff) # Number of decorrelation sweeps
n_sweeps = int(ensemble_number / n_chains) * decorrelate # Number of sweeps

print(f"Lattice size: {L}")
print(f"Temperature range: {T_start} to {T_end}")
print(f"Number of temperatures: {len(T)}")
print(f"Number of samples per temperature: {ensemble_number}")
print(f"Number of sweeps: {n_sweeps}")
print(f"Number of thermalization sweeps: {n_therm}")
print(f"Number of chains: {n_chains}")
print(f"Number of decorrelate: {decorrelate}")

sampler_xy = XYModel(
    L=L,
    T=T,
    n_chains=n_chains,
    # adaptive=True,
    # target_acceptance=0.6,
    # adapt_rate=0.1,
    device=torch.device(device),
    use_amp=True,
    pt_enabled=True, # Suggestions: Parallel tempering enabled for better sampling.
    )

start = time.time()
samples_xy = sampler_xy(
    n_sweeps=n_sweeps,
    n_therm=n_therm,
    decorrelate=decorrelate,
    pt_interval=pt_interval
)
end = time.time()
print(f"Elapsed time: {end - start:.2f} s")
print(f"Samples shape: {samples_ising.shape}")

# Update spins with the collected samples and compute physical observables.
sampler_xy.spins = samples_xy

energy_xy = sampler_xy.compute_average_energy().cpu().numpy()
capacity_xy = sampler_xy.compute_specific_heat_capacity().cpu().numpy()
stiffness_xy = sampler_xy.compute_spin_stiffness().cpu().numpy()
magnetization_xy = sampler_xy.compute_magnetization().cpu().numpy()
susceptibility_xy = sampler_xy.compute_susceptibility().cpu().numpy()
vortex_density_xy = sampler_xy.compute_vortex_density().cpu().numpy()
temp = sampler_xy.T.cpu().numpy()

```

### Ising Model Simulation
```bash
import time

try:
    from gpu_mc import IsingModel
except ModuleNotFoundError:
    !git clone https://github.com/ToelUl/Thermal-Phase-Transition-MC-GPU-Simulation.git
    !cp -r Thermal-Phase-Transition-MC-GPU-Simulation/gpu_mc ./
    from gpu_mc import IsingModel

import torch

L = 16 # Lattice size
T_start = 1.0 # Start temperature
T_end = 3.5 # End temperature
precision = 0.05 # Temperature precision
device = "cuda:0"
T = torch.linspace(T_start, T_end, int((T_end-T_start)//precision)+1, device=device)
ensemble_number = 3000 # Number of samples
n_chains = 30 # Number of parallel chains. Suggested: 10~50
factor_therm = 5 # 10~50
factor_decorrelate = 1 # 1~10
tau = L**2 # Autocorrelation time
n_therm =  int(factor_therm * tau) # Number of thermalization sweeps
decorrelate = int(factor_decorrelate * L) # Number of decorrelation sweeps
n_sweeps = int(ensemble_number / n_chains) * decorrelate # Number of sweeps

print(f"Lattice size: {L}")
print(f"Temperature range: {T_start} to {T_end}")
print(f"Number of temperatures: {len(T)}")
print(f"Number of samples per temperature: {ensemble_number}")
print(f"Number of sweeps: {n_sweeps}")
print(f"Number of thermalization sweeps: {n_therm}")
print(f"Number of chains: {n_chains}")
print(f"Number of decorrelate: {decorrelate}")

sampler_ising = IsingModel(
    L=L,
    T=T,
    n_chains=n_chains,
    device=torch.device(device),
    )

start = time.time()
samples_ising = sampler_ising(
    n_sweeps=n_sweeps,
    n_therm=n_therm,
    decorrelate=decorrelate
)
end = time.time()
print(f"Elapsed time: {end - start:.2f} s")
print(f"Samples shape: {samples_ising.shape}")

# Update spins with the collected samples and compute physical observables.
sampler_ising.spins = samples_ising

energy_ising = sampler_ising.compute_average_energy().cpu().numpy()
capacity_ising = sampler_ising.compute_specific_heat_capacity().cpu().numpy()
magnetization_ising = sampler_ising.compute_magnetization().cpu().numpy()
exact_magnetization_ising = sampler_ising.compute_exact_magnetization().cpu().numpy()
susceptibility_ising = sampler_ising.compute_susceptibility().cpu().numpy()
binder_cumulant_ising = sampler_ising.compute_binder_cumulant().cpu().numpy()
domain_wall_density_ising = sampler_ising.compute_domain_wall_density().cpu().numpy()
temp = sampler_ising.T.cpu().numpy()

```

### Potts Model Simulation
```bash
import time

try:
    from gpu_mc import PottsModel
except ModuleNotFoundError:
    !git clone https://github.com/ToelUl/Thermal-Phase-Transition-MC-GPU-Simulation.git
    !cp -r Thermal-Phase-Transition-MC-GPU-Simulation/gpu_mc ./
    from gpu_mc import PottsModel

import torch

L = 16 # Lattice size
q = 3 # Number of states in the Potts model
T_start = 0.5 # Start temperature
T_end = 1.5 # End temperature
precision = 0.03 # Temperature precision
device = "cuda:0"
T = torch.linspace(T_start, T_end, int((T_end-T_start)//precision)+1, device=device)
ensemble_number = 3000 # Number of samples
n_chains = 30 # Number of parallel chains. Suggested: 10~50
factor_therm = 5 # 10~50
factor_decorrelate = 1 # 1~10
tau = L**2 # Autocorrelation time
n_therm =  int(factor_therm * tau) # Number of thermalization sweeps
decorrelate = int(factor_decorrelate * L) # Number of decorrelation sweeps
n_sweeps = int(ensemble_number / n_chains) * decorrelate # Number of sweeps

print(f"Lattice size: {L}")
print(f"Number of states: {q}")
print(f"Temperature range: {T_start} to {T_end}")
print(f"Number of temperatures: {len(T)}")
print(f"Number of samples per temperature: {ensemble_number}")
print(f"Number of sweeps: {n_sweeps}")
print(f"Number of thermalization sweeps: {n_therm}")
print(f"Number of chains: {n_chains}")
print(f"Number of decorrelate: {decorrelate}")

sampler_potts = PottsModel(
    L=L,
    T=T,
    q=q,
    n_chains=n_chains,
    device=torch.device(device),
    use_amp=True,
    )

start = time.time()
samples_potts = sampler_potts(
    n_sweeps=n_sweeps,
    n_therm=n_therm,
    decorrelate=decorrelate
)
end = time.time()
print(f"Elapsed time: {end - start:.2f} s")
print(f"Samples shape: {samples_potts.shape}")

# Update spins with the collected samples and compute physical observables.
sampler_potts.spins = samples_potts

energy_potts = sampler_potts.compute_average_energy().cpu().numpy()
capacity_potts = sampler_potts.compute_specific_heat_capacity().cpu().numpy()
magnetization_potts = sampler_potts.compute_magnetization().cpu().numpy()
susceptibility_potts = sampler_potts.compute_susceptibility().cpu().numpy()
binder_cumulant_potts = sampler_potts.compute_binder_cumulant().cpu().numpy()
entropy_potts = sampler_potts.compute_entropy().cpu().numpy()
temp = sampler_potts.T.cpu().numpy()

```
