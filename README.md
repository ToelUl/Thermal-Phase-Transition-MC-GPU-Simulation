# Thermal Phase Transition MC GPU Simulation

This project implements Monte Carlo simulations for studying thermal phase transitions using PyTorch. It leverages CUDA acceleration, a checkerboard (alternating) Metropolis update scheme, and parallel tempering to improve sampling efficiency. Models implemented include the XY model, Ising model, and q-state Potts model.

## Documentation
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://deepwiki.com/ToelUl/Thermal-Phase-Transition-MC-GPU-Simulation)

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
$$
P_{\text{acc}} = \min\left(1, \exp\left(-\frac{\Delta E}{T}\right)\right)
$$
where $\left(\Delta E\right)$ is the energy change and $\left(T\right)$ is the temperature. The use of parallel tempering further improves exploration of the energy landscape by periodically swapping configurations between temperature chains.

### Models Included

- **XY Model:**  
  Spins are continuous, represented by an angle $\left(\theta \in [0, 2\pi)\right)$.  
  Energy:  
$$
E = -J \sum_{\langle i,j \rangle} \cos(\theta_i - \theta_j)
$$
  *Adaptive updates* can be applied to adjust the angular perturbation dynamically.

- **Ising Model:**  
  Spins take values $\left(\pm1\right)$.  
  Energy:  
$$
E = -J \sum_{\langle i,j \rangle} s_i s_j
$$

- **Potts Model:**  
  Spins take integer values in $\left(\{0, 1, \dots, q-1\}\right)$.  
  Energy is computed using the Kronecker delta:
$$
E = -J \sum_{\langle i,j \rangle} \delta(s_i, s_j)
$$

---

## 3. System Requirements

- Python 3.9 or higher
- [PyTorch](https://pytorch.org/) with CUDA support (if using GPU acceleration)
- Other dependencies as listed in the provided `requirements.txt`


