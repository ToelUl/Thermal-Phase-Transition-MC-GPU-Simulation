import gc
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from abc import ABC, abstractmethod

# =============================================================================
# Abstract Base Class: MonteCarloSampler
# =============================================================================
class MonteCarloSampler(nn.Module, ABC):
    r"""Abstract Monte Carlo sampler providing the skeleton for the sampling process.

    The class implements the general workflow:
      - forward(): Performs thermalization, production sampling, and parallel tempering exchanges.
      - one_sweep(): Executes a single sweep that updates both black and white sub-lattices.
      - parallel_tempering_exchange(): Attempts exchanges between adjacent temperature chains.
      - _prepare_checkerboard_indices(): Precomputes the checkerboard indices and their neighbors.

    Subclasses must implement the following abstract methods:
      1. init_spins(): Initialize the spin configuration (e.g., angles for XY, ±1 for Ising).
      2. metropolis_update_sub_lattice(): Perform Metropolis updates on a given sub-lattice.
      3. compute_energy(): Compute the energy tensor of the entire system.
    """

    def __init__(self,
                 L: int,
                 T: Tensor,
                 n_chains: int = 30,
                 J: float = 1.0,
                 device: torch.device = torch.device("cpu"),
                 use_amp: bool = False,
                 large_size_simulate: bool = False,
                 pt_enabled: bool = True) -> None:
        """
        Args:
            L (int): Linear size of the lattice.
            T (Tensor): Temperature tensor; its batch dimension sets the number of samples.
            n_chains (int, optional): Number of Monte Carlo chains per temperature. Defaults to 1.
            J (float, optional): Coupling constant. Defaults to 1.0.
            device (torch.device, optional): Device on which to perform computations. Defaults to CPU.
            use_amp (bool, optional): Flag to enable automatic mixed precision. Defaults to False.
            large_size_simulate (bool, optional): Flag for large-scale simulation (move samples to CPU). Defaults to False.
            pt_enabled (bool, optional): Flag to enable parallel tempering exchanges. Defaults to True.
        """
        super().__init__()
        self.L = L
        self.T = T.to(device)
        self.batch_size = self.T.shape[0]
        self.n_chains = n_chains
        self.J = J
        self.device = device
        self.use_amp = use_amp
        self.large_size_simulate = large_size_simulate
        self.pt_enabled = pt_enabled

        # Pre-register a zero tensor for future comparisons
        self.register_buffer('zero', torch.tensor(0, device=device))

        # Initialize the spin configuration; implementation is deferred to subclasses.
        self.init_spins()

        # Precompute the checkerboard indices and neighbor coordinates for sub-lattice updates.
        self._prepare_checkerboard_indices()

    @abstractmethod
    def init_spins(self) -> None:
        r"""Initialize the spin configuration.

        This method must be implemented by subclasses.
        For example:
          - XY model: angles ∈ [0, 2π)
          - Ising model: spins ∈ {+1, -1}
        """
        pass

    @abstractmethod
    def metropolis_update_sub_lattice(self, lattice_color: str, adaptive: bool = False) -> None:
        r"""Perform the Metropolis update on the specified sub-lattice.

        Args:
            lattice_color (str): Indicates which sub-lattice to update ('black' or 'white').
            adaptive (bool, optional): Flag for adaptive update parameters. Defaults to False.
        """
        pass

    @abstractmethod
    def compute_energy(self) -> Tensor:
        r"""Compute the energy of the system.

        Returns:
            Tensor: Energy tensor with shape [batch_size, n_chains].
        """
        pass

    def _prepare_checkerboard_indices(self) -> None:
        r"""Prepare checkerboard (black/white) indices and their four periodic neighbors.

        This method precomputes the indices for black and white sub-lattices as well as the corresponding
        neighbor coordinates (up, down, left, right) with periodic boundary conditions.
        The indices are broadcasted to shape [batch_size, n_chains, N_sites] for use in sub-lattice updates.
        """
        L = self.L
        i_coords = torch.arange(L, device=self.device)
        j_coords = torch.arange(L, device=self.device)
        grid_i, grid_j = torch.meshgrid(i_coords, j_coords, indexing='ij')

        # Create a checkerboard pattern: black sites where (i + j) is even; white sites otherwise.
        black_mask = ((grid_i + grid_j) % 2 == 0)
        white_mask = ~black_mask

        black_indices = torch.nonzero(black_mask, as_tuple=False).to(self.device)
        white_indices = torch.nonzero(white_mask, as_tuple=False).to(self.device)
        self.num_black = black_indices.shape[0]
        self.num_white = white_indices.shape[0]

        B, C = self.batch_size, self.n_chains
        batch_idx = torch.arange(B, device=self.device).view(B, 1, 1)
        chain_idx = torch.arange(C, device=self.device).view(1, C, 1)

        # Expand indices for black sites.
        black_i = black_indices[:, 0].view(1, 1, self.num_black).expand(B, C, self.num_black)
        black_j = black_indices[:, 1].view(1, 1, self.num_black).expand(B, C, self.num_black)

        # Expand indices for white sites.
        white_i = white_indices[:, 0].view(1, 1, self.num_white).expand(B, C, self.num_white)
        white_j = white_indices[:, 1].view(1, 1, self.num_white).expand(B, C, self.num_white)

        self.register_buffer('black_batch_idx', batch_idx.expand(B, C, self.num_black))
        self.register_buffer('black_chain_idx', chain_idx.expand(B, C, self.num_black))
        self.register_buffer('black_i_sites', black_i)
        self.register_buffer('black_j_sites', black_j)

        self.register_buffer('white_batch_idx', batch_idx.expand(B, C, self.num_white))
        self.register_buffer('white_chain_idx', chain_idx.expand(B, C, self.num_white))
        self.register_buffer('white_i_sites', white_i)
        self.register_buffer('white_j_sites', white_j)

        # Compute neighbors for both black and white sites.
        black_neighbors_i, black_neighbors_j = self._compute_neighbors(black_indices)
        white_neighbors_i, white_neighbors_j = self._compute_neighbors(white_indices)

        # Reshape and expand neighbor indices.
        black_neighbors_i = black_neighbors_i.view(1, 1, self.num_black, 4).expand(B, C, self.num_black, 4)
        black_neighbors_j = black_neighbors_j.view(1, 1, self.num_black, 4).expand(B, C, self.num_black, 4)
        white_neighbors_i = white_neighbors_i.view(1, 1, self.num_white, 4).expand(B, C, self.num_white, 4)
        white_neighbors_j = white_neighbors_j.view(1, 1, self.num_white, 4).expand(B, C, self.num_white, 4)

        self.register_buffer('black_neighbors_i', black_neighbors_i)
        self.register_buffer('black_neighbors_j', black_neighbors_j)
        self.register_buffer('white_neighbors_i', white_neighbors_i)
        self.register_buffer('white_neighbors_j', white_neighbors_j)

        # Clean up intermediate variables and force garbage collection.
        del (black_neighbors_i, black_neighbors_j, white_neighbors_i, white_neighbors_j,
             black_indices, white_indices, batch_idx, chain_idx,
             grid_i, grid_j, i_coords, j_coords, black_mask, white_mask)
        gc.collect()

    def _compute_neighbors(self, indices_2d: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Compute the four periodic neighbors for given 2D indices.

        The neighbors are determined by periodic boundary conditions:
            - Up:    (i-1, j)
            - Down:  (i+1, j)
            - Left:  (i, j-1)
            - Right: (i, j+1)

        Args:
            indices_2d (Tensor): Tensor of shape [N, 2] containing the (i, j) coordinates.

        Returns:
            Tuple[Tensor, Tensor]: Two tensors containing the neighbor indices along the first and second axes, each of shape [N, 4].
        """
        L = self.L
        i_sites = indices_2d[:, 0]
        j_sites = indices_2d[:, 1]

        up_i = (i_sites - 1) % L
        up_j = j_sites
        down_i = (i_sites + 1) % L
        down_j = j_sites
        left_i = i_sites
        left_j = (j_sites - 1) % L
        right_i = i_sites
        right_j = (j_sites + 1) % L

        neighbors_i = torch.stack([up_i, down_i, left_i, right_i], dim=1)
        neighbors_j = torch.stack([up_j, down_j, left_j, right_j], dim=1)

        del i_sites, j_sites, up_i, up_j, down_i, down_j, left_i, left_j, right_i, right_j
        return neighbors_i, neighbors_j

    def set_parallel_tempering(self, enabled: bool) -> None:
        """Enable or disable parallel tempering exchanges.

        Args:
            enabled (bool): If True, parallel tempering is enabled.
        """
        self.pt_enabled = enabled

    def forward(self,
                n_sweeps: int = 1000,
                n_therm: int = 10000,
                decorrelate: int = 10,
                pt_interval: int = 10) -> Tensor:
        r"""Execute the main Monte Carlo sampling process.

        The process includes:
          1. Thermalization: n_therm sweeps.
          2. Production sampling: n_sweeps sweeps (samples recorded every 'decorrelate' steps).
          3. (Optional) Parallel tempering exchange.

        Args:
            n_sweeps (int, optional): Number of production sweeps. Defaults to 5000.
            n_therm (int, optional): Number of thermalization sweeps. Defaults to 2000.
            decorrelate (int, optional): Interval between recorded samples. Defaults to 100.
            pt_interval (int, optional): Frequency of parallel tempering exchanges during production. Defaults to 10.

        Returns:
            Tensor: Collected samples with shape [batch_size, n_chains×num_samples, L, L].
        """
        T = self.T.to(self.device)
        assert T.shape[0] == self.batch_size, "The batch dimension of T must match batch_size."
        sample_list = []

        with torch.autocast(device_type=self.device.type,
                              enabled=(self.device.type in ["cuda", "cpu"] and self.use_amp)):
            with torch.no_grad():
                # Thermalization phase
                for sweep in range(n_therm):
                    self.one_sweep()
                    if self.pt_enabled and (sweep % pt_interval == 0):
                        self.parallel_tempering_exchange()
                    if sweep % 10 == 0:
                        gc.collect(generation=0)

                gc.collect(generation=1)

                # Production sampling phase
                for sweep in range(n_sweeps):
                    self.one_sweep()
                    if self.pt_enabled and (sweep % pt_interval == 0):
                        self.parallel_tempering_exchange()
                    if sweep % 10 == 0:
                        gc.collect(generation=0)

                    if sweep % decorrelate == 0:
                        # For large simulations, move sample to CPU to mitigate GPU memory constraints.
                        if self.large_size_simulate:
                            sample_list.append(self.spins.clone().cpu())
                            torch.cuda.empty_cache()
                        else:
                            sample_list.append(self.spins.clone())

                # Stack and permute samples to shape [B, C, num_samples, L, L]
                samples = torch.stack(sample_list, dim=0)
                samples = samples.permute(1, 2, 0, 3, 4).contiguous()
                del sample_list
                gc.collect()

        return samples.view(self.batch_size, -1, self.L, self.L)

    def one_sweep(self, adaptive: bool = False) -> None:
        r"""Perform a single Monte Carlo sweep.

        The sweep consists of sequentially updating both the black and white sub-lattices.
        The specific update rule is implemented in the subclass via the metropolis_update_sub_lattice() method.

        Args:
            adaptive (bool, optional): If True, use adaptive update parameters. Defaults to False.
        """
        self.metropolis_update_sub_lattice('black', adaptive=adaptive)
        self.metropolis_update_sub_lattice('white', adaptive=adaptive)

    def parallel_tempering_exchange(self) -> None:
        r"""Perform parallel tempering exchange between adjacent temperature chains.

        The method exchanges configurations between neighboring batches based on the Metropolis criterion.
        It compares energies and temperature values to decide whether to swap spins.
        """
        T = self.T
        B, C, _, _ = self.spins.shape  # spins shape: [B, C, L, L]
        energies = self.compute_energy()  # shape: [B, C]
        T_b = T.view(B, 1).expand(B, C)
        idx = torch.arange(B, device=self.device)

        # Randomly select starting indices for exchange pairs.
        if torch.randint(0, 2, (1,), device=self.device) == self.zero:
            start_idx = idx[::2]
        else:
            start_idx = idx[1::2]
        start_idx = start_idx[start_idx < (B - 1)]
        partner_idx = start_idx + 1

        E_start = energies[start_idx, :]
        E_partner = energies[partner_idx, :]
        T_start = T_b[start_idx, :]
        T_partner = T_b[partner_idx, :]

        # Calculate acceptance probability for the swap.
        delta = (1.0 / T_start - 1.0 / T_partner) * (E_start - E_partner)
        prob = torch.exp(delta).clamp(max=1.0)
        rand_vals = torch.rand_like(prob)
        swap_mask = (rand_vals < prob)

        if swap_mask.any():
            # Swap spins between the paired configurations.
            spins_start = self.spins[start_idx, :, :, :].clone()
            spins_partner = self.spins[partner_idx, :, :, :].clone()
            tmp = spins_start.clone()
            spins_start[swap_mask] = spins_partner[swap_mask]
            spins_partner[swap_mask] = tmp[swap_mask]
            self.spins[start_idx, :, :, :] = spins_start
            self.spins[partner_idx, :, :, :] = spins_partner
            del tmp, spins_start, spins_partner

        # Clean up intermediate variables.
        del B, C, energies, T_b, idx, start_idx, partner_idx, E_start, E_partner, T_start, T_partner, delta, prob, rand_vals, swap_mask
        gc.collect(generation=0)
