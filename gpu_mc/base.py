import gc
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional
from abc import ABC, abstractmethod

# =============================================================================
# Abstract Base Class: MonteCarloSampler
# =============================================================================
class MonteCarloSampler(nn.Module, ABC):
    r"""Abstract Monte Carlo sampler providing the skeleton for the sampling process.

    This abstract base class defines the common structure and workflow for
    Monte Carlo simulations of lattice models using PyTorch. It handles
    aspects like device placement, automatic mixed precision (AMP),
    checkerboard updates, parallel tempering, and sample collection.

    The typical simulation workflow managed by the `forward` method includes:
      - Thermalization phase to bring the system towards equilibrium.
      - Production phase where configurations are sampled periodically.
      - Optional Parallel Tempering exchanges between replicas at different
        temperatures to improve sampling efficiency across potential energy barriers.

    Subclasses are required to implement the model-specific details:
      1. `init_spins()`: How to initialize the spin configurations (e.g.,
         random angles for XY model, random +/-1 for Ising model).
      2. `metropolis_update_sub_lattice()`: The core Metropolis-Hastings update
         rule applied to one sub-lattice (e.g., black or white sites in a
         checkerboard scheme). This should typically compute the energy change
         of a proposed move rather than the full system energy.
      3. `compute_energy()`: A method to calculate the total energy of the
         current spin configurations across all batches and chains. This is
         primarily used for parallel tempering exchanges.

    Attributes:
        L (int): Linear size of the square lattice (L x L).
        T (Tensor): Tensor containing the temperatures for each batch replica.
            Shape: [batch_size].
        n_chains (int): Number of independent Monte Carlo chains simulated
            per temperature (i.e., per batch element).
        J (float): Coupling constant for the interaction term (model-specific).
        device (torch.device): The PyTorch device (CPU or GPU) for computations.
        use_amp (bool): Whether to enable Automatic Mixed Precision for potentially
            faster computation on compatible hardware (GPUs).
        large_size_simulate (bool): If True, samples collected during production
            are moved to CPU memory immediately to conserve GPU memory for very
            large lattices. This incurs a performance cost due to data transfer.
        pt_enabled (bool): If True, parallel tempering exchanges are performed.
        batch_size (int): Number of different temperatures being simulated in parallel.
        spins (Tensor): The current state of the spins for all chains and batches.
            Shape: [batch_size, n_chains, L, L]. This must be initialized by
            the subclass's `init_spins` method.
        num_black (int): Number of sites in the 'black' sub-lattice.
        num_white (int): Number of sites in the 'white' sub-lattice.
        (various registered buffers): Precomputed indices for checkerboard updates
            and neighbor lookups to accelerate computations.
    """

    def __init__(self,
                 L: int,
                 T: Tensor,
                 n_chains: int = 30,
                 J: float = 1.0,
                 device: Optional[torch.device] = None,
                 use_amp: bool = False,
                 large_size_simulate: bool = False,
                 pt_enabled: bool = True) -> None:
        """Initializes the MonteCarloSampler.

        Args:
            L (int): Linear size of the lattice.
            T (Tensor): 1D Tensor of temperatures. Its length determines the batch size.
            n_chains (int, optional): Number of Monte Carlo chains per temperature. Defaults to 30.
            J (float, optional): Coupling constant (model-specific interpretation). Defaults to 1.0.
            device (torch.device, optional): Device for computations. If None, attempts to use
                CUDA if available, otherwise CPU. Defaults to None.
            use_amp (bool, optional): Enable Automatic Mixed Precision. Defaults to False.
            large_size_simulate (bool, optional): Move collected samples to CPU to save GPU memory.
                Defaults to False.
            pt_enabled (bool, optional): Enable parallel tempering exchanges. Defaults to True.
        """
        super().__init__()
        self.L = L
        if T.ndim != 1:
            raise ValueError(f"Temperature tensor T must be 1D, but got shape {T.shape}")
        self.batch_size = T.shape[0]

        # Determine compute device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            # Add check for MPS (Apple Silicon GPU) if needed
            # elif torch.backends.mps.is_available():
            #     self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.T = T.to(self.device)
        self.n_chains = n_chains
        self.J = J
        self.use_amp = use_amp
        self.large_size_simulate = large_size_simulate
        self.pt_enabled = pt_enabled

        # Validate parameters
        if self.large_size_simulate and self.device.type == 'cpu':
             print("Warning: large_size_simulate=True has no effect when device is CPU.")
             self.large_size_simulate = False
        if self.use_amp and self.device.type not in ['cuda', 'mps']: # Add 'mps' if needed
            print(f"Warning: use_amp=True is typically only effective on CUDA devices, but device is {self.device.type}.")
            # self.use_amp = False # Optionally disable AMP if not on GPU

        # Pre-register a zero tensor for comparisons (used in parallel_tempering)
        self.register_buffer('zero', torch.tensor(0, device=self.device))

        # Spin configuration must be initialized by the subclass
        # It should define self.spins with shape [batch_size, n_chains, L, L]
        self.init_spins()
        if not hasattr(self, 'spins') or self.spins.shape != (self.batch_size, self.n_chains, self.L, self.L):
             raise NotImplementedError("Subclass must implement init_spins() and define self.spins"
                                       f" with shape ({self.batch_size}, {self.n_chains}, {self.L}, {self.L})")

        # Precompute indices for efficient checkerboard updates
        self._prepare_checkerboard_indices()

        print(f"Initialized MonteCarloSampler on device: {self.device}")
        print(f" L={L}, BatchSize={self.batch_size}, ChainsPerTemp={n_chains}, AMP={use_amp}, PT={pt_enabled}, LargeSim={large_size_simulate}")


    @abstractmethod
    def init_spins(self) -> None:
        r"""Initialize the spin configuration tensor `self.spins`.

        This method **must** be implemented by subclasses. It needs to create
        the `self.spins` tensor with the correct dimensions and initial state
        appropriate for the specific model being simulated.

        Shape of `self.spins` must be `[batch_size, n_chains, L, L]`.
        The data type should be appropriate (e.g., float for XY, float or int for Ising).
        The tensor should be created on `self.device`.

        Example for Ising model:
            ```python
            self.spins = torch.randint(0, 2,
                                       (self.batch_size, self.n_chains, self.L, self.L),
                                       device=self.device,
                                       dtype=torch.float32) * 2.0 - 1.0
            ```
        Example for XY model:
            ```python
            self.spins = torch.rand((self.batch_size, self.n_chains, self.L, self.L),
                                    device=self.device) * 2.0 * torch.pi
            ```
        """
        pass

    @abstractmethod
    def metropolis_update_sub_lattice(self, lattice_color: str, adaptive: bool = False) -> None:
        r"""Perform Metropolis-Hastings updates on one sub-lattice ('black' or 'white').

        This method **must** be implemented by subclasses. It defines the core
        update step of the Monte Carlo simulation for the specific model.

        It should operate **in-place** on the `self.spins` tensor.

        Key steps typically involve:
        1. Proposing a change to the spins on the specified sub-lattice.
        2. Calculating the change in energy ($\Delta E$) resulting from the proposed change.
           Crucially, this should be the *local* energy change, not the total system energy.
        3. Calculating the acceptance probability based on $\Delta E$ and temperature $T$.
           $P_{acc} = \min(1, \exp(-\Delta E / T))$
        4. Accepting or rejecting the proposed change based on the probability.

        Args:
            lattice_color (str): Specifies which sub-lattice to update ('black' or 'white').
                                 The precomputed indices for the corresponding sub-lattice
                                 (e.g., `self.black_i_sites`, `self.black_neighbors_i`)
                                 should be used.
            adaptive (bool, optional): Flag indicating if adaptive parameters (e.g., proposal
                                       width) should be used. Defaults to False. Subclasses
                                       can choose how to implement or ignore this flag.
        """
        pass

    @abstractmethod
    def compute_energy(self) -> Tensor:
        r"""Compute the total energy of the current spin configurations.

        This method **must** be implemented by subclasses. It calculates the
        total energy for each chain in each batch based on the current `self.spins`
        configuration and the model's Hamiltonian (implicitly defined by the
        energy calculation).

        Returns:
            Tensor: A tensor containing the total energy for each chain.
                    Shape: `[batch_size, n_chains]`.
        """
        pass

    def _prepare_checkerboard_indices(self) -> None:
        r"""Precompute indices for checkerboard updates and neighbor lookups.

        This method creates and registers buffers containing the coordinates `(i, j)`
        for sites belonging to the 'black' and 'white' sub-lattices of a checkerboard
        pattern. It also computes the coordinates of their four nearest neighbors
        using periodic boundary conditions.

        These precomputed indices are expanded to match the `batch_size` and `n_chains`
        dimensions, allowing for vectorized updates in `metropolis_update_sub_lattice`.
        """
        L = self.L
        # Generate grid coordinates
        i_coords = torch.arange(L, device=self.device)
        j_coords = torch.arange(L, device=self.device)
        grid_i, grid_j = torch.meshgrid(i_coords, j_coords, indexing='ij')

        # Create checkerboard masks
        black_mask = ((grid_i + grid_j) % 2 == 0)
        white_mask = ~black_mask

        # Get 2D indices for black and white sites
        black_indices_2d = torch.nonzero(black_mask, as_tuple=False) # Shape [num_black, 2]
        white_indices_2d = torch.nonzero(white_mask, as_tuple=False) # Shape [num_white, 2]
        self.num_black = black_indices_2d.shape[0]
        self.num_white = white_indices_2d.shape[0]

        # Ensure the lattice size is compatible with checkerboard updates
        # (Total sites should equal black + white sites)
        assert self.num_black + self.num_white == L * L, "Checkerboard decomposition failed."

        # --- Prepare indices expanded for batch and chain dimensions ---
        B, C = self.batch_size, self.n_chains
        # Base indices for broadcasting
        batch_idx = torch.arange(B, device=self.device).view(B, 1, 1)
        chain_idx = torch.arange(C, device=self.device).view(1, C, 1)

        # Black site indices (expanded)
        black_i = black_indices_2d[:, 0].view(1, 1, self.num_black).expand(B, C, self.num_black)
        black_j = black_indices_2d[:, 1].view(1, 1, self.num_black).expand(B, C, self.num_black)
        self.register_buffer('black_batch_idx', batch_idx.expand(B, C, self.num_black))
        self.register_buffer('black_chain_idx', chain_idx.expand(B, C, self.num_black))
        self.register_buffer('black_i_sites', black_i)
        self.register_buffer('black_j_sites', black_j)

        # White site indices (expanded)
        white_i = white_indices_2d[:, 0].view(1, 1, self.num_white).expand(B, C, self.num_white)
        white_j = white_indices_2d[:, 1].view(1, 1, self.num_white).expand(B, C, self.num_white)
        self.register_buffer('white_batch_idx', batch_idx.expand(B, C, self.num_white))
        self.register_buffer('white_chain_idx', chain_idx.expand(B, C, self.num_white))
        self.register_buffer('white_i_sites', white_i)
        self.register_buffer('white_j_sites', white_j)

        # --- Compute and register neighbor indices ---
        black_neighbors_i, black_neighbors_j = self._compute_neighbors(black_indices_2d)
        white_neighbors_i, white_neighbors_j = self._compute_neighbors(white_indices_2d)

        # Reshape and expand neighbor indices for batch and chains
        # Shape becomes [B, C, num_sites_sublattice, 4] where 4 is for (up, down, left, right)
        black_neighbors_i = black_neighbors_i.view(1, 1, self.num_black, 4).expand(B, C, self.num_black, 4)
        black_neighbors_j = black_neighbors_j.view(1, 1, self.num_black, 4).expand(B, C, self.num_black, 4)
        white_neighbors_i = white_neighbors_i.view(1, 1, self.num_white, 4).expand(B, C, self.num_white, 4)
        white_neighbors_j = white_neighbors_j.view(1, 1, self.num_white, 4).expand(B, C, self.num_white, 4)

        self.register_buffer('black_neighbors_i', black_neighbors_i)
        self.register_buffer('black_neighbors_j', black_neighbors_j)
        self.register_buffer('white_neighbors_i', white_neighbors_i)
        self.register_buffer('white_neighbors_j', white_neighbors_j)

        # Clean up large intermediate tensors that are no longer needed
        del (black_neighbors_i, black_neighbors_j, white_neighbors_i, white_neighbors_j,
             black_indices_2d, white_indices_2d, batch_idx, chain_idx, black_i, black_j, white_i, white_j,
             grid_i, grid_j, i_coords, j_coords, black_mask, white_mask)
        # Optional: Trigger garbage collection if memory pressure is observed after setup
        # gc.collect()


    def _compute_neighbors(self, indices_2d: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Compute the four periodic neighbors for given 2D indices.

        Calculates the `(i, j)` coordinates of the up, down, left, and right
        neighbors for each site specified in `indices_2d`, applying periodic
        boundary conditions based on the lattice size `self.L`.

        Args:
            indices_2d (Tensor): Tensor of shape `[N, 2]` containing the `(i, j)`
                                 coordinates of N sites.

        Returns:
            Tuple[Tensor, Tensor]:
                - neighbors_i: Tensor of shape `[N, 4]` containing the i-coordinates
                  of the (up, down, left, right) neighbors.
                - neighbors_j: Tensor of shape `[N, 4]` containing the j-coordinates
                  of the (up, down, left, right) neighbors.
        """
        L = self.L
        i_sites = indices_2d[:, 0] # Shape [N]
        j_sites = indices_2d[:, 1] # Shape [N]

        # Compute neighbor coordinates with periodic boundary conditions
        up_i = (i_sites - 1 + L) % L # More robust modulo for negative numbers
        up_j = j_sites
        down_i = (i_sites + 1) % L
        down_j = j_sites
        left_i = i_sites
        left_j = (j_sites - 1 + L) % L # More robust modulo for negative numbers
        right_i = i_sites
        right_j = (j_sites + 1) % L

        # Stack coordinates: Dimension 1 corresponds to [up, down, left, right]
        neighbors_i = torch.stack([up_i, down_i, left_i, right_i], dim=1) # Shape [N, 4]
        neighbors_j = torch.stack([up_j, down_j, left_j, right_j], dim=1) # Shape [N, 4]

        # No need to explicitly delete local variables like i_sites, j_sites etc.
        # Python's garbage collector handles this when the function exits.
        return neighbors_i, neighbors_j

    def set_parallel_tempering(self, enabled: bool) -> None:
        """Enable or disable parallel tempering exchanges during the simulation.

        Args:
            enabled (bool): Set to True to enable PT, False to disable.
        """
        self.pt_enabled = enabled
        print(f"Parallel Tempering set to: {self.pt_enabled}")

    def forward(self,
                n_sweeps: int = 5000,
                n_therm: int = 2000,
                decorrelate: int = 100,
                pt_interval: int = 10) -> Tensor:
        r"""Execute the main Monte Carlo sampling process.

        Performs the thermalization and production phases of the simulation.

        The process involves:
          1. Thermalization: Run `n_therm` sweeps to allow the system to reach
             a state independent of the initial configuration. Parallel tempering
             exchanges may occur during this phase if enabled. No samples are stored.
          2. Production: Run `n_sweeps` sweeps. Every `decorrelate` sweeps, the
             current spin configuration `self.spins` is cloned and stored. Parallel
             tempering exchanges continue during this phase if enabled.
          3. Sample Collection: Samples are efficiently collected by pre-allocating
             a tensor and filling it incrementally.

        Args:
            n_sweeps (int, optional): Number of production sweeps. Defaults to 5000.
            n_therm (int, optional): Number of thermalization sweeps. Defaults to 2000.
            decorrelate (int, optional): Interval between storing samples during the
                                        production phase. Must be positive. Defaults to 100.
            pt_interval (int, optional): Frequency (in sweeps) at which parallel tempering
                                       exchanges are attempted. Must be positive. Defaults to 10.

        Returns:
            Tensor: A tensor containing the collected spin configurations from the
                    production phase. The shape is `[batch_size, n_chains * num_samples, L, L]`,
                    where `num_samples = n_sweeps // decorrelate`.

        Raises:
            ValueError: If `decorrelate` or `pt_interval` are not positive.
            ValueError: If the input temperature tensor `self.T` batch size doesn't match
                        the `batch_size` determined during initialization.
        """
        # --- Input Validation ---
        if self.T.shape[0] != self.batch_size:
             raise ValueError(f"The batch dimension of the temperature tensor T ({self.T.shape[0]}) "
                              f"must match the sampler's batch_size ({self.batch_size}). "
                              "Did T change after initialization?")
        if decorrelate <= 0:
            raise ValueError(f"decorrelate interval must be positive, got {decorrelate}")
        if pt_interval <= 0:
            raise ValueError(f"pt_interval must be positive, got {pt_interval}")

        # --- Sample Storage Setup ---
        num_samples_to_collect = n_sweeps // decorrelate
        if num_samples_to_collect == 0:
             print(f"Warning: n_sweeps ({n_sweeps}) is less than decorrelate ({decorrelate}). "
                   "No samples will be collected during production.")
             # Return an empty tensor with the expected final shape but zero samples
             return torch.empty((self.batch_size, 0, self.L, self.L),
                                dtype=self.spins.dtype, device=self.device)

        # Determine the device for storing samples based on the large_size_simulate flag
        sample_storage_device = torch.device('cpu') if self.large_size_simulate else self.device

        # Pre-allocate tensor for storing samples.
        # Initial Shape: [num_samples, batch_size, n_chains, L, L]
        samples = torch.empty(
            (num_samples_to_collect, self.batch_size, self.n_chains, self.L, self.L),
            dtype=self.spins.dtype,
            device=sample_storage_device
        )
        sample_idx = 0 # Index to track the next slot in the samples tensor

        # Determine device type for AMP context
        amp_device_type = self.device.type
        if amp_device_type not in ["cuda", "mps", "cpu"]: # Add 'mps' if Apple Silicon is used
            amp_device_type = "cpu" # Fallback for AMP context if device is unusual

        # --- Simulation Execution ---
        print(f"Starting simulation: {n_therm} thermalization sweeps, {n_sweeps} production sweeps.")
        with torch.autocast(device_type=amp_device_type, enabled=self.use_amp):
            with torch.no_grad(): # Disable gradient calculations for performance
                # === Thermalization Phase ===
                print("Thermalization phase...")
                for sweep in range(n_therm):
                    self.one_sweep()
                    # Attempt parallel tempering exchange every pt_interval sweeps
                    if self.pt_enabled and (sweep + 1) % pt_interval == 0:
                        self.parallel_tempering_exchange()

                    # Optional: Add progress reporting here if needed
                    # if (sweep + 1) % (n_therm // 10) == 0:
                    #    print(f" Thermalization sweep {sweep+1}/{n_therm}")


                # === Production Sampling Phase ===
                print("Production phase...")
                for sweep in range(n_sweeps):
                    self.one_sweep()
                    # Attempt parallel tempering exchange every pt_interval sweeps
                    if self.pt_enabled and (sweep + 1) % pt_interval == 0:
                        self.parallel_tempering_exchange()

                    # Record sample every decorrelate sweeps
                    if (sweep + 1) % decorrelate == 0:
                        if sample_idx < num_samples_to_collect:
                            # Clone the current spins state
                            current_spins = self.spins.clone()
                            # Store the cloned spins in the pre-allocated tensor
                            if self.large_size_simulate:
                                samples[sample_idx] = current_spins.cpu() # Move to CPU if specified
                            else:
                                samples[sample_idx] = current_spins # Keep on original device
                            sample_idx += 1
                        # This else block should ideally not be reached due to num_samples_to_collect calculation
                        # else:
                        #    print(f"Warning: Attempted to write sample beyond allocated space at sweep {sweep+1}.")

                    # Optional: Add progress reporting here if needed
                    # if (sweep + 1) % (n_sweeps // 10) == 0:
                    #     print(f" Production sweep {sweep+1}/{n_sweeps} (Collected {sample_idx}/{num_samples_to_collect} samples)")


        print(f"Simulation finished. Collected {sample_idx * self.n_chains} samples.")

        # --- Reshape Samples for Output ---
        # Current shape: [num_samples, B, C, L, L]
        # Target shape:  [B, C * num_samples, L, L]

        # Permute to bring Batch and Chain dimensions first: [B, C, num_samples, L, L]
        samples = samples.permute(1, 2, 0, 3, 4)

        # Ensure the tensor is contiguous in memory before reshaping with view()
        # This is often necessary after permute().
        samples = samples.contiguous()

        # Reshape to merge Chain and Sample dimensions: [B, C * num_samples, L, L]
        final_samples = samples.view(self.batch_size, -1, self.L, self.L)

        del samples
        gc.collect()

        return final_samples

    def one_sweep(self, adaptive: bool = False) -> None:
        r"""Perform a single full Monte Carlo sweep over the lattice.

        A standard sweep consists of updating all spins once. This implementation
        uses a checkerboard decomposition, updating all 'black' sites first,
        followed by all 'white' sites. This ensures that updates within each
        sub-lattice step are independent.

        The actual update logic is delegated to the subclass's implementation
        of `metropolis_update_sub_lattice`.

        Args:
            adaptive (bool, optional): Passed down to `metropolis_update_sub_lattice`.
                                       Indicates if adaptive parameters should be used.
                                       Defaults to False.
        """
        self.metropolis_update_sub_lattice('black', adaptive=adaptive)
        self.metropolis_update_sub_lattice('white', adaptive=adaptive)

    def parallel_tempering_exchange(self) -> None:
        r"""Attempt parallel tempering exchanges between adjacent temperatures.

        This method facilitates the exchange of configurations (spin states)
        between simulations running at adjacent temperatures in the batch.
        The exchange is accepted or rejected based on the Metropolis criterion
        applied to the energy difference and temperature difference between
        the pair of replicas being considered for swapping.

        The selection of pairs (e.g., (0,1), (2,3), ... or (1,2), (3,4), ...)
        is randomized at each call to ensure detailed balance.

        Requires the subclass to implement `compute_energy()`.
        """
        # Get current energies and temperatures
        # Ensure T is expanded correctly: [B, C] where C is broadcasted
        T_expanded = self.T.view(self.batch_size, 1).expand(self.batch_size, self.n_chains) # Shape [B, C]
        try:
             energies = self.compute_energy() # Expected shape [B, C]
             if energies.shape != (self.batch_size, self.n_chains):
                 raise ValueError(f"compute_energy() returned shape {energies.shape}, expected ({self.batch_size}, {self.n_chains})")
        except NotImplementedError:
             print("Warning: Skipping Parallel Tempering exchange because compute_energy() is not implemented.")
             return
        except Exception as e:
             print(f"Warning: Error during compute_energy(): {e}. Skipping PT exchange.")
             return


        B = self.batch_size
        idx = torch.arange(B, device=self.device)

        # Randomly choose which pairs to attempt swapping (even/odd start)
        if torch.rand(1, device=self.device) < 0.5: # Compare float rand to 0.5
            # Try swapping pairs (0,1), (2,3), ...
            start_indices = idx[::2]
        else:
            # Try swapping pairs (1,2), (3,4), ...
            start_indices = idx[1::2]

        # Ensure indices are within bounds (last index might not have a partner)
        valid_start_indices = start_indices[start_indices < (B - 1)]
        if valid_start_indices.numel() == 0:
             return # No pairs to swap

        partner_indices = valid_start_indices + 1

        # Gather energies and temperatures for the selected pairs
        # Shapes will be [num_pairs, n_chains]
        E_start = energies[valid_start_indices]
        E_partner = energies[partner_indices]
        T_start = T_expanded[valid_start_indices]
        T_partner = T_expanded[partner_indices]

        # Calculate the exponent for the Metropolis acceptance probability
        # delta = (beta_partner - beta_start) * (E_partner - E_start)
        #       = (1/T_partner - 1/T_start) * (E_partner - E_start)
        # Note the sign convention used here matches the probability exp(delta)
        delta = (1.0 / T_start - 1.0 / T_partner) * (E_start - E_partner)

        # Calculate acceptance probability P = min(1, exp(delta))
        # Use clamp(max=1.0) for numerical stability instead of min(1, exp(delta))
        prob = torch.exp(delta).clamp_(max=1.0) # In-place clamp

        # Generate random numbers for acceptance check
        rand_vals = torch.rand_like(prob) # Shape [num_pairs, n_chains]

        # Determine which pairs+chains actually swap
        swap_mask = (rand_vals < prob) # Shape [num_pairs, n_chains]

        # Perform the swaps if any are accepted
        if swap_mask.any():
            # We need to swap the actual spin configurations in self.spins
            # Expand the swap_mask to match the spin dimensions for broadcasting/masking
            # Mask shape needs to be [num_pairs, n_chains, 1, 1] to broadcast over L, L
            expanded_swap_mask = swap_mask.view(swap_mask.shape[0], self.n_chains, 1, 1)

            # Get references (not clones initially) to the spins to be potentially swapped
            spins_start_ref = self.spins[valid_start_indices]   # Shape [num_pairs, C, L, L]
            spins_partner_ref = self.spins[partner_indices] # Shape [num_pairs, C, L, L]

            # Clone *only* the data needed for the swap to avoid modifying the original
            # before assignment, especially important for torch.where logic.
            original_spins_start = spins_start_ref.clone()
            original_spins_partner = spins_partner_ref.clone()

            # Use torch.where to perform the swap efficiently without extra clones
            # where(condition, value_if_true, value_if_false)
            self.spins[valid_start_indices] = torch.where(
                expanded_swap_mask, original_spins_partner, original_spins_start
            )
            self.spins[partner_indices] = torch.where(
                expanded_swap_mask, original_spins_start, original_spins_partner
            )

            # Clean up temporary clones
            del original_spins_start, original_spins_partner, expanded_swap_mask

        # No need to explicitly delete local variables like energies, T_expanded, delta, etc.
        # Minimal GC impact compared to loop-based GC.


    @staticmethod
    def high_precision_derivative(
        seq: torch.Tensor,
        spacing: float = 1.0
    ) -> Tensor:
        """Compute the first derivative of a 1D tensor using finite differences.

        Uses 4th-order central differences for interior points and 2nd-order
        forward/backward differences at the boundaries for improved accuracy
        compared to simpler schemes. Leverages vectorized operations and
        in-place modifications where possible for performance.

        Args:
            seq (torch.Tensor): A 1D tensor of function values, shape `(N,)`.
            spacing (float, optional): The uniform spacing `h` between the
                                       points in `seq`. Defaults to 1.0.

        Returns:
            torch.Tensor: A 1D tensor of shape `(N,)` containing the approximate
                          derivative values at each point.

        Raises:
            ValueError: If `seq` is not a 1D tensor or if its length `N` is less than 5,
                        as 5 points are required for the 4th-order stencil.
        """
        if seq.ndim != 1:
            raise ValueError(f"Input tensor must be 1D, but got {seq.ndim} dimensions.")
        N = seq.size(0)
        if N < 5:
            raise ValueError(f"Input tensor length must be at least 5 for 4th-order "
                             f"central differences, but got length {N}.")

        # Allocate output tensor
        deriv = torch.empty_like(seq)
        h = spacing

        # --- Interior points: 4th-order central difference ---
        # Formula: f'(x) ≈ [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / (12h)
        # Applies to indices i = 2, 3, ..., N-3
        # Note: Using in-place operations (.add_, .div_) on slices can sometimes be
        # slightly faster by avoiding allocation of temporaries, but direct assignment
        # is often optimized well by PyTorch JIT / compilers anyway.
        # Readability might favor direct calculation:
        f_ip2 = seq[4:]       # f[i+2] for i in [2, N-3]
        f_ip1 = seq[3:-1]     # f[i+1] for i in [2, N-3]
        f_im1 = seq[1:-3]     # f[i-1] for i in [2, N-3]
        f_im2 = seq[0:-4]     # f[i-2] for i in [2, N-3]
        deriv[2:-2] = (-f_ip2 + 8*f_ip1 - 8*f_im1 + f_im2) / (12 * h)

        # --- Boundary points: 2nd-order differences ---

        # Left boundary (i=0): 2nd-order forward difference
        # Formula: f'(x₀) ≈ [-3f(x₀) + 4f(x₁) - f(x₂)] / (2h)
        deriv[0] = (-3 * seq[0] + 4 * seq[1] - seq[2]) / (2 * h)

        # Point i=1: 2nd-order central difference (can use this)
        # Formula: f'(x₁) ≈ [f(x₂) - f(x₀)] / (2h)
        deriv[1] = (seq[2] - seq[0]) / (2 * h)

        # Point i=N-2: 2nd-order central difference (can use this)
        # Formula: f'(x_{N-2}) ≈ [f(x_{N-1}) - f(x_{N-3})] / (2h)
        deriv[-2] = (seq[-1] - seq[-3]) / (2 * h)

        # Right boundary (i=N-1): 2nd-order backward difference
        # Formula: f'(x_{N-1}) ≈ [3f(x_{N-1}) - 4f(x_{N-2}) + f(x_{N-3})] / (2h)
        deriv[-1] = (3 * seq[-1] - 4 * seq[-2] + seq[-3]) / (2 * h)

        return deriv