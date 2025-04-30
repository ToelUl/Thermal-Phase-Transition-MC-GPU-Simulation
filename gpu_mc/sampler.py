import gc
import torch
from torch import Tensor
import torch.nn.functional as F

from .base import MonteCarloSampler

# =============================================================================
# Subclass: XYModel
# =============================================================================
class XYModel(MonteCarloSampler):
    r"""Implementation of the 2D XY model (including the Berezinskii-Kosterlitz-Thouless transition).

    This class retains the full functionality and performance optimizations (in-place updates, vectorization)
    of the original BKTXYSampler, with added support for adaptive updates.
    """

    def __init__(self,
                 L: int,
                 T: Tensor,
                 n_chains: int = 30,
                 J: float = 1.0,
                 device: torch.device = torch.device("cpu"),
                 max_delta: float = torch.pi,
                 adaptive: bool = False,
                 target_acceptance: float = 0.6,
                 adapt_rate: float = 0.1,
                 adapt_interval: int = 5,
                 ema_alpha: float = 0.1,
                 use_amp: bool = True,
                 large_size_simulate: bool = False,
                 pt_enabled: bool = True) -> None:
        """
        Args:
            L (int): Lattice size.
            T (Tensor): Temperature tensor.
            n_chains (int, optional): Number of chains. Defaults to 30.
            J (float, optional): Coupling constant. Defaults to 1.0.
            device (torch.device, optional): Compute device. Defaults to CPU.
            max_delta (float, optional): Maximum angular perturbation. Defaults to π.
            adaptive (bool, optional): Flag to enable adaptive max_delta adjustments. Defaults to False.
            target_acceptance (float, optional): Target acceptance rate for adaptive updates. Defaults to 0.6.
            adapt_rate (float, optional): Adaptation rate. Defaults to 0.1.
            adapt_interval (int, optional): Number of sweeps between adaptations. Defaults to 5.
            ema_alpha (float, optional): Exponential moving average factor for the acceptance rate. Defaults to 0.1.
            use_amp (bool, optional): Enable automatic mixed precision. Defaults to True.
            large_size_simulate (bool, optional): Flag for large-scale simulation. Defaults to False.
            pt_enabled (bool, optional): Enable parallel tempering. Defaults to True.
        """
        super().__init__(
            L=L,
            T=T,
            n_chains=n_chains,
            J=J,
            device=device,
            use_amp=use_amp,
            large_size_simulate=large_size_simulate,
            pt_enabled=pt_enabled,
        )
        # XY-model specific parameters for adaptive updates.
        self.adaptive = adaptive
        self.target_acceptance = target_acceptance
        self.adapt_rate = adapt_rate
        self.adapt_interval = adapt_interval
        self.ema_alpha = ema_alpha

        # Register the maximum angular perturbation as a buffer.
        self.register_buffer('max_delta', torch.tensor(max_delta, device=device, dtype=torch.float32))
        # Initialize counters for acceptance statistics.
        self.register_buffer('accept_count', torch.tensor(0, dtype=torch.int64, device=self.device))
        self.register_buffer('total_trials', torch.tensor(0, dtype=torch.int64, device=self.device))
        self.sweep_count = 0
        # Initialize the exponential moving average of the acceptance rate.
        self.register_buffer('ema_accept_rate', torch.tensor(self.target_acceptance, device=self.device, dtype=torch.float32))

    def init_spins(self) -> None:
        r"""Initialize the spin configuration for the XY model.

        Spins are represented as angles ∈ [0, 2π). The tensor shape is [batch_size, n_chains, L, L].
        """
        theta_init = 2 * torch.pi * torch.rand(
            (self.batch_size, self.n_chains, self.L, self.L),
            dtype=torch.float32,
            device=self.device
        )
        self.spins = theta_init

    def one_sweep(self, adaptive: bool = False) -> None:
        r"""Perform one sweep and adjust max_delta if adaptive updates are enabled.

        Overrides the base class one_sweep to include adaptive adjustments.

        Args:
            adaptive (bool, optional): Whether to use adaptive update parameters. Defaults to False.
        """
        super().one_sweep(adaptive=self.adaptive)
        if self.adaptive:
            self.sweep_count += 1
            if self.sweep_count % self.adapt_interval == 0:
                self.adjust_max_delta()

    def metropolis_update_sub_lattice(self, lattice_color: str, adaptive: bool = False) -> None:
        r"""Perform a Metropolis update on the specified sub-lattice for the XY model.

        For each site in the sub-lattice, an angular perturbation is applied, the energy difference is computed,
        and the new angle is accepted or rejected according to the Metropolis criterion.

        Args:
            lattice_color (str): 'black' or 'white' indicating which sub-lattice to update.
            adaptive (bool, optional): Flag for using adaptive parameters. Defaults to False.
        """
        T = self.T

        # Select indices based on the chosen sub-lattice.
        if lattice_color == 'black':
            batch_idx = self.black_batch_idx
            chain_idx = self.black_chain_idx
            i_sites = self.black_i_sites
            j_sites = self.black_j_sites
            neighbors_i = self.black_neighbors_i
            neighbors_j = self.black_neighbors_j
            N = self.num_black
        else:
            batch_idx = self.white_batch_idx
            chain_idx = self.white_chain_idx
            i_sites = self.white_i_sites
            j_sites = self.white_j_sites
            neighbors_i = self.white_neighbors_i
            neighbors_j = self.white_neighbors_j
            N = self.num_white

        theta_old = self.spins[batch_idx, chain_idx, i_sites, j_sites]

        # Generate random angular perturbations in the interval [-max_delta/2, max_delta/2].
        dtheta = torch.rand(self.batch_size, self.n_chains, N, device=self.device)
        dtheta.mul_(self.max_delta).add_(-self.max_delta / 2)
        theta_new = torch.remainder(theta_old + dtheta, 2 * torch.pi)

        # Retrieve neighbor angles.
        theta_neighbors = self.spins[
            batch_idx.unsqueeze(-1),
            chain_idx.unsqueeze(-1),
            neighbors_i,
            neighbors_j
        ]  # Shape: [B, C, N, 4]

        # Compute the energy difference ΔE for the perturbation.
        diff_new = theta_new.unsqueeze(-1) - theta_neighbors
        diff_old = theta_old.unsqueeze(-1) - theta_neighbors
        delta_cos = torch.cos(diff_new) - torch.cos(diff_old)
        delta_E = -self.J * delta_cos.sum(dim=-1)

        # Expand temperature tensor and compute acceptance probability.
        T_exp = T.view(self.batch_size, 1).expand(self.batch_size, self.n_chains)
        T_sites = T_exp[batch_idx, chain_idx]  # Shape: [B, C, N]
        p_acc = torch.exp(-delta_E / T_sites)

        rand_vals = torch.rand(self.batch_size, self.n_chains, N, device=self.device)
        accept_mask = (rand_vals < p_acc)

        if self.adaptive:
            self.accept_count.add_(accept_mask.sum())
            self.total_trials.add_(accept_mask.numel())

        # Update spins based on the acceptance decision.
        theta_updated = torch.where(accept_mask, theta_new, theta_old)
        self.spins[batch_idx, chain_idx, i_sites, j_sites] = theta_updated

        # Clean up intermediate variables.
        del dtheta, diff_new, diff_old, theta_updated, delta_cos, T_exp, delta_E, rand_vals, p_acc, theta_new, theta_old, accept_mask

    def adjust_max_delta(self) -> None:
        r"""Adaptively adjust the maximum angular perturbation (max_delta) based on the acceptance rate.

        The method uses an exponential moving average of the acceptance rate and modifies max_delta accordingly.
        """
        if self.total_trials == self.zero:
            return

        current_rate = self.accept_count.float() / self.total_trials.float()
        self.ema_accept_rate.mul_(1 - self.ema_alpha).add_(self.ema_alpha * current_rate)

        factor = torch.exp(self.adapt_rate * (self.ema_accept_rate - self.target_acceptance))
        new_max_delta = torch.clamp(self.max_delta * factor, min=1e-4, max=2 * torch.pi)
        self.max_delta.copy_(new_max_delta)

        self.accept_count.zero_()
        self.total_trials.zero_()
        del factor, new_max_delta, current_rate

    def compute_energy(self) -> Tensor:
        r"""Compute the total energy for the XY model.

        The energy is given by:
            E = -J * Σ cos(θ(i) - θ(neighbor))
        Only the right and down neighbors are considered to avoid double counting.

        Returns:
            Tensor: Energy tensor of shape [batch_size, n_chains].
        """
        theta = self.spins.to(self.device)
        t_top = torch.roll(theta, shifts=1, dims=3)
        t_right = torch.roll(theta, shifts=-1, dims=2)
        E_local = torch.cos(theta - t_top) + torch.cos(theta - t_right)
        E_batch = -self.J * E_local.sum(dim=(2, 3))

        del theta, t_top, t_right, E_local
        return E_batch

    def compute_average_energy(self) -> Tensor:
        r"""Compute the average energy per site for the XY model.

        Returns:
            Tensor: Average energy tensor.
        """
        return self.compute_energy().mean(dim=1) / self.L**2

    def compute_specific_heat_capacity(self) -> Tensor:
        r"""Compute the heat capacity per site for the XY model.
        The heat capacity is given by:
            C = (1 / T^2) * var(E)
        where var(E) is the variance of the energy.
        The factor of 1 / L^2 is included to normalize the heat capacity per site.

        Returns:
            Tensor: Heat capacity tensor.
        """
        T = self.T.to(self.device)
        c = torch.var(self.compute_energy(), dim=1) / T**2
        return c / self.L**2

    def compute_spin_stiffness(self) -> Tensor:
        r"""Compute the spin stiffness for the XY model.
        The spin stiffness is given by:
            ρ_s = (J * <cos(θ(i) - θ(i+1))> - (J^2 / T) * <sin(θ(i) - θ(i+1))^2>) / L^2
        where <...> denotes the average over all sites.
        The factor of 1 / L^2 is included to normalize the stiffness per site.

        Reference: https://arxiv.org/pdf/1101.3281#page=50

        Returns:
            Tensor: Spin stiffness tensor.
        """
        theta = self.spins.to(self.device)
        diff_y = torch.roll(theta, shifts=1, dims=3) - theta
        avg_links_y = torch.cos(diff_y).sum(dim=(2, 3)).mean(dim=1)
        avg_currents2_y = (torch.sin(diff_y).sum(dim=(2, 3)) ** 2).mean(dim=1)
        del diff_y, theta
        return (self.J * avg_links_y - (self.J**2 / self.T) * avg_currents2_y) / self.L**2

    def compute_magnetization(self) -> Tensor:
        r"""Compute the magnetization per site for the XY model.
        The magnetization is given by:
            m = (1 / L^2) * Σ (m_x^2 + m_y^2)^(1/2)
        where the sum is over all lattice sites.

        Reference: https://iopscience.iop.org/article/10.1088/0953-8984/4/24/011

        Returns:
            Tensor: Magnetization tensor.
        """
        theta = self.spins.to(self.device)
        mx = torch.cos(theta).sum(dim=(2, 3)).unsqueeze(-1)
        my = torch.sin(theta).sum(dim=(2, 3)).unsqueeze(-1)
        m = torch.stack([mx, my], dim=2).norm(dim=2).squeeze(dim=2)
        del theta, mx, my
        return m.mean(dim=1) / self.L**2

    def compute_susceptibility(self) -> Tensor:
        r"""Compute the susceptibility per site for the XY model.
        The susceptibility is given by:
            χ = (1 / T) * var(m)
        where m is the magnetization per site.

        Returns:
            Tensor: Susceptibility tensor.
        """
        theta = self.spins.to(self.device)
        mx = torch.cos(theta).sum(dim=(2, 3)).unsqueeze(-1)
        my = torch.sin(theta).sum(dim=(2, 3)).unsqueeze(-1)
        m = torch.stack([mx, my], dim=2).norm(dim=2).squeeze(dim=2)
        del theta, mx, my
        return m.var(dim=1) * (1.0 / self.T) / self.L**2

    def _principal_value(self, delta: Tensor) -> Tensor:
        """Map angle to range [−π,π].

        Arg:
            delta: Tensor of shape [num_temp, num_samples, H, W].

        Returns:
            Tensor of shape [num_temp, num_samples, H, W]. Elements ∈ [−π,π]
        """
        return (delta + torch.pi) % (2*torch.pi) - torch.pi

    def compute_vortex_density(self) -> Tensor:
        """Compute the vortex density for the XY model.
        The vortex density is computed using the formula:
            ρ_v = (1 / L^2) * Σ |ω(i,j)| / (2π)
        where ω(i,j) is the vorticity tensor at site (i,j).
        The sum is over all lattice sites, and the factor of 1 / L^2 is included to normalize the density per site.

        Reference: https://arxiv.org/pdf/2207.13748#page=20

        Returns:
            Tensor: Vortex density tensor.
        """
        theta = self.spins.to(self.device)
        theta_ip = torch.roll(theta, shifts=-1, dims=-2)      # i+1, j
        theta_jp = torch.roll(theta, shifts=-1, dims=-1)      # i, j+1
        theta_ipp_jp = torch.roll(theta_ip, shifts=-1, dims=-1)  # i+1, j+1

        d1 = self._principal_value(delta=theta_jp     - theta)         # (i,j)->(i,j+1)
        d2 = self._principal_value(delta=theta_ipp_jp - theta_jp)      # (i,j+1)->(i+1,j+1)
        d3 = self._principal_value(delta=theta_ip     - theta_ipp_jp)  # (i+1,j+1)->(i+1,j)
        d4 = self._principal_value(delta=theta        - theta_ip)      # (i+1,j)->(i,j)

        omega = d1 + d2 + d3 + d4  # ∈ [−4π,4π]
        q = torch.round(omega / (2*torch.pi)).to(torch.int32)
        del theta, theta_ip, theta_jp, theta_ipp_jp, d1, d2, d3, d4, omega
        gc.collect()
        return q.to(torch.float).abs().mean(dim=(1, 2, 3))


# =============================================================================
# Subclass: IsingModel
# =============================================================================
class IsingModel(MonteCarloSampler):
    r"""Implementation of the 2D Ising model using a spin-flip Metropolis update.

    The update is performed using a checkerboard sub-lattice approach.
    """

    def __init__(self,
                 L: int,
                 T: Tensor,
                 n_chains: int = 30,
                 J: float = 1.0,
                 device: torch.device = torch.device("cpu"),
                 use_amp: bool = True,
                 large_size_simulate: bool = False,
                 pt_enabled: bool = False) -> None:
        """
        Args:
            L (int): Lattice size.
            T (Tensor): Temperature tensor.
            n_chains (int, optional): Number of chains. Defaults to 30.
            J (float, optional): Coupling constant. Defaults to 1.0.
            device (torch.device, optional): Compute device. Defaults to CPU.
            use_amp (bool, optional): Enable automatic mixed precision. Defaults to True.
            large_size_simulate (bool, optional): Flag for large-scale simulation. Defaults to False.
            pt_enabled (bool, optional): Enable parallel tempering. Defaults to False.
        """
        super().__init__(
            L=L,
            T=T,
            n_chains=n_chains,
            J=J,
            device=device,
            use_amp=use_amp,
            large_size_simulate=large_size_simulate,
            pt_enabled=pt_enabled,
        )

    def init_spins(self) -> None:
        r"""Initialize spins for the Ising model.

        Spins take values ±1. The resulting tensor has shape [batch_size, n_chains, L, L].
        """
        spins_init = torch.randint(0, 2, (self.batch_size, self.n_chains, self.L, self.L),
                                    device=self.device, dtype=torch.float32)
        # Map {0,1} to {-1,+1}
        spins_init = 2 * spins_init - 1
        self.spins = spins_init

    def metropolis_update_sub_lattice(self, lattice_color: str, adaptive: bool = False) -> None:
        r"""Perform the Metropolis update on the specified sub-lattice for the Ising model.

        The update consists of attempting to flip the spin at each site and accepting the flip
        based on the computed energy difference.

        Args:
            lattice_color (str): 'black' or 'white' indicating which sub-lattice to update.
            adaptive (bool, optional): Not used for the Ising update. Defaults to False.
        """
        T = self.T

        if lattice_color == 'black':
            batch_idx = self.black_batch_idx
            chain_idx = self.black_chain_idx
            i_sites = self.black_i_sites
            j_sites = self.black_j_sites
            neighbors_i = self.black_neighbors_i
            neighbors_j = self.black_neighbors_j
            N = self.num_black
        else:
            batch_idx = self.white_batch_idx
            chain_idx = self.white_chain_idx
            i_sites = self.white_i_sites
            j_sites = self.white_j_sites
            neighbors_i = self.white_neighbors_i
            neighbors_j = self.white_neighbors_j
            N = self.num_white

        s_old = self.spins[batch_idx, chain_idx, i_sites, j_sites]
        s_new = -s_old

        # Retrieve neighboring spins.
        s_neighbors = self.spins[
            batch_idx.unsqueeze(-1),
            chain_idx.unsqueeze(-1),
            neighbors_i,
            neighbors_j
        ]  # Shape: [B, C, N, 4]

        # Compute energy difference ΔE = -J * (s_new * Σ(neighbors) - s_old * Σ(neighbors))
        sum_neighbors = s_neighbors.sum(dim=-1)
        delta_E = -self.J * (s_new * sum_neighbors - s_old * sum_neighbors)

        T_exp = T.view(self.batch_size, 1).expand(self.batch_size, self.n_chains)
        T_sites = T_exp[batch_idx, chain_idx]  # Shape: [B, C, N]
        p_acc = torch.exp(-delta_E / T_sites)

        rand_vals = torch.rand(self.batch_size, self.n_chains, N, device=self.device)
        accept_mask = (rand_vals < p_acc)

        s_updated = torch.where(accept_mask, s_new, s_old)
        self.spins[batch_idx, chain_idx, i_sites, j_sites] = s_updated

        del s_new, s_old, s_neighbors, sum_neighbors, T_exp, delta_E, rand_vals, p_acc, accept_mask, s_updated

    def compute_energy(self) -> Tensor:
        r"""Compute the energy of the Ising model.

        The energy is computed as:
            E = -J * Σ( s(i) * s(up) + s(i) * s(right) )
        where only the top and right neighbors are considered to avoid double counting.

        Returns:
            Tensor: Energy tensor with shape [batch_size, n_chains].
        """
        s = self.spins.to(self.device)
        s_up = torch.roll(s, shifts=1, dims=2)
        s_right = torch.roll(s, shifts=-1, dims=3)
        E_local = s * s_up + s * s_right
        E_batch = -self.J * E_local.sum(dim=(2, 3))

        del s, s_up, s_right, E_local
        return E_batch

    def compute_average_energy(self) -> Tensor:
        r"""Compute the average energy per site for the Ising model.

        Returns:
            Tensor: Average energy tensor.
        """
        return self.compute_energy().mean(dim=1) / self.L**2

    def compute_specific_heat_capacity(self) -> Tensor:
        r"""Compute the heat capacity per site for the Ising model.
        The heat capacity is given by:
            C = (1 / T^2) * var(E)
        where var(E) is the variance of the energy.
        The factor of 1 / L^2 is included to normalize the heat capacity per site.

        Returns:
            Tensor: Heat capacity tensor.
        """
        c = torch.var(self.compute_energy(), dim=1) / self.T.to(self.device)**2
        return c / self.L**2

    def compute_magnetization(self) -> Tensor:
        r"""Compute the magnetization per site for the Ising model.
        The magnetization is given by:
            m = (1 / L^2) * Σ s(i)
        where the sum is over all lattice sites.

        Returns:
            Tensor: Magnetization tensor.
        """
        return self.spins.to(self.device).mean(dim=(2, 3)).abs().mean(dim=1)

    def compute_susceptibility(self) -> Tensor:
        r"""Compute the susceptibility per site for the Ising model.
        The susceptibility is given by:
            χ = (1 / T) * var(m)
        where m is the magnetization per site.

        Returns:
            Tensor: Susceptibility tensor.
        """
        return self.spins.to(self.device).mean(dim=(2, 3)).abs().var(dim=1) / self.T.to(self.device)

    def compute_binder_cumulant(self) -> Tensor:
        r"""Compute the Binder cumulant for the Ising model.

        The Binder cumulant is given by:
            U_4 = 1 - (1 / 3) * (《m^4》 / 《m^2》^2)
        where m^2 is the square of the magnetization and m^4 is the fourth moment of the magnetization.

        Returns:
            Tensor: Binder cumulant tensor.
        """
        m2 = self.spins.to(self.device).mean(dim=(2, 3)).pow(2).mean(dim=1)
        m4 = self.spins.to(self.device).mean(dim=(2, 3)).pow(4).mean(dim=1)
        return 1.0 - m4.div(3.0 * m2 * m2)

    def compute_domain_wall_density(self) -> Tensor:
        r"""Compute the domain wall density for the Ising model.

        The domain wall density is given by:
            ρ_dw = ⟨(1−s_i s_j)/2⟩
        where the average is taken over all pairs of neighboring spins.

        Returns:
            Tensor: Domain wall density tensor.
        """
        spins = self.spins.to(self.device)
        shift_x = spins.roll(shifts=-1, dims=3)
        shift_y = spins.roll(shifts=1, dims=2)
        dw_x = (1 - spins * shift_x) / 2
        dw_y = (1 - spins * shift_y) / 2
        dw_density = (dw_x + dw_y).mean(dim=(2, 3))
        return dw_density.mean(dim=1)

    def compute_exact_magnetization(self) -> Tensor:
        r"""Compute the exact spontaneous magnetization for the Ising model.

        The spontaneous magnetization is given by:
            m = (1 - sinh(2 * J / T)^(-4))^(1/8)
        This formula is valid for T < 2J / log(1 + sqrt(2)).
        The function returns 0 for T >= 2J / log(1 + sqrt(2)).

        Reference: https://journals.aps.org/pr/abstract/10.1103/PhysRev.85.808

        Returns:
            Tensor: Exact spontaneous magnetization tensor.
        """
        k = self.J / self.T
        kc = 0.5 * torch.log(1 + torch.sqrt(2 * torch.ones_like(self.T)))
        m = (1 - (1/torch.sinh(2 * k)).pow(4)).pow(0.125)
        return m.masked_fill_(k<=kc, 0.0)


# =============================================================================
# Subclass: PottsModel
# =============================================================================
class PottsModel(MonteCarloSampler):
    r"""Implementation of the 2D q-state Potts model.

    In the Potts model, spins are integer states in {0, 1, ..., q-1} and the energy is given by:
      E = -J * Σ δ(s(i), s(j)),
    where δ is the Kronecker delta (only contributing when neighboring spins are equal).
    """

    def __init__(self,
                 L: int,
                 T: Tensor,
                 q: int = 3,
                 n_chains: int = 1,
                 J: float = 1.0,
                 device: torch.device = torch.device("cpu"),
                 use_amp: bool = False,
                 large_size_simulate: bool = False,
                 pt_enabled: bool = False) -> None:
        """
        Args:
            L (int): Lattice size.
            T (Tensor): Temperature tensor.
            q (int, optional): Number of states in the Potts model. Defaults to 3.
            n_chains (int, optional): Number of chains. Defaults to 1.
            J (float, optional): Coupling constant. Defaults to 1.0.
            device (torch.device, optional): Compute device. Defaults to CPU.
            use_amp (bool, optional): Enable automatic mixed precision. Defaults to False.
            large_size_simulate (bool, optional): Flag for large-scale simulation. Defaults to False.
            pt_enabled (bool, optional): Enable parallel tempering. Defaults to False.
        """
        if q < 2:
            raise ValueError("q must be ≥2.")
        self.q = q

        super().__init__(
            L=L,
            T=T,
            n_chains=n_chains,
            J=J,
            device=device,
            use_amp=use_amp,
            large_size_simulate=large_size_simulate,
            pt_enabled=pt_enabled,
        )

    def init_spins(self) -> None:
        r"""Initialize the spin configuration for the Potts model.

        Spins take integer values in the range [0, q-1]. The tensor shape is [batch_size, n_chains, L, L].
        """
        spins_init = torch.randint(
            0, self.q,
            (self.batch_size, self.n_chains, self.L, self.L),
            device=self.device,
            dtype=torch.int64
        )
        self.spins = spins_init

    def metropolis_update_sub_lattice(self, lattice_color: str, adaptive: bool = False) -> None:
        r"""Perform the Metropolis update on the specified sub-lattice for the Potts model.

        For each site in the sub-lattice, a new state (different from the current state) is proposed.
        The energy difference is computed using the Kronecker delta and the move is accepted/rejected based on
        the Metropolis criterion.

        Args:
            lattice_color (str): 'black' or 'white' indicating which sub-lattice to update.
            adaptive (bool, optional): Not used for the Potts update. Defaults to False.
        """
        T = self.T

        if lattice_color == 'black':
            batch_idx = self.black_batch_idx
            chain_idx = self.black_chain_idx
            i_sites = self.black_i_sites
            j_sites = self.black_j_sites
            neighbors_i = self.black_neighbors_i
            neighbors_j = self.black_neighbors_j
            N = self.num_black
        else:
            batch_idx = self.white_batch_idx
            chain_idx = self.white_chain_idx
            i_sites = self.white_i_sites
            j_sites = self.white_j_sites
            neighbors_i = self.white_neighbors_i
            neighbors_j = self.white_neighbors_j
            N = self.num_white

        s_old = self.spins[batch_idx, chain_idx, i_sites, j_sites]

        # Propose a new state different from the current state.
        s_rand = torch.randint(
            0, self.q,
            (self.batch_size, self.n_chains, N),
            device=self.device,
            dtype=torch.int64
        )
        s_new = torch.where(s_rand == s_old, (s_rand + 1) % self.q, s_rand)

        # Retrieve neighbor spins.
        s_neighbors = self.spins[
            batch_idx.unsqueeze(-1),
            chain_idx.unsqueeze(-1),
            neighbors_i,
            neighbors_j
        ]  # Shape: [B, C, N, 4]

        # Compute the energy difference:
        # E = -J * Σ δ(s, neighbor)  -->  ΔE = -J * [Σ δ(s_new, neighbor) - Σ δ(s_old, neighbor)]
        matches_old = (s_neighbors == s_old.unsqueeze(-1)).sum(dim=-1)
        matches_new = (s_neighbors == s_new.unsqueeze(-1)).sum(dim=-1)
        delta_E = -self.J * (matches_new - matches_old).float()

        T_exp = T.view(self.batch_size, 1).expand(self.batch_size, self.n_chains)
        T_sites = T_exp[batch_idx, chain_idx]  # Shape: [B, C, N]
        p_acc = torch.exp(-delta_E / T_sites)

        rand_vals = torch.rand(self.batch_size, self.n_chains, N, device=self.device)
        accept_mask = (rand_vals < p_acc)

        s_updated = torch.where(accept_mask, s_new, s_old)
        self.spins[batch_idx, chain_idx, i_sites, j_sites] = s_updated

        del s_rand, s_new, s_old, s_neighbors, matches_old, matches_new, delta_E, T_exp, rand_vals, p_acc, accept_mask, s_updated

    def compute_energy(self) -> Tensor:
        r"""Compute the energy of the Potts model.

        The energy is given by:
            E = -J * Σ( δ(s(i), s(up)) + δ(s(i), s(right)) )
        Only the top and right neighbors are considered to avoid double counting.

        Returns:
            Tensor: Energy tensor with shape [batch_size, n_chains].
        """
        s = self.spins.to(self.device)
        s_up = torch.roll(s, shifts=1, dims=2)
        s_right = torch.roll(s, shifts=-1, dims=3)

        match_up = (s == s_up).float()
        match_right = (s == s_right).float()

        E_local = match_up + match_right
        E_batch = -self.J * E_local.sum(dim=(2, 3))

        del s, s_up, s_right, match_up, match_right, E_local
        return E_batch

    def compute_average_energy(self) -> Tensor:
        r"""Compute the average energy per site for the Potts model.

        Returns:
            Tensor: Average energy tensor.
        """
        return self.compute_energy().mean(dim=1) / self.L**2

    def compute_specific_heat_capacity(self) -> Tensor:
        r"""Compute the heat capacity per site for the Potts model.
        The heat capacity is given by:
            C = (1 / T^2) * var(E)
        where var(E) is the variance of the energy.
        The factor of 1 / L^2 is included to normalize the heat capacity per site.

        Returns:
            Tensor: Heat capacity tensor with shape [batch_size].
        """
        T = self.T.to(self.device)
        energy = self.compute_energy()  # Shape: [batch_size, n_chains]
        c = torch.var(energy, dim=1) / (T**2)
        return c / self.L**2

    def compute_magnetization(self) -> Tensor:
        r"""Compute the magnetization per site for the Potts model.
        The magnetization is given by:
            n_α = (1 / L^2) * Σ delta(s(i), s(j))
            Σ n_α = 1
            m = (q * n_α_max - 1) / (q - 1)
        where the sum is over all lattice sites and n_α_max is the maximum occupancy fraction.

        Returns:
            Tensor: Magnetization tensor with shape [batch_size].
        """
        spins = self.spins.to(self.device)
        # One-hot encode → shape (*batch, *dims, q)
        one_hot = F.one_hot(spins, num_classes=self.q).to(torch.float32)
        # Occupancy fractions along spatial dims
        n_alpha = one_hot.mean(dim=(2, 3))  # (*batch, q)
        # Max-colour definition
        m_scalar = (self.q * n_alpha.max(dim=-1).values - 1) / (self.q - 1)
        return m_scalar.mean(dim=1)

    def compute_susceptibility(self) -> Tensor:
        r"""Compute the susceptibility per site for the Potts model.
        The susceptibility is given by:
            χ = (1 / T) * var(m)
        where m is the magnetization per site.

        Returns:
            Tensor: Susceptibility tensor.
        """
        spins = self.spins.to(self.device)
        n_alpha = F.one_hot(spins, num_classes=self.q).to(torch.float32).mean(dim=(2, 3))
        m_scalar = (self.q * n_alpha.max(dim=-1).values - 1) / (self.q - 1)
        return m_scalar.var(dim=1) / self.T.to(self.device)

    def compute_binder_cumulant(self) -> Tensor:
        r"""Compute the Binder cumulant for the Potts model.

        The Binder cumulant is given by:
            U_4 = 1 - (1 / 3) * (《m^4》 / 《m^2》^2)
        where m^2 is the square of the magnetization and m^4 is the fourth moment of the magnetization.

        Returns:
            Tensor: Binder cumulant tensor.
        """
        spins = self.spins.to(self.device)
        n_alpha = F.one_hot(spins, num_classes=self.q).to(torch.float32).mean(dim=(2, 3))
        m_scalar = (self.q * n_alpha.max(dim=-1).values - 1) / (self.q - 1)
        m2 = m_scalar.pow(2).mean(dim=1)
        m4 = m_scalar.pow(4).mean(dim=1)
        return 1.0 - m4.div(3.0 * m2 * m2)

    def compute_entropy(self) -> Tensor:
        r"""Compute the entropy per site for the Potts model.

        The entropy is given by:
            S = -Σ p_i * log(p_i)
        where p_i is the probability of each state.

        Returns:
            Tensor: Entropy tensor with shape [batch_size].
        """
        spins = self.spins.to(self.device)
        n_alpha = F.one_hot(spins, num_classes=self.q).to(torch.float32).mean(dim=(2, 3))
        entropy = -torch.sum(n_alpha * torch.log(n_alpha + 1e-10), dim=-1)
        return entropy.mean(dim=1)
