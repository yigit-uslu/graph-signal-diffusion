"""
Wireless channel model for resource allocation problems.

This module implements a realistic wireless channel model with:
- Random spatial deployment of transmitters and receivers
- Large-scale fading (path loss + log-normal shadowing)
- Small-scale fading (Rayleigh fading over time)
- Optimal TX-RX pairing based on channel gain
"""

import hashlib
import numpy as np
from scipy.spatial import distance
from typing import Optional, Dict, Tuple
import abc

import logging

log = logging.getLogger(__name__)

class WirelessChannel(abc.ABC):
    """
    Wireless channel model for multi-user interference networks.
    
    The channel generates realistic wireless propagation scenarios by:
    1. Deploying transmitters and receivers uniformly in a square area
    2. Computing large-scale fading (path loss + shadowing) - static per realization
    3. Pairing each transmitter to the best receiver based on channel gain
    4. Generating time-varying small-scale fading (Rayleigh)
    
    Parameters
    ----------
    n_links : int
        Number of transmitter-receiver pairs (users)
    deployment_range : float, optional
        Side length of square deployment area in meters (default: 1000m)
    min_tx_tx_distance : float, optional
        Minimum distance between transmitters in meters (default: 20m)
    min_tx_rx_distance : float, optional
        Minimum distance between paired TX-RX in meters (default: 10m)
    max_tx_rx_distance : float or None, optional
        Maximum distance between paired TX-RX in meters (default: 50m).
        Set to None for no upper distance constraint.
    path_loss_exponent_short : float, optional
        Path loss exponent for distances <= 100m (default: 2.0)
    path_loss_exponent_long : float, optional
        Path loss exponent for distances > 100m (default: 4.0)
    shadowing_std : float, optional
        Standard deviation of log-normal shadowing in dB (default: 7.0)
    carrier_freq : float, optional
        Carrier frequency in Hz (default: 2.4 GHz)
    speed : float, optional
        Receiver speed in m/s for Doppler (default: 1.0 m/s)
    num_fading_paths : int, optional
        Number of paths in Rayleigh fading model (default: 100)
    delta_t : float, optional
        Time interval between channel realizations in seconds (default: 0.01s = 10ms)
    seed : int, optional
        Random seed for reproducibility (default: None)
    skip_deployment : bool, optional
        If True, skip automatic deployment and expect manual state setting (default: False)
    max_deployment_attempts : int, optional
        Maximum attempts for network deployment (default: 1000)
    tx_locations : np.ndarray, optional
        Pre-placed TX locations (n_links x 2) for skip_deployment=True (default: None)
    rx_locations : np.ndarray, optional
        Pre-placed RX locations (n_links x 2) for skip_deployment=True (default: None)
    
    Notes
    -----
    Initialization modes:
    1. Normal deployment (skip_deployment=False):
       - Deploys TX/RX using global seed
       - Computes shadowing and optimal pairing (uses RNG after deployment)
       - Fully reproducible with same seed
    
    2. Pre-placed locations (skip_deployment=True, tx_locations, rx_locations):
       - Uses provided locations (no deployment)
       - Generates **local seed** from hash of locations (deterministic)
       - Computes shadowing with local seed (always same for same locations)
       - Computes optimal pairing with local seed (always same for same locations)
       - Result: Same locations → Same shadowing → Same associations
       - Optimal pairing won't change locations (already optimal with same seed)
       - Use case: Reproducible merge_channels, save/load channel by locations
    """
    
    def __init__(
        self,
        n_links: int,
        deployment_range: float = 1000.0,
        min_tx_tx_distance: float = 35.0,
        min_tx_rx_distance: float = 10.0,
        max_tx_rx_distance: Optional[float] = 50.0,
        path_loss_exponent_short: float = 2.0,
        path_loss_exponent_long: float = 4.0,
        shadowing_std: float = 7.0,
        carrier_freq: float = 2.4e9,
        speed: float = 1.0,
        num_fading_paths: int = 100,
        delta_t: float = 0.01,
        seed: Optional[int] = None,
        skip_deployment: bool = False,
        max_deployment_attempts: int = 1000,
        tx_locations: Optional[np.ndarray] = None,
        rx_locations: Optional[np.ndarray] = None,
    ):
        self.n_links = n_links
        self.deployment_range = deployment_range
        self.min_tx_tx_distance = min_tx_tx_distance
        self.min_tx_rx_distance = min_tx_rx_distance
        self.max_tx_rx_distance = max_tx_rx_distance
        self.path_loss_exponent_short = path_loss_exponent_short
        self.path_loss_exponent_long = path_loss_exponent_long
        self.shadowing_std = shadowing_std
        self.carrier_freq = carrier_freq
        self.speed = speed
        self.num_fading_paths = num_fading_paths
        self.delta_t = delta_t
        self.seed = seed
        self.max_deployment_attempts = max_deployment_attempts
        
        # Use a local RandomState so channel construction never touches the
        # global numpy RNG (avoiding subtle reproducibility bugs when multiple
        # channels are created or other code calls np.random in between).
        self._rng = np.random.RandomState(seed)

        # Handle different initialization paths
        if skip_deployment:
            # Pre-placed locations: use local seed for deterministic shadowing/pairing
            if tx_locations is not None and rx_locations is not None:
                self.tx_locations = tx_locations
                self.rx_locations = rx_locations
                self.shadowing_db_deployment = None
                self._is_preplaced = True  # Flag to skip surgical redeployment

                # Derive a stable local seed from locations so the same locations
                # always produce the same shadowing/pairing regardless of call order.
                local_seed = self._generate_local_seed(tx_locations, rx_locations)
                self._rng = np.random.RandomState(local_seed)
                
                # Compute shadowing and pairing with local seed
                # Since locations are pre-placed (already optimal), we skip surgical redeployment
                self._compute_large_scale_fading()
                self._assign_optimal_pairing()
                
                log.info(f"✓ Pre-placed network: {self.n_links} links (deterministic shadowing/pairing)")
            else:
                # No locations provided - this is an error for skip_deployment=True
                raise ValueError(
                    "When skip_deployment=True, you must provide both tx_locations and rx_locations"
                )
        else:
            # Normal deployment path
            self._is_preplaced = False
            self._deploy_network()
            self._compute_large_scale_fading()
            self._assign_optimal_pairing()
    
    def _generate_local_seed(self, tx_locations: np.ndarray, rx_locations: np.ndarray) -> int:
        """
        Generate deterministic seed from TX/RX locations.
        
        This ensures that the same locations always produce the same shadowing
        and optimal pairing, making the channel fully reproducible from locations alone.
        
        Parameters
        ----------
        tx_locations : np.ndarray
            TX locations (n_links x 2)
        rx_locations : np.ndarray
            RX locations (n_links x 2)
        
        Returns
        -------
        int
            Deterministic seed in range [0, 2^31-1]
        """
        # Concatenate and flatten locations
        locations = np.concatenate([tx_locations, rx_locations]).flatten()
        
        # Round to avoid floating point precision issues
        locations_rounded = np.round(locations, decimals=6)
        
        # Convert to bytes and hash (sha256 is stable across Python sessions,
        # unlike the built-in hash() which is randomised by PYTHONHASHSEED).
        locations_bytes = locations_rounded.tobytes()
        digest = hashlib.sha256(locations_bytes).digest()

        # Map to a positive 32-bit integer for np.random.RandomState.
        seed = int.from_bytes(digest[:4], "little") & 0x7FFF_FFFF

        return seed
    
    def _deploy_network(self) -> None:
        """Deploy network, with soft-failure fallback for small networks.

        V2's ``_assign_optimal_pairing`` relies on this fallback: when
        deployment fails for small networks (n_links <= 200), it proceeds
        with degraded associations and surgically redeploys problematic RXs.
        """
        try:
            self._try_deploy()
        except RuntimeError:
            if self.n_links > 200:
                raise
            log.warning(
                f"Could not satisfy interference criterion for {self.n_links} links "
                f"after {self.max_deployment_attempts} attempts. Proceeding anyway."
            )
            self.shadowing_db_deployment = None

    def _try_deploy(self) -> None:
        """
        Deploy transmitters and receivers in the deployment area.

        Always raises ``RuntimeError`` on failure so that callers
        (V1's ``_deploy_network``, V3's recursive subdivision) can
        decide how to handle it.

        This method ensures:
        1. TX-TX minimum separation constraint
        2. TX-RX minimum distance constraint
        3. Each TX dominates at least one RX (strongest signal to that RX)

        The third condition is critical for wireless resource allocation:
        it ensures each transmitter can communicate with at least one receiver
        where it has the strongest signal (dominates over all interferers).
        """
        R = self.deployment_range
        m = self.n_links

        max_deployment_attempts = self.max_deployment_attempts
        tx_attempt = 0

        # Deploy transmitters with minimum separation constraint
        log.info(f"Deploying {m} transmitters...")
        while tx_attempt < max_deployment_attempts:
            tx_attempt += 1

            # Random uniform deployment in [-R/2, R/2] x [-R/2, R/2]
            self.tx_locations = self._rng.uniform(-R/2, R/2, (m, 2))

            # Check minimum TX-TX separation
            tx_tx_distances = distance.cdist(self.tx_locations, self.tx_locations, 'euclidean')
            np.fill_diagonal(tx_tx_distances, np.inf)  # Ignore self-distances

            if np.min(tx_tx_distances) >= self.min_tx_tx_distance:
                log.info(f"✓ TX deployment successful (attempt {tx_attempt})")
                break
        else:
            raise RuntimeError(
                f"Could not deploy {m} transmitters with {self.min_tx_tx_distance}m minimum "
                f"separation in {R}x{R}m area after {max_deployment_attempts} attempts. "
                f"Try: (1) increasing deployment_range, (2) decreasing min_tx_tx_distance, "
                f"or (3) using subnetwork merging for large networks."
            )

        # Deploy receivers with interference criterion
        log.info(f"Deploying {m} receivers with interference constraints...")
        attempt = 0
        while attempt < max_deployment_attempts:
            attempt += 1

            # Random angle and radius for each receiver relative to corresponding TX
            phi = 2 * np.pi * self._rng.uniform(size=(m,))
            r = self._sample_tx_rx_radii(size=(m,))

            # Place receivers relative to transmitters, clipped to deployment area
            self.rx_locations = self.tx_locations + np.stack((r * np.cos(phi), r * np.sin(phi)), axis=1)
            self.rx_locations = np.clip(self.rx_locations, -R/2, R/2)

            # Check minimum TX-RX distance constraint
            tx_rx_distances = distance.cdist(self.tx_locations, self.rx_locations, 'euclidean')

            if np.min(tx_rx_distances) < self.min_tx_rx_distance:
                continue

            # Compute temporary large-scale fading to check interference conditions
            path_loss_db = self._compute_path_loss(tx_rx_distances)
            shadowing_db = self.shadowing_std * self._rng.randn(*tx_rx_distances.shape)
            total_loss_db = path_loss_db + shadowing_db
            H_l_temp = np.sqrt(10 ** (-total_loss_db / 10))

            # Check interference criterion: Each TX must dominate at least one RX
            # associations[i,j] = True if TX_i has strongest signal to RX_j
            associations = (H_l_temp == np.max(H_l_temp, axis=0, keepdims=True))

            # Each transmitter must have at least one receiver where it's dominant
            if np.min(np.sum(associations, axis=1)) > 0:
                # Success! Store the shadowing for this valid deployment
                self.shadowing_db_deployment = shadowing_db
                log.info(f"✓ Network deployed: {m} TX-RX pairs in {R}x{R}m area (attempt {attempt})")
                return

        raise RuntimeError(
            f"Could not satisfy interference dominance criterion for {m} links after "
            f"{max_deployment_attempts} attempts. "
            f"Try: (1) increasing deployment_range, (2) decreasing min_tx_tx_distance, "
            f"or (3) using subnetwork merging for large networks."
        )

    def _sample_tx_rx_radii(self, size=None):
        """Sample TX-relative RX radii with optional max-distance constraint.

        When ``max_tx_rx_distance`` is None, sampling uses the deployment square's
        diagonal as a finite proposal radius and downstream clipping keeps points
        inside the deployment region.
        """
        low_sq = float(self.min_tx_rx_distance) ** 2
        if self.max_tx_rx_distance is None:
            # Maximum possible distance between two points in the square deployment area.
            max_radius = float(np.sqrt(2.0) * self.deployment_range)
            high_sq = max_radius ** 2
        else:
            high_sq = float(self.max_tx_rx_distance) ** 2

        if high_sq < low_sq:
            raise ValueError(
                "max_tx_rx_distance must be >= min_tx_rx_distance, or None for no constraint. "
                f"Got min_tx_rx_distance={self.min_tx_rx_distance}, "
                f"max_tx_rx_distance={self.max_tx_rx_distance}."
            )

        return np.sqrt(self._rng.uniform(low=low_sq, high=high_sq, size=size))
    

    def _compute_path_loss(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute path loss using dual-slope model.
        
        Parameters
        ----------
        distances : np.ndarray
            Distance matrix (m x n)
        
        Returns
        -------
        path_loss_db : np.ndarray
            Path loss in dB
        """
        k0 = 39.0  # Reference loss at 1m
        d_breakpoint = 100.0  # Breakpoint distance

        distances = np.asarray(distances, dtype=float)

        # Vectorized dual-slope path loss model, continuous at the breakpoint.
        path_loss_breakpoint = k0 + 10 * self.path_loss_exponent_short * np.log10(d_breakpoint)
        short_regime = k0 + 10 * self.path_loss_exponent_short * np.log10(distances)
        long_regime = path_loss_breakpoint + 10 * self.path_loss_exponent_long * np.log10(
            distances / d_breakpoint
        )
        return np.where(distances <= d_breakpoint, short_regime, long_regime)
    
    def _compute_large_scale_fading(self) -> None:
        """
        Compute large-scale fading coefficients (path loss + shadowing).
        
        This is static per network realization and represents the average
        channel gain over time.
        
        Note: If shadowing was already computed during deployment (to check
        interference conditions), we reuse it to maintain consistency.
        """
        # Compute all TX-RX distances
        self.distances = distance.cdist(self.tx_locations, self.rx_locations, 'euclidean')
        
        # Compute path loss
        path_loss_db = self._compute_path_loss(self.distances)
        
        # Use shadowing from deployment if available (ensures consistency)
        # Otherwise generate new shadowing
        if hasattr(self, 'shadowing_db_deployment') and self.shadowing_db_deployment is not None:
            shadowing_db = self.shadowing_db_deployment
        else:
            shadowing_db = self.shadowing_std * self._rng.randn(*self.distances.shape)
        
        # Total loss in dB
        total_loss_db = path_loss_db + shadowing_db
        
        # Convert to linear scale (amplitude)
        self.large_scale_fading = np.sqrt(10 ** (-total_loss_db / 10))
        
        # Store components for analysis
        self.path_loss_db = path_loss_db
        self.shadowing_db = shadowing_db
        
        log.info(f"✓ Large-scale fading computed (mean: {np.mean(total_loss_db):.1f} dB)")
    
    def _assign_optimal_pairing(self) -> None:
        """
        Assign TX-RX pairing based on interference dominance.
        
        Pairing strategy:
        1. For each RX, find which TX has the strongest signal (dominates)
        2. This creates associations where interference is minimized
        3. Each TX should dominate at least one RX (enforced during deployment)
        
        The resulting pairing ensures that each transmitter communicates with
        a receiver where it has the strongest channel gain, minimizing the
        impact of inter-user interference.
        """
        m, n = self.large_scale_fading.shape
        
        # For each receiver (column), find which transmitter has the strongest signal
        # associations[i,j] = True if TX_i has the strongest signal to RX_j
        self.associations = (
            self.large_scale_fading == np.max(self.large_scale_fading, axis=0, keepdims=True)
        )
        
        # Resolve ties: if multiple TXs have same max gain to an RX, pick first one
        for rx_idx in range(n):
            dominant_txs = np.where(self.associations[:, rx_idx])[0]
            if len(dominant_txs) > 1:
                # Keep only the first dominant TX
                self.associations[:, rx_idx] = False
                self.associations[dominant_txs[0], rx_idx] = True
        
        # Count how many receivers each transmitter dominates
        num_dominated_per_tx = np.sum(self.associations, axis=1)
        
        # Create explicit TX-RX pairs ensuring one-to-one matching
        # Use a greedy matching algorithm: prioritize TXs with fewer dominated RXs
        self.tx_rx_pairs = []
        unassigned_txs = []
        assigned_rxs = set()
        
        # Sort TXs by number of dominated RXs (ascending) to prioritize constrained TXs
        tx_order = np.argsort(num_dominated_per_tx)
        
        for tx_idx in tx_order:
            dominated_rxs = np.where(self.associations[tx_idx, :])[0]
            # Filter out already assigned RXs
            available_rxs = [rx for rx in dominated_rxs if rx not in assigned_rxs]
            
            if len(available_rxs) > 0:
                # Assign TX to closest available dominated RX
                if len(available_rxs) == 1:
                    chosen_rx = available_rxs[0]
                else:
                    # Pick closest RX among available dominated RXs
                    distances_to_available = [self.distances[tx_idx, rx] for rx in available_rxs]
                    chosen_rx = available_rxs[np.argmin(distances_to_available)]
                
                self.tx_rx_pairs.append((tx_idx, chosen_rx))
                assigned_rxs.add(chosen_rx)
            else:
                # This TX doesn't dominate any available RX
                # Assign to closest unassigned RX as fallback
                unassigned_rxs = [rx for rx in range(n) if rx not in assigned_rxs]
                if len(unassigned_rxs) > 0:
                    # Pick the closest unassigned RX
                    closest_rx = unassigned_rxs[np.argmin([self.distances[tx_idx, rx] for rx in unassigned_rxs])]
                    self.tx_rx_pairs.append((tx_idx, closest_rx))
                    assigned_rxs.add(closest_rx)
                    unassigned_txs.append(tx_idx)
                else:
                    # Should never happen if m == n
                    unassigned_txs.append(tx_idx)
        
        self.tx_rx_pairs = np.array(self.tx_rx_pairs)
        
        num_paired_rx = len(np.unique(self.tx_rx_pairs[:, 1]))
        num_dominated_rxs = np.sum(np.any(self.associations, axis=0))
        
        if len(unassigned_txs) > 0:
            log.warning(f"⚠ Warning: {len(unassigned_txs)} TXs don't dominate any RX: {unassigned_txs}")
        
        log.info(f"✓ Interference-aware pairing: {m} TXs dominate {num_dominated_rxs} RXs, "
              f"{m} TXs paired with {num_paired_rx} unique RXs")
        
        # """
        # This is commented out for debugging purposes


        # Update associations matrix to reflect the final one-to-one pairing
        # This ensures each TX serves exactly one RX and each RX has exactly one TX
        self.associations = np.zeros((m, n), dtype=bool)
        for tx_idx, rx_idx in self.tx_rx_pairs:
            self.associations[tx_idx, rx_idx] = True

        # Assert one-to-one mapping
        assert np.all(np.sum(self.associations, axis=1) == 1), "Each TX must be paired with exactly one RX"
        assert np.all(np.sum(self.associations, axis=0) == 1), "Each RX must be paired with exactly one TX"
        log.info("✓ One-to-one TX-RX pairing established.")

        # """
    
    def _generate_batched_rayleigh_power(
        self,
        num_timesteps: int,
        chunk_size: int = 2048,
    ) -> np.ndarray:
        """Generate time-varying channel power in chunks for all TX-RX links.

        This keeps RNG consumption link-ordered (matching the legacy nested-loop
        implementation) while moving heavy trigonometric work into batched matrix
        operations.

        Parameters
        ----------
        num_timesteps : int
            Number of time steps to generate.
        chunk_size : int, optional
            Number of TX-RX links processed per batch (default: 2048).
            Controls peak memory for intermediate trig matrices, not the output.
            Chunks are processed sequentially to preserve RNG draw order
            (required for reproducibility); NumPy/BLAS parallelism handles
            the heavy matrix work within each chunk.

        Memory
        ------
        The output array ``(m*n, num_timesteps)`` is pre-allocated in full
        (``m * n * num_timesteps * 8`` bytes).  ``chunk_size`` controls peak
        memory for intermediate trig matrices, not for the output itself.
        For example, a 100x100 network with 10 000 timesteps requires ~800 MB.
        """
        m, n = self.large_scale_fading.shape
        num_links = m * n
        if num_links == 0:
            return np.zeros((m, n, num_timesteps))

        output_bytes = num_links * num_timesteps * 8
        if output_bytes > 512 * 1024 * 1024:  # 512 MB
            log.warning(
                f"Large Rayleigh power allocation: {num_links} links x "
                f"{num_timesteps} timesteps = {output_bytes / (1024**3):.1f} GB"
            )

        N = self.num_fading_paths
        N0 = N // 4
        if N0 <= 0:
            raise ValueError(
                f"num_fading_paths must be >= 4 to generate Rayleigh fading. Got {N}."
            )

        t_vec = np.arange(num_timesteps) * self.delta_t
        a_0 = np.pi / (2 * N)
        alpha = a_0 + 2 * np.pi * np.arange(N0) / N
        w_M = 2 * np.pi * self.carrier_freq * self.speed / 3e8

        # Precompute deterministic basis terms once per call.
        base_i = w_M * np.outer(np.cos(alpha), t_vec)
        base_q = w_M * np.outer(np.sin(alpha), t_vec)
        cos_base_i = np.cos(base_i)
        sin_base_i = np.sin(base_i)
        cos_base_q = np.cos(base_q)
        sin_base_q = np.sin(base_q)

        inv_n0 = 1.0 / np.sqrt(N0)
        scale = inv_n0 * inv_n0
        two_pi = 2.0 * np.pi

        large_scale_flat = self.large_scale_fading.reshape(num_links)
        large_scale_power_flat = large_scale_flat * large_scale_flat
        H_power_flat = np.empty((num_links, num_timesteps))

        for start in range(0, num_links, chunk_size):
            end = min(start + chunk_size, num_links)
            batch_size = end - start

            # Shape (batch, 2, N0) preserves legacy RNG draw order per link:
            # first theta[0:N0], then theta_p[0:N0], then next link, ...
            random_phases = two_pi * self._rng.uniform(0.0, 1.0, size=(batch_size, 2, N0))
            theta = random_phases[:, 0, :]
            theta_p = random_phases[:, 1, :]

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            cos_theta_p = np.cos(theta_p)
            sin_theta_p = np.sin(theta_p)

            # cos(x + theta) = cos(x)cos(theta) - sin(x)sin(theta)
            i_sum = cos_theta @ cos_base_i - sin_theta @ sin_base_i
            # sin(x + theta_p) = sin(x)cos(theta_p) + cos(x)sin(theta_p)
            q_sum = cos_theta_p @ sin_base_q + sin_theta_p @ cos_base_q

            rayleigh_power = (i_sum * i_sum + q_sum * q_sum) * scale
            H_power_flat[start:end, :] = large_scale_power_flat[start:end, None] * rayleigh_power

        return H_power_flat.reshape(m, n, num_timesteps)
    
    def sample_realization(
        self,
        num_timesteps: int,
        disable_small_scale_fading: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate a channel realization over time.
        
        Parameters
        ----------
        num_timesteps : int
            Number of time steps to generate
        disable_small_scale_fading : bool, optional
            If True, only use large-scale fading (no time variation)
        
        Returns
        -------
        channel_data : dict
            Dictionary containing:
            - 'H': Channel power gains (m x n x T)
            - 'H_l': Large-scale power gains (m x n)
            - 'tx_locations': Transmitter locations (m x 2)
            - 'rx_locations': Receiver locations (n x 2)
            - 'associations': TX-RX pairing matrix (m x n)
            - 'distances': TX-RX distances (m x n)
        """
        m, n = self.large_scale_fading.shape
        
        if disable_small_scale_fading:
            # Static channel - no time variation
            H_power = np.tile(
                (self.large_scale_fading ** 2)[:, :, np.newaxis],
                (1, 1, num_timesteps)
            )
        else:
            H_power = self._generate_batched_rayleigh_power(num_timesteps=num_timesteps)

        H_l_power = np.abs(self.large_scale_fading) ** 2
        
        return {
            'H': H_power,  # Time-varying channel power (m x n x T)
            'H_l': H_l_power,  # Large-scale channel power (m x n)
            'tx_locations': self.tx_locations,
            'rx_locations': self.rx_locations,
            'associations': self.associations,
            'distances': self.distances,
            'path_loss_db': self.path_loss_db,
            'shadowing_db': self.shadowing_db,
        }
    
    def get_network_info(self) -> Dict:
        """Get static network configuration info."""
        return {
            'n_links': self.n_links,
            'deployment_range': self.deployment_range,
            'tx_locations': self.tx_locations,
            'rx_locations': self.rx_locations,
            'associations': self.associations,
            'tx_rx_pairs': self.tx_rx_pairs,
            'mean_tx_tx_distance': np.mean(distance.pdist(self.tx_locations)),
            'mean_paired_distance': np.mean([
                self.distances[tx, rx] for tx, rx in self.tx_rx_pairs
            ]),
        }


def merge_channels(channels: list, spacing: float = 100.0, layout: str = 'circular') -> WirelessChannel:
    """
    Merge multiple channel instances into a single larger network.
    
    Subnetworks are placed with spatial separation to preserve the 
    interference dominance criterion. Two layout strategies are supported:
    
    1. **Circular layout** (default): Place first subnetwork at origin, then
       arrange remaining subnetworks on a circle around it. This scales
       naturally to any number of subnetworks.
       
    2. **Linear layout**: Place subnetworks side-by-side along x-axis.
       Works well for 2-3 subnetworks.
    
    Both approaches:
    - Minimize cross-subnetwork interference naturally through distance
    - Recompute associations to allow beneficial cross-subnetwork swaps
    - Create heterogeneous networks (e.g., dense urban + sparse rural)
    
    Parameters
    ----------
    channels : list of WirelessChannel
        List of channel instances to merge
    spacing : float, optional
        Minimum gap between subnetwork boundaries in meters (default: 100m)
    layout : str, optional
        Placement strategy: 'circular' or 'linear' (default: 'circular')
    
    Returns
    -------
    merged_channel : WirelessChannel
        New channel with merged topology
    """
    if len(channels) == 0:
        raise ValueError("Cannot merge empty list of channels")
    
    if len(channels) == 1:
        return channels[0]
    
    # Compute centers and radii for each subnetwork
    subnetwork_info = []
    for ch in channels:
        all_locs = np.vstack([ch.tx_locations, ch.rx_locations])
        center = np.mean(all_locs, axis=0)
        # Radius: maximum distance from center to any node
        distances_from_center = np.linalg.norm(all_locs - center, axis=1)
        radius = np.max(distances_from_center)
        
        subnetwork_info.append({
            'channel': ch,
            'center': center,
            'radius': radius,
            'tx_locations': ch.tx_locations,
            'rx_locations': ch.rx_locations
        })
    
    if layout == 'circular':
        # Circular placement: reference subnetwork at origin, others on circle
        offsets = []
        
        # Place first subnetwork at origin
        ref_info = subnetwork_info[0]
        offset_ref = -ref_info['center']  # Shift center to origin
        offsets.append(offset_ref)
        
        if len(channels) > 1:
            # Compute circle radius: sum of reference radius, max other radius, and spacing
            max_other_radius = max(info['radius'] for info in subnetwork_info[1:])
            circle_radius = ref_info['radius'] + max_other_radius + spacing
            
            # Distribute remaining subnetworks evenly around the circle
            n_other = len(channels) - 1
            for i, info in enumerate(subnetwork_info[1:]):
                # Angle for this subnetwork (evenly distributed)
                angle = 2 * np.pi * i / n_other
                
                # Position on circle
                circle_pos = circle_radius * np.array([np.cos(angle), np.sin(angle)])
                
                # Offset: move from current center to circle position
                offset = circle_pos - info['center']
                offsets.append(offset)
        
        layout_description = f"circular (radius={circle_radius:.1f}m)" if len(channels) > 1 else "single"
        
    elif layout == 'linear':
        # Linear placement along x-axis (original approach)
        offsets = []
        current_x = 0.0
        
        for i, info in enumerate(subnetwork_info):
            # Compute bounding box
            all_locs = np.vstack([info['tx_locations'], info['rx_locations']])
            min_x = np.min(all_locs[:, 0])
            max_x = np.max(all_locs[:, 0])
            min_y = np.min(all_locs[:, 1])
            width = max_x - min_x
            
            if i == 0:
                # First subnetwork: shift to start at x=0
                offset_x = -min_x
                offset_y = -min_y
                current_x = width
            else:
                # Place to the right with spacing
                offset_x = current_x + spacing - min_x
                offset_y = -min_y
                current_x = offset_x + max_x
            
            offsets.append(np.array([offset_x, offset_y]))
        
        layout_description = f"linear ({spacing}m spacing)"
    
    else:
        raise ValueError(f"Unknown layout: {layout}. Use 'circular' or 'linear'")
    
    # Apply offsets and concatenate locations
    all_tx = []
    all_rx = []
    
    for info, offset in zip(subnetwork_info, offsets):
        tx_shifted = info['tx_locations'] + offset
        rx_shifted = info['rx_locations'] + offset
        all_tx.append(tx_shifted)
        all_rx.append(rx_shifted)
    
    all_tx = np.vstack(all_tx)
    all_rx = np.vstack(all_rx)
    
    n_links = all_tx.shape[0]
    
    # Compute total deployment range for merged network
    merged_width = np.max(all_tx[:, 0]) - np.min(all_tx[:, 0])
    merged_height = np.max(all_tx[:, 1]) - np.min(all_tx[:, 1])
    merged_range = max(merged_width, merged_height) * 1.2  # Add 20% margin
    
    # Create new channel with merged locations.
    # Use the same class as the input channels so the merged channel retains
    # V2's surgical redeployment or V3's subdivision capability.
    channel_cls = type(channels[0])
    merged = channel_cls(
        n_links=n_links,
        deployment_range=merged_range,
        min_tx_tx_distance=channels[0].min_tx_tx_distance,
        min_tx_rx_distance=channels[0].min_tx_rx_distance,
        max_tx_rx_distance=channels[0].max_tx_rx_distance,
        path_loss_exponent_short=channels[0].path_loss_exponent_short,
        path_loss_exponent_long=channels[0].path_loss_exponent_long,
        shadowing_std=channels[0].shadowing_std,
        carrier_freq=channels[0].carrier_freq,
        speed=channels[0].speed,
        num_fading_paths=channels[0].num_fading_paths,
        seed=None,  # Don't reseed for merged network
        skip_deployment=True,  # Use pre-placed locations
        tx_locations=all_tx,  # Pass locations directly
        rx_locations=all_rx,
    )
    
    # Report statistics
    subnetwork_sizes = [ch.n_links for ch in channels]
    log.info(f"✓ Merged {len(channels)} subnetworks {subnetwork_sizes} into network with {n_links} links")
    log.info(f"  Spatial layout: {layout_description}")
    
    return merged


class WirelessChannelV2(WirelessChannel):
    """
    Enhanced wireless channel model with surgical RX redeployment.
    
    This version extends WirelessChannel with an improved pairing algorithm that:
    1. Enforces max_tx_rx_distance constraint for all TX-RX pairs
    2. Surgically redeploys only problematic RXs (those violating constraints)
    3. Preserves well-paired TX-RX associations
    4. Validates global min_tx_rx_distance constraint after redeployment
    
    The redeployment ensures all TX-RX pairs can communicate effectively
    by preventing pairs at excessive distances that would fail even without
    interference.
    
    All parameters are inherited from WirelessChannel. Use this class when
    you need strict enforcement of distance constraints for reliable
    communication quality.
    """
    
    def _assign_optimal_pairing(self) -> None:
        """
        Assign each TX to its best RX with surgical redeployment for constraint violations.
        
        This enhanced pairing algorithm:
        1. Uses greedy matching prioritizing constrained TXs (fewer dominated RXs)
        2. Identifies RXs that need redeployment:
           - RXs paired with non-dominating TXs (fallback pairs)
           - RXs that violate max_tx_rx_distance constraint
        3. Surgically redeploys problematic RXs in annular region around their TX
        4. Recomputes channel gains and associations after redeployment
        
        The surgical approach is more efficient than full deployment retry because
        it preserves well-paired associations and only fixes problematic ones.
        """
        m = self.n_links
        n = self.n_links
        
        # For each receiver (column), find which transmitter has the strongest signal
        # associations[i,j] = True if TX_i has the strongest signal to RX_j
        self.associations = (
            self.large_scale_fading == np.max(self.large_scale_fading, axis=0, keepdims=True)
        )
        
        # Resolve ties: if multiple TXs have same max gain to an RX, pick first one
        for rx_idx in range(n):
            dominant_txs = np.where(self.associations[:, rx_idx])[0]
            if len(dominant_txs) > 1:
                # Keep only the first dominant TX
                self.associations[:, rx_idx] = False
                self.associations[dominant_txs[0], rx_idx] = True
        
        # Count how many receivers each transmitter dominates
        num_dominated_per_tx = np.sum(self.associations, axis=1)
        
        # Create explicit TX-RX pairs ensuring one-to-one matching
        # Use a greedy matching algorithm: prioritize TXs with fewer dominated RXs
        self.tx_rx_pairs = []
        unassigned_txs = []
        assigned_rxs = set()
        rxs_to_redeploy = []  # Track which RXs need redeployment
        
        # Sort TXs by number of dominated RXs (ascending) to prioritize constrained TXs
        tx_order = np.argsort(num_dominated_per_tx)
        
        for tx_idx in tx_order:
            dominated_rxs = np.where(self.associations[tx_idx, :])[0]
            # Filter out already assigned RXs
            available_rxs = [rx for rx in dominated_rxs if rx not in assigned_rxs]
            
            if len(available_rxs) > 0:
                # Assign TX to closest available dominated RX
                if len(available_rxs) == 1:
                    chosen_rx = available_rxs[0]
                else:
                    # Pick closest RX among available dominated RXs
                    distances_to_available = [self.distances[tx_idx, rx] for rx in available_rxs]
                    chosen_rx = available_rxs[np.argmin(distances_to_available)]
                
                self.tx_rx_pairs.append((tx_idx, chosen_rx))
                assigned_rxs.add(chosen_rx)
            else:
                # This TX doesn't dominate any available RX
                # Find an unassigned RX to redeploy around this TX
                unassigned_rxs = [rx for rx in range(n) if rx not in assigned_rxs]
                if len(unassigned_rxs) > 0:
                    # Pick the closest unassigned RX to redeploy
                    closest_rx = unassigned_rxs[np.argmin([self.distances[tx_idx, rx] for rx in unassigned_rxs])]
                    rxs_to_redeploy.append((tx_idx, closest_rx))
                    self.tx_rx_pairs.append((tx_idx, closest_rx))
                    assigned_rxs.add(closest_rx)
                    unassigned_txs.append(tx_idx)
                else:
                    # Should never happen if m == n
                    unassigned_txs.append(tx_idx)
        
        # Also check if any already-paired RXs violate max_tx_rx_distance constraint
        if self.max_tx_rx_distance is not None:
            for tx_idx, rx_idx in self.tx_rx_pairs:
                if self.distances[tx_idx, rx_idx] > self.max_tx_rx_distance:
                    # This pair violates max distance - need to redeploy
                    if (tx_idx, rx_idx) not in rxs_to_redeploy:
                        rxs_to_redeploy.append((tx_idx, rx_idx))
        
        # Redeploy RXs that need to be moved into annular region around their paired TX
        # Skip surgical redeployment for pre-placed networks (already optimal)
        if len(rxs_to_redeploy) > 0 and not getattr(self, '_is_preplaced', False):
            log.info(f"  Redeploying {len(rxs_to_redeploy)} RXs into annular regions around their paired TXs...")
            R = self.deployment_range
            
            for tx_idx, rx_idx in rxs_to_redeploy:
                # Deploy RX in annular region [min_tx_rx_distance, max_tx_rx_distance] around TX
                max_redeploy_attempts = 100
                for attempt in range(max_redeploy_attempts):
                    # Random angle
                    phi = 2 * np.pi * self._rng.uniform()
                    # Random radius in annular region
                    r = self._sample_tx_rx_radii()
                    
                    # New position relative to TX, clipped to deployment area
                    new_rx_pos = self.tx_locations[tx_idx] + np.array([r * np.cos(phi), r * np.sin(phi)])
                    new_rx_pos = np.clip(new_rx_pos, -R/2, R/2)
                    
                    # Update RX location
                    self.rx_locations[rx_idx] = new_rx_pos
                    
                    # Check if this position satisfies min_tx_rx_distance for all TXs
                    distances_to_all_txs = np.linalg.norm(self.tx_locations - new_rx_pos, axis=1)
                    if np.all(distances_to_all_txs >= self.min_tx_rx_distance):
                        break
                else:
                    log.warning(f"    ⚠ Warning: Could not find valid position for RX {rx_idx} after {max_redeploy_attempts} attempts")
            
            # Clear cached shadowing so it gets regenerated for new RX positions
            self.shadowing_db_deployment = None
            
            # Recompute distances and large-scale fading for redeployed RXs
            self.distances = distance.cdist(self.tx_locations, self.rx_locations, 'euclidean')
            self._compute_large_scale_fading()
            
            # Recompute associations after redeployment
            self.associations = (
                self.large_scale_fading == np.max(self.large_scale_fading, axis=0, keepdims=True)
            )
            for rx_idx in range(n):
                dominant_txs = np.where(self.associations[:, rx_idx])[0]
                if len(dominant_txs) > 1:
                    self.associations[:, rx_idx] = False
                    self.associations[dominant_txs[0], rx_idx] = True
        
        self.tx_rx_pairs = np.array(self.tx_rx_pairs)
        
        num_paired_rx = len(np.unique(self.tx_rx_pairs[:, 1]))
        num_dominated_rxs = np.sum(np.any(self.associations, axis=0))
        
        if len(unassigned_txs) > 0:
            log.warning(f"⚠ Warning: {len(unassigned_txs)} TXs don't dominate any RX: {unassigned_txs}")
        
        log.info(f"✓ Interference-aware pairing: {m} TXs dominate {num_dominated_rxs} RXs, "
              f"{m} TXs paired with {num_paired_rx} unique RXs")
        
        # Update associations matrix to reflect the final one-to-one pairing
        # This ensures each TX serves exactly one RX and each RX has exactly one TX
        self.associations = np.zeros((m, n), dtype=bool)
        for tx_idx, rx_idx in self.tx_rx_pairs:
            self.associations[tx_idx, rx_idx] = True

        # Assert one-to-one mapping
        assert np.all(np.sum(self.associations, axis=1) == 1), "Each TX must be paired with exactly one RX"
        assert np.all(np.sum(self.associations, axis=0) == 1), "Each RX must be paired with exactly one TX"
        log.info("✓ One-to-one TX-RX pairing established.")




"""
WirelessChannelV3 with recursive subnetwork subdivision.

This module extends WirelessChannelV2 with an adaptive deployment strategy
that recursively subdivides the deployment area into smaller subnetworks
when deployment fails.
"""

class WirelessChannelV3(WirelessChannelV2):
    """
    Enhanced wireless channel with recursive subnetwork subdivision on deployment failure.
    
    This version extends WirelessChannelV2 with an adaptive spatial subdivision strategy:
    1. Attempts standard deployment in full area
    2. On failure, divides area into 4 quadrants (2x2 subgrid)
    3. Distributes links across quadrants (1/4 each)
    4. Recursively subdivides failing quadrants
    5. Combines successful subnetwork deployments
    
    This approach is particularly useful for:
    - Large networks that struggle with interference dominance criterion
    - Dense deployments where spatial clustering helps
    - Networks with tight distance constraints
    
    Parameters
    ----------
    max_recursion_depth : int, optional
        Maximum depth of recursive subdivision (default: 3)
        Depth 0 = full network
        Depth 1 = 4 subnetworks
        Depth 2 = 16 subnetworks
        Depth 3 = 64 subnetworks
    
    All other parameters inherited from WirelessChannelV2.
    
    Example
    -------
    >>> channel = WirelessChannelV3(
    ...     n_links=100,
    ...     deployment_range=200.0,
    ...     max_recursion_depth=2,
    ...     seed=42
    ... )
    """
    
    def __init__(
        self,
        n_links: int,
        deployment_range: float,
        max_recursion_depth: int = 3,
        **kwargs
    ):
        """Initialize WirelessChannelV3 with recursive subdivision capability."""
        self.max_recursion_depth = max_recursion_depth
        self._subdivision_log = []  # Track subdivision history
        super().__init__(n_links=n_links, deployment_range=deployment_range, **kwargs)
    
    def _deploy_network(self) -> None:
        """
        Deploy network with recursive subdivision fallback.
        
        First attempts standard deployment. If that fails after max_attempts,
        recursively subdivides the area into smaller regions and deploys
        subnetworks independently.
        """
        # Try standard deployment first
        try:
            self._try_deploy()
            self._subdivision_log.append({
                'depth': 0,
                'success': True,
                'n_links': self.n_links,
                'range': self.deployment_range
            })
        except RuntimeError as e:
            # Standard deployment failed, try recursive subdivision
            log.info(f"\n{'='*60}")
            log.info("Standard deployment failed. Attempting recursive subdivision...")
            log.info(f"{'='*60}")
            
            self._subdivision_log.append({
                'depth': 0,
                'success': False,
                'n_links': self.n_links,
                'range': self.deployment_range,
                'error': str(e)
            })
            
            # Clear any partial deployment
            self.tx_locations = None
            self.rx_locations = None
            self.shadowing_db_deployment = None
            
            # Deploy using recursive subdivision
            tx_locs, rx_locs, shadowing = self._deploy_recursive(
                n_links=self.n_links,
                center_x=0.0,
                center_y=0.0,
                subrange=self.deployment_range,
                depth=0
            )
            
            # Store combined results
            self.tx_locations = tx_locs
            self.rx_locations = rx_locs
            
            # IMPORTANT: Clear block-diagonal shadowing from subdivision
            # We need to regenerate shadowing for the full combined network
            # to capture all cross-subnetwork interference terms properly
            self.shadowing_db_deployment = None
            
            log.info(f"\n{'='*60}")
            log.info(f"✓ Recursive subdivision successful!")
            log.info(f"  Total subdivisions: {len(self._subdivision_log) - 1}")
            log.info(f"  Final subnetworks: {sum(1 for entry in self._subdivision_log if entry['success'])}")
            log.info(f"  Regenerating shadowing for full combined network...")
            log.info(f"{'='*60}\n")
    
    def _deploy_recursive(
        self,
        n_links: int,
        center_x: float,
        center_y: float,
        subrange: float,
        depth: int
    ) -> tuple:
        """
        Recursively deploy subnetwork with spatial subdivision fallback.
        
        Parameters
        ----------
        n_links : int
            Number of links to deploy in this subnetwork
        center_x : float
            X-coordinate of subnetwork center
        center_y : float
            Y-coordinate of subnetwork center
        subrange : float
            Size of this subnetwork area
        depth : int
            Current recursion depth
        
        Returns
        -------
        tx_locations : ndarray of shape (n_links, 2)
            Transmitter positions
        rx_locations : ndarray of shape (n_links, 2)
            Receiver positions
        shadowing_db : ndarray of shape (n_links, n_links)
            Shadowing values for deployed links
        """
        if n_links == 0:
            return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 0))
        
        # Try to deploy this subnetwork
        log.info(f"{'  ' * depth}Attempting subnetwork: {n_links} links, {subrange:.1f}m range, depth {depth}")
        
        # Temporarily mutate self to deploy the sub-region.
        # try/finally guarantees restoration even on unexpected exceptions
        # (e.g. MemoryError, KeyboardInterrupt) not just RuntimeError.
        original_range = self.deployment_range
        original_n_links = self.n_links
        try:
            self.deployment_range = subrange
            self.n_links = n_links

            # Attempt deployment
            self._try_deploy()

            # Shift locations to subnetwork center
            tx_locs = self.tx_locations + np.array([center_x, center_y])
            rx_locs = self.rx_locations + np.array([center_x, center_y])
            shadowing = self.shadowing_db_deployment

            self._subdivision_log.append({
                'depth': depth,
                'success': True,
                'n_links': n_links,
                'range': subrange,
                'center': (center_x, center_y)
            })

            log.info(f"{'  ' * depth}✓ Subnetwork deployed successfully")
            return tx_locs, rx_locs, shadowing

        except RuntimeError:
            # Check if we can subdivide further
            if depth >= self.max_recursion_depth:
                raise RuntimeError(
                    f"Maximum recursion depth {self.max_recursion_depth} reached. "
                    f"Cannot deploy {n_links} links in {subrange:.1f}m area."
                )

            if n_links < 4:
                raise RuntimeError(
                    f"Cannot subdivide {n_links} links (need at least 4 for subdivision)."
                )

            log.warning(f"{'  ' * depth}⚠ Deployment failed, subdividing into 4 quadrants...")

            self._subdivision_log.append({
                'depth': depth,
                'success': False,
                'n_links': n_links,
                'range': subrange,
                'center': (center_x, center_y),
                'subdivided': True
            })
            
            # Subdivide into 4 quadrants
            new_subrange = subrange / 2
            offset = subrange / 4  # Distance from center to quadrant centers
            
            # Distribute links across quadrants
            links_per_quadrant = n_links // 4
            remainder = n_links % 4
            quadrant_links = [links_per_quadrant] * 4
            for i in range(remainder):
                quadrant_links[i] += 1
            
            # Quadrant centers (bottom-left, bottom-right, top-left, top-right)
            quadrants = [
                (center_x - offset, center_y - offset, quadrant_links[0]),  # Bottom-left
                (center_x + offset, center_y - offset, quadrant_links[1]),  # Bottom-right
                (center_x - offset, center_y + offset, quadrant_links[2]),  # Top-left
                (center_x + offset, center_y + offset, quadrant_links[3]),  # Top-right
            ]
            
            # Recursively deploy each quadrant
            all_tx = []
            all_rx = []
            all_shadowing = []
            
            for qx, qy, q_links in quadrants:
                if q_links > 0:
                    tx, rx, shad = self._deploy_recursive(
                        n_links=q_links,
                        center_x=qx,
                        center_y=qy,
                        subrange=new_subrange,
                        depth=depth + 1
                    )
                    all_tx.append(tx)
                    all_rx.append(rx)
                    all_shadowing.append(shad)
            
            # Combine results
            combined_tx = np.vstack(all_tx) if all_tx else np.zeros((0, 2))
            combined_rx = np.vstack(all_rx) if all_rx else np.zeros((0, 2))
            
            # Combine shadowing matrices (block diagonal structure)
            if all_shadowing:
                total_links = sum(s.shape[0] for s in all_shadowing)
                combined_shadowing = np.zeros((total_links, total_links))
                offset = 0
                for shad in all_shadowing:
                    size = shad.shape[0]
                    combined_shadowing[offset:offset+size, offset:offset+size] = shad
                    offset += size
            else:
                combined_shadowing = np.zeros((0, 0))
            
            log.info(f"{'  ' * depth}✓ Subdivision complete: {len(all_tx)} quadrants combined")
            return combined_tx, combined_rx, combined_shadowing

        finally:
            # Always restore self.* regardless of how the block exits
            # (success, RuntimeError, or any other exception).
            self.deployment_range = original_range
            self.n_links = original_n_links

    def get_subdivision_summary(self) -> dict:
        """
        Get summary of subdivision process.
        
        Returns
        -------
        dict
            Summary statistics of the subdivision process including:
            - max_depth_reached: Maximum recursion depth used
            - total_subdivisions: Total number of subdivision attempts
            - successful_subnetworks: Number of successfully deployed subnetworks
            - failed_subnetworks: Number of failed deployment attempts
        """
        if not self._subdivision_log:
            return {
                'max_depth_reached': 0,
                'total_subdivisions': 0,
                'successful_subnetworks': 0,
                'failed_subnetworks': 0
            }
        
        return {
            'max_depth_reached': max(entry['depth'] for entry in self._subdivision_log),
            'total_subdivisions': len(self._subdivision_log),
            'successful_subnetworks': sum(1 for entry in self._subdivision_log if entry.get('success', False)),
            'failed_subnetworks': sum(1 for entry in self._subdivision_log if not entry.get('success', False)),
            'subdivision_log': self._subdivision_log
        }
