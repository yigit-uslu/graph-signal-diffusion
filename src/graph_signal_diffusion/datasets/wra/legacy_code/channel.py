from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy import sparse
import seaborn as sns
import torch
import abc

import wandb
# from core.DualSampler import FRLambdaSampler, URLambdaSampler, WMMSELambdaSampler
from .channel_utils import UDN_PL, calc_rates, convert_P_max_and_noise_PSD, convert_channels, deploy_tx_rx_pairs, create_channel_matrix_over_time, get_network_area, load_channels_from_path, long_term_fading, normalize_matrix, save_channel_data_to_path
from .data_utils import (
    Data_modTxIndex, WirelessDataset, permute_pygraph
)
import os   
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian, from_scipy_sparse_matrix
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import GDC
from .channel_utils import wmmse, ITLinQ
# from core.config import T_DEFAULT, MAX_LOGGED_NETWORKS, DELTA_T
from .utils import assert_same_dtype_and_size, seed_everything
import copy
from scipy.spatial import distance
from scipy.linalg import block_diag


MAX_LOGGED_NETWORKS = 2
DELTA_T = 50
T_DEFAULT = 100


class ChannelFactory(abc.ABC):
    def __init__(self, accelerator, m, n, density_mode, base_R, num_channels, channel_load_path,
                    fast_fading = True,
                    channel_save_path = None,
                    use_single_channel = False,
                    channel_seed = None,
                    max_n_per_subnetwork = 13, # 50
                    channel_kws = None
                    ):
        self.m = m
        self.n = n
        self.base_R = base_R
        self.fast_fading = fast_fading
        self.num_channels = num_channels
        self.channel_load_path = channel_load_path
        self.channel_save_path = channel_save_path
        self.use_single_channel = use_single_channel
        self.channel_seed = channel_seed
        self.max_n_per_subnetwork = max_n_per_subnetwork
        self.channel_kws = channel_kws

        self.instantiate_from_subnetworks = False
        if channel_load_path not in [None, "none"]:
            accelerator.print(f'Loading channel data from {channel_load_path}.')

            tx_rx_locs_H_l, channel_gains = load_channels_from_path(channel_load_path)

            if isinstance(tx_rx_locs_H_l[0], list) and isinstance(channel_gains[0], list):
                # we will preload the ChannelfromSubnetworks data
                self.instantiate_from_subnetworks = True

                self.m = [len(loc_and_h_l['locTx']) for loc_and_h_l in tx_rx_locs_H_l]
                self.n = [len(loc_and_h_l['locRx']) for loc_and_h_l in tx_rx_locs_H_l]
                self.R = [get_network_area(m, density_mode, base_R = base_R) for m in self.m]

            else:
                self.m = m
                self.n = n
                self.R = get_network_area(m, density_mode, base_R = base_R)

        else:
            accelerator.print(f'No channel data load path found at {channel_load_path}! Creating channel data...')
            tx_rx_locs_H_l, channel_gains = list(zip(*[(None, None) for _ in range(num_channels)]))

            m_list, n_list, R_list = divide_network_into_subnetworks(m, n, base_R, density_mode, max_n_per_subnetwork = self.max_n_per_subnetwork)
            if len(m_list) > 1:
                self.instantiate_from_subnetworks = True
                self.m = m_list
                self.n = n_list
                self.R = R_list

            else:
                self.m = m_list[0]
                self.n = n_list[0]
                self.R = R_list[0]

        self.tx_rx_locs_H_l = tx_rx_locs_H_l
        self.channel_gains = channel_gains


    def generate_channel_objects(self, accelerator, *args, **kwargs):

        if self.instantiate_from_subnetworks:
            channels_list = [ChannelfromSubNetworks(
                m=self.m, n=self.n,
                R=self.R,
                tx_rx_locs_H_l = loc_and_h_l,
                channel_gains=data,
                fast_fading=self.fast_fading,
                seed = [self.channel_seed + i if self.channel_seed is not None else self.channel_seed for i in range(len(self.m))],
                channel_kws=self.channel_kws
                ) for loc_and_h_l, data in tqdm(zip(self.tx_rx_locs_H_l, self.channel_gains))
                ]
        
            # if self.use_single_channel:
            #     print(f"Repeating Channel 0 {len(channels_list)} times.")
            #     channels_list = [channels_list[0] for _ in range(len(channels_list))]
        
        else:
            channels_list = [Channel(m=self.m, n=self.n,
                             R=self.R,
                             tx_rx_locs_H_l = loc_and_h_l,
                             channel_gains=data,
                             fast_fading=self.fast_fading,
                             seed=self.channel_seed,
                             channel_kws=self.channel_kws
                             ) for loc_and_h_l, data in tqdm(zip(self.tx_rx_locs_H_l, self.channel_gains))]

        if self.use_single_channel:
            accelerator.print(f"Repeating Channel 0 {len(channels_list)} times.")
            channels_list = [channels_list[0] for _ in range(len(channels_list))]

        return channels_list
    

class Channel(abc.ABC):
    def __init__(self, m, n, R, tx_rx_locs_H_l = None, channel_gains = None, fast_fading = True, seed = None, channel_kws = None):
        self.m = m
        self.n = n
        self.R = R
        self.fast_fading = fast_fading
        self.seed = seed
        self.channel_kws = channel_kws

        if self.seed is not None:
            seed_everything(self.seed)

        if tx_rx_locs_H_l is None:
            tx_rx_locs_H_l = deploy_tx_rx_pairs(m=m,
                                                n=n,
                                                R=R,
                                                min_D_TxTx=self.min_D_TxTx,
                                                min_D_TxRx=self.min_D_TxRx,
                                                max_D_TxRx=self.max_D_TxRx,
                                                shadowing=self.shadowing
                                                )
            
        self.tx_rx_locs = {'tx': tx_rx_locs_H_l['locTx'],
                           'rx': tx_rx_locs_H_l['locRx']
                           }
        
        self.tx_rx_associations = tx_rx_locs_H_l['associations'] if 'associations' in tx_rx_locs_H_l else None
        self.large_scale_fading_dict = tx_rx_locs_H_l['large_scale_fading_dict'] if 'large_scale_fading_dict' in tx_rx_locs_H_l else None

        self.H_l_sqrt = tx_rx_locs_H_l['H_l']

        self.safety_check_large_scale_fading_data()

        if channel_gains is None:
            channel_gains = self.sample()
        self.channel_gains = channel_gains
    

    @property
    def T_eff(self):
        return T_DEFAULT
    @property
    def T_warmup(self):
        return 50 # number of steps to stabilize user rates
    @property
    def T(self):
        return self.T_eff + self.T_warmup
    
    @property
    def min_D_TxTx(self):
        if self.channel_kws is not None and "min_D_TxTx" in self.channel_kws and self.channel_kws["min_D_TxTx"] is not None:
            return self.channel_kws["min_D_TxTx"]
        else:
            return 20
        # return 20
        return 35 # minimum Tx-Tx distance
    @property
    def min_D_TxRx(self):
        if self.channel_kws is not None and "min_D_TxRx" in self.channel_kws and self.channel_kws["min_D_TxRx"] is not None:
            return self.channel_kws["min_D_TxRx"]
        else:
            return 10 # minimum Tx-Rx distance
    @property
    def max_D_TxRx(self):
        if self.channel_kws is not None and "max_D_TxRx" in self.channel_kws and self.channel_kws["max_D_TxRx"] is not None:
            return self.channel_kws["max_D_TxRx"]
        else:
            return 50 # maximum Tx-Rx distance
    @property
    def shadowing(self):
        if self.channel_kws is not None and "shadowing" in self.channel_kws and self.channel_kws["shadowing"] is not None:
            return self.channel_kws["shadowing"]
        else:
            return 7 # shadowing standard dev
        
    @property
    def f_c(self):
        return 2.4e9 # carrier freq. (Hz)
    @property
    def speed(self):
        return 1.0 # receiver speed (m/s)
    @property
    def num_fading_paths(self):
        return 100

    def sample(self, n_samples = None):
        n_samples = self.T if n_samples is None else n_samples
        channel_gains = create_channel_matrix_over_time(H_l=self.H_l_sqrt,
                                                        T=n_samples,
                                                        num_fading_paths=self.num_fading_paths,
                                                        f_c=self.f_c,
                                                        speed=self.speed,
                                                        disable_short_term_fading = (self.fast_fading == False)
                                                        )
        
        return channel_gains
    

    def get_channel_data(self):
        
        channel_gains = self.channel_gains
        tx_rx_locs_H_l = {'locTx': self.tx_rx_locs['tx'],
                          'locRx': self.tx_rx_locs['rx'],
                          'H_l': self.H_l_sqrt,
                          'associations': self.tx_rx_associations,
                          'large_scale_fading_dict': self.large_scale_fading_dict
                          }
        
        return {'channel_gains': channel_gains, 'tx_rx_locs_H_l': tx_rx_locs_H_l}
    

    def safety_check_large_scale_fading_data(self):
        """ 
        Function that checks whether large-scale channel fading data is consistent with the network parameters.
        """

        if self.large_scale_fading_dict is not None:
            # Check whether D_TxRx generates same H_l
            D_TxRx = distance.cdist(self.tx_rx_locs['tx'], self.tx_rx_locs['rx'], 'euclidean')
            path_loss = UDN_PL(D_TxRx)

            assert np.allclose(path_loss, self.large_scale_fading_dict['path_loss']), "Path loss is not consistent with the large-scale fading dictionary."
            
            L = path_loss + self.large_scale_fading_dict["shadowing_loss"]
            H_l = np.sqrt(np.power(10, -L / 10))
            assert np.allclose(H_l, self.H_l_sqrt), "H_l is not consistent with the large-scale fading dictionary losses."


        # print("Large-scale fading data is consistent with the network parameters.")
        
    


    def plot_tx_rx_locs(self, ax = None, alpha = 1.0, xmin = None, xmax = None, ymin = None, ymax = None):
        if ax is None:
            fig, ax = plt.subplots()

        if xmin is None:
            xmin = -self.R / 2
        if xmax is None:
            xmax = self.R / 2
        if ymin is None:
            ymin = -self.R / 2
        if ymax is None:
            ymax = self.R / 2

        # Define colormap
        cmap = plt.get_cmap('viridis')
        colors_tx = cmap(np.linspace(0, 1, self.tx_rx_locs['tx'].shape[0]))
        colors_rx = cmap(np.linspace(0, 1, self.tx_rx_locs['rx'].shape[0]))
        
        ax.scatter(self.tx_rx_locs['tx'][:, 0], self.tx_rx_locs['tx'][:, 1], label='Tx', marker = 'd', c=colors_tx, alpha = alpha)
        ax.scatter(self.tx_rx_locs['rx'][:, 0], self.tx_rx_locs['rx'][:, 1], label='Rx', marker = 'x', c=colors_rx, alpha = alpha)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.legend()
        return ax
    


    def plot_H_l(self, ax = None, alpha = 1.0, xmin = None, xmax = None, ymin = None, ymax = None):
        if ax is None:
            fig, ax = plt.subplots()

        if xmin is None:
            xmin = -self.R / 2
        if xmax is None:
            xmax = self.R / 2
        if ymin is None:
            ymin = -self.R / 2
        if ymax is None:
            ymax = self.R / 2

        color_data = np.copy(self.H_l_sqrt)

        # Define custom colormap: gray → red
        # colors = [(0.5, 0.5, 0.5), (1, 0, 0)]  # RGB: Gray to Red
        cmap = plt.get_cmap('coolwarm_r')

        # Normalize channel gains for color mapping
        norm = LogNorm(vmin=np.min(color_data), vmax=np.max(color_data))
        # colors = cmap(norm(color_data))

        sns.heatmap(data=color_data, cmap = cmap, norm = norm, annot=False, cbar=True, cbar_kws={'label': 'H_l'}, ax=ax)
        ax.grid(True)

        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # # plt.colorbar(sm, ax=ax, label='Channel Gains')

        # # Create a smaller colorbar with these options:
        # # shrink: Makes the colorbar smaller (0.5 = 50% of original size)
        # # aspect: Controls the ratio of long to short dimensions
        # # pad: Controls spacing between plot and colorbar
        # cbar = plt.colorbar(sm, ax=ax, label='H_l', shrink=0.75, aspect=10, pad=0.05)
        
        # # Optionally make the label and ticks smaller
        # cbar.ax.tick_params(labelsize=10)
        # cbar.set_label('H_l', size=14)
 
        return ax
    

    def get_random_loc_perturbations(self, n, delta_r):
        phi = 2 * np.pi * np.random.uniform(size = (n,))
        # r = delta_r
        # locs = np.clip(locs + np.stack((r * np.cos(phi), r * np.sin(phi)), axis=1), a_min= -self.R / 2, a_max=self.R / 2)
        delta_locs = np.stack((delta_r * np.cos(phi), delta_r * np.sin(phi)), axis=1)

        return delta_locs
    

    def perturb_locs(self, locs, delta_r = None, delta_locs = None, a_min = None, a_max = None):
        if delta_locs is None:
            n = locs.shape[0]
            delta_locs = self.get_random_loc_perturbations(n, delta_r)
        
        locs = locs + delta_locs
        if a_min is not None or a_max is not None:
            locs = np.clip(locs, a_min=a_min, a_max=a_max)
        return locs
    

    def perturb_angles(self, locs_tx, locs_rx, associations, delta_phi = None):

        raise NotImplementedError

        D_TxRx = distance.cdist(locs_tx, locs_rx, 'euclidean')

        pass

        # return locs
    

    def get_random_long_term_channel_gain_perturbations(self, n, delta_g):
        return delta_g * np.random.randn(n)
    

    def perturb_long_term_channel_gains(self, channel_gains, delta_g, delta_channel_gains = None):
        if delta_channel_gains is None:
            n = channel_gains.shape[0]
            delta_channel_gains = self.get_random_long_term_channel_gain_perturbations(n, delta_g)
        channel_gains = np.clip(channel_gains + delta_g, a_min=0, a_max=None)

        return channel_gains
    

class ChannelfromSubNetworks(Channel):
    """
    A parent Channel object that is created from a list of sub-networks. Transmitters and receivers of each subnetwork are
    placed in a subgrid in the network area of the parent network. Channel gains are adjusted accordingly.
    """
    def __init__(self, m, n, R, tx_rx_locs_H_l=None, channel_gains=None, fast_fading=True, seed=None, channel_kws=None):
        self.m = m
        self.n = n
        self.R = R
        self.fast_fading = fast_fading
        self.seed = seed
        self.tx_rx_locs_H_l = tx_rx_locs_H_l
        self.channel_gains = channel_gains
        self.channel_kws = channel_kws

        self.check_inputs()
        self.r_by_c_grid = self.get_grid_size()

        subnetworks = []
        for id, (m, n, R, tx_rx_locs_H_l, channel_gains, seed) in tqdm(enumerate(zip(self.m, self.n, self.R, self.tx_rx_locs_H_l, self.channel_gains, self.seed))):
            subnetwork = Channel(m=m, n=n, R=R,
                                 tx_rx_locs_H_l=tx_rx_locs_H_l,
                                 channel_gains=channel_gains,
                                 fast_fading=fast_fading, seed=seed,
                                 channel_kws=channel_kws
                                 )
            
            tx_rx_locs_H_l = subnetwork.get_channel_data()['tx_rx_locs_H_l']
            tx_locs, rx_locs, associations = tx_rx_locs_H_l['locTx'], tx_rx_locs_H_l['locRx'], tx_rx_locs_H_l['associations']

            delta_tx = delta_rx = np.array([subnetwork.R + subnetwork.min_D_TxTx, subnetwork.R + subnetwork.min_D_TxTx]) * np.array([id % self.r_by_c_grid[1], -(id // self.r_by_c_grid[0]) ])
            # delta_tx = np.expand_dims(delta_tx, axis = 0)
            # delta_rx = np.expand_dims(delta_rx, axis = 0)

            tx_locs_new = subnetwork.perturb_locs(locs = tx_locs, delta_locs = delta_tx)
            rx_locs_new = subnetwork.perturb_locs(locs = rx_locs, delta_locs = delta_rx)

            associations_new = associations

            subnetwork.tx_rx_locs['tx'] = tx_locs_new
            subnetwork.tx_rx_locs['rx'] = rx_locs_new

            D_TxRx = distance.cdist(tx_locs_new, rx_locs_new, 'euclidean')
            H_l, large_scale_fading_dict = long_term_fading(D_TxRx, shadowing=subnetwork.shadowing,
                                                            shadowing_loss = subnetwork.large_scale_fading_dict["shadowing_loss"],
                                                            return_long_term_fading_data=True)
            subnetwork.H_l_sqrt = H_l
            subnetwork.large_scale_fading_dict = large_scale_fading_dict
            subnetwork.tx_rx_associations = associations_new

            # fig, ax = plt.subplots(figsize = (10, 10))
            # subnetwork.plot_tx_rx_locs(ax=ax, xmin = -subnetwork.R / 2 + delta_tx[0], xmax = subnetwork.R / 2 + delta_tx[0],
            #                                   ymin = -subnetwork.R / 2 + delta_tx[1], ymax = subnetwork.R / 2 + delta_tx[1])

            # plt.savefig(f'./ChannelPerturbTest/subnetwork_{id}.png', dpi = 300)
            # plt.close(fig)

            subnetworks.append(subnetwork)

        self.merge_subnetworks(subnetworks)
        


    def merge_subnetworks(self, subnetworks):
        """
        Merge subnetworks into a single network.
        """
        locTx = np.concatenate([subnetwork.tx_rx_locs['tx'] for subnetwork in subnetworks], axis = 0)
        locRx = np.concatenate([subnetwork.tx_rx_locs['rx'] for subnetwork in subnetworks], axis = 0)

        associations = block_diag(*[subnetwork.tx_rx_associations for subnetwork in subnetworks])

        # Recenter the network
        centroid = np.mean(locTx, axis = 0)
        locTx = locTx - centroid
        locRx = locRx - centroid

        self.R = 2 * max(np.max(np.abs(locTx)), np.max(np.abs(locRx)))
        print("Network width: ", self.R, 'subnetwork width: ', subnetworks[0].R, "n_subnetworks: ", len(subnetworks))

        self.m = len(locTx)
        self.n = len(locRx)
        self.seed = subnetworks[0].seed
        self.fast_fading = subnetworks[0].fast_fading
        
        D_TxRx = distance.cdist(locTx, locRx, 'euclidean')
        H_l, large_scale_fading_dict = long_term_fading(D_TxRx, self.shadowing, return_long_term_fading_data=True)

        self.tx_rx_locs = {'tx': locTx, 'rx': locRx}
        self.tx_rx_associations = associations
        self.H_l_sqrt = H_l
        self.large_scale_fading_dict = large_scale_fading_dict

        self.safety_check_large_scale_fading_data()

        channel_gains = self.sample()
        self.channel_gains = channel_gains

        # fig, ax = plt.subplots(figsize = (int(10 * self.R / subnetworks[0].R), int(10 *  self.R / subnetworks[0].R)))
        # self.plot_tx_rx_locs(ax=ax)

        # plt.savefig(f'./ChannelPerturbTest/Network.png', dpi = 300)
        # plt.close(fig)
            

    def check_inputs(self):
        if not isinstance(self.m, list): # and isinstance(self.m_list, int):
            self.m = [self.m]

        if not isinstance(self.n, list): # and isinstance(self.n_list, int):
            self.n = [self.n]
        
        if not isinstance(self.R, list): # and isinstance(self.R_list, int):
            self.R = [self.R]

        if not isinstance(self.seed, list): # and isinstance(self.seed_list, int):
            self.seed = [self.seed]

        if not isinstance(self.tx_rx_locs_H_l, list): # and isinstance(self.tx_rx_locs_H_l_list, dict):
            self.tx_rx_locs_H_l = [self.tx_rx_locs_H_l]

        if not isinstance(self.channel_gains, list): # and isinstance(self.channel_gains_list, dict):
            self.channel_gains = [self.channel_gains]

        n_subnetworks = max(len(self.m), len(self.n), len(self.R), len(self.seed), len(self.tx_rx_locs_H_l), len(self.channel_gains))
        self.m = self.m + [self.m[-1]] * (n_subnetworks - len(self.m))
        self.n = self.n + [self.n[-1]] * (n_subnetworks - len(self.n))
        self.R = self.R + [self.R[-1]] * (n_subnetworks - len(self.R))
        self.seed = self.seed + [self.seed[-1]] * (n_subnetworks - len(self.seed))
        self.tx_rx_locs_H_l = self.tx_rx_locs_H_l + [self.tx_rx_locs_H_l[-1]] * (n_subnetworks - len(self.tx_rx_locs_H_l))
        self.channel_gains = self.channel_gains + [self.channel_gains[-1]] * (n_subnetworks - len(self.channel_gains))

        assert len(self.m) == len(self.n) == len(self.R) == len(self.seed) == len(self.tx_rx_locs_H_l) == len(self.channel_gains) == n_subnetworks, \
            "Length of m_list, n_list, and R_list must be the same."
        

    def get_grid_size(self):
        """
        """
        total_subgrids = len(self.n)

        c = int(np.sqrt(total_subgrids))
        r = total_subgrids // c

        while not (r * c == total_subgrids) and r > 0:
            c += 1
            r == total_subgrids // c

        return (r, c)

    @property
    def max_n_per_subgrid(self):
        return 50
    

def divide_network_into_subnetworks(m, n, base_R, density_mode, max_n_per_subnetwork = 50):

    R = get_network_area(m, density_mode, base_R = base_R)

    num_subnetworks = int(np.ceil(n / max_n_per_subnetwork))
    n_per_subnetwork = n // num_subnetworks
    m_per_subnetwork = m // num_subnetworks
    R_subnetwork = R // np.sqrt(num_subnetworks)

    m_list = [m_per_subnetwork] * num_subnetworks
    n_list = [n_per_subnetwork] * num_subnetworks
    R_list = [R_subnetwork] * num_subnetworks

    m_list[-1] = m - m_per_subnetwork * (num_subnetworks - 1)
    n_list[-1] = n - n_per_subnetwork * (num_subnetworks - 1)
    # R_list[-1] = R - R_subnetwork * (num_subnetworks - 1)

    assert sum(m_list) == m, f"Sum of m_list ({sum(m_list)}) is not equal to m ({m})."
    assert sum(n_list) == n, f"Sum of n_list ({sum(n_list)}) is not equal to n ({n})."

    return m_list, n_list, R_list
    




class ChannelPerturbationWrapper(abc.ABC):
    def __init__(self, channel: Channel, perturbation_type: str, perturbation_params: dict, render_perturbation: bool = True, save_path: str = None):
        """ 
        Wraps a Channel object to create a Perturbed Channel object.
        """
        self.channel = channel

        if perturbation_type in ['none', None]:
            print("No perturbation wrapper applied.")
            return

        if 'seed' in perturbation_params and perturbation_params['seed'] is not None:
            seed_everything(perturbation_params['seed'])
            print(f"Channel perturbation wrapper seed = {perturbation_params['seed']} is applied.")


        if perturbation_type == 'perturb_locs':
            tx_locs_new = self.perturb_locs(locs = self.channel.tx_rx_locs['tx'], delta_locs = None, delta_r = perturbation_params['delta_r']['tx'])
            rx_locs_new = self.perturb_locs(locs = self.channel.tx_rx_locs['rx'], delta_locs = None, delta_r = perturbation_params['delta_r']['rx'])

            D_TxRx = distance.cdist(tx_locs_new, rx_locs_new, 'euclidean')
            H_l, large_scale_fading_dict = long_term_fading(D_TxRx, shadowing=self.channel.shadowing,
                                                            shadowing_loss = self.channel.large_scale_fading_dict["shadowing_loss"],
                                                            return_long_term_fading_data=True)
            
            if render_perturbation:
                fig, ax = plt.subplots(1, 1, figsize = (10, 10))
                ax = self.render_perturbation(ax=ax)
            
            self.channel.tx_rx_locs['tx'] = tx_locs_new
            self.channel.tx_rx_locs['rx'] = rx_locs_new
            self.channel.H_l_sqrt = H_l
            self.channel.large_scale_fading_dict = large_scale_fading_dict

            self.channel.safety_check_large_scale_fading_data()

            if render_perturbation:
                # with plt.rc_context(rc = {"lines.markeralpha": 0.5}):
                ax = self.render_perturbation(ax=ax, alpha = 0.5, perturbation_type = perturbation_type)

                if save_path is not None:
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(f'{save_path}/render_{perturbation_type}_perturbation.pdf', dpi = 300)
                plt.close(fig)
            
            channel_gains = self.channel.sample()
            self.channel.channel_gains = channel_gains

            print("Channel perturbation wrapper successfully applied.")


        elif perturbation_type == 'perturb_angles':
            raise NotImplementedError
            tx_locs_new = self.perturb_locs(locs = self.channel.tx_rx_locs['tx'], delta_locs = None, delta_r = perturbation_params['delta_r']['tx'])
            rx_locs_new = self.perturb_locs(locs = self.channel.tx_rx_locs['rx'], delta_locs = None, delta_r = perturbation_params['delta_r']['rx'])

            D_TxRx = distance.cdist(tx_locs_new, rx_locs_new, 'euclidean')
            H_l, large_scale_fading_dict = long_term_fading(D_TxRx, shadowing=self.channel.shadowing,
                                                            shadowing_loss = self.channel.large_scale_fading_dict["shadowing_loss"],
                                                            return_long_term_fading_data=True)
            
            if render_perturbation:
                fig, ax = plt.subplots(1, 1, figsize = (10, 10))
                ax = self.render_perturbation(ax=ax)
            
            self.channel.tx_rx_locs['tx'] = tx_locs_new
            self.channel.tx_rx_locs['rx'] = rx_locs_new
            self.channel.H_l_sqrt = H_l
            self.channel.large_scale_fading_dict = large_scale_fading_dict

            self.channel.safety_check_large_scale_fading_data()

            if render_perturbation:
                # with plt.rc_context(rc = {"lines.markeralpha": 0.5}):
                ax = self.render_perturbation(ax=ax, alpha = 0.5)

                if save_path is not None:
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(f'{save_path}/render_{perturbation_type}_perturbation.pdf', dpi = 300)
                plt.close(fig)
            
            channel_gains = self.channel.sample()
            self.channel.channel_gains = channel_gains

        elif perturbation_type == 'resample_shadowing_loss':
            # shadowing_loss_new = self.channel.large_scale_fading_dict["shadowing_loss"] + perturbation_params['delta_shadowing']

            D_TxRx = distance.cdist(self.channel.tx_rx_locs['tx'], self.channel.tx_rx_locs['rx'], 'euclidean')
            
            # Resample log-normal shadowing
            print(f"Resampling shadowing loss with sigma = {self.channel.shadowing}.")
            H_l, large_scale_fading_dict = long_term_fading(D_TxRx, shadowing=self.channel.shadowing,
                                                            shadowing_loss = None,
                                                            return_long_term_fading_data=True)
            
            if render_perturbation:
                fig, axs = plt.subplots(1, 2, figsize = (40, 20))
                ax = self.render_perturbation(ax=axs[0], perturbation_type = perturbation_type)
            
            self.channel.H_l_sqrt = H_l
            self.channel.large_scale_fading_dict = large_scale_fading_dict

            self.channel.safety_check_large_scale_fading_data()

            if render_perturbation:
                # with plt.rc_context(rc = {"lines.markeralpha": 0.5}):
                ax = self.render_perturbation(ax=axs[1], alpha = 0.5, perturbation_type = perturbation_type)

                if save_path is not None:
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(f'{save_path}/render_{perturbation_type}_perturbation.pdf', dpi = 300)
                plt.close(fig)
            
            channel_gains = self.channel.sample()
            self.channel.channel_gains = channel_gains


        else:
            raise ValueError(f"Invalid perturbation type: {perturbation_type}")
        

    def render_perturbation(self, ax, *args, **kwargs):
        perturbation_type = kwargs.pop('perturbation_type', 'perturb_locs')

        if perturbation_type == 'perturb_locs':
            return self.channel.plot_tx_rx_locs(ax=ax, *args, **kwargs)
        
        elif perturbation_type == 'perturb_angles':
            raise NotImplementedError
        
        elif perturbation_type == 'resample_shadowing_loss':
            return self.channel.plot_H_l(ax=ax, *args, **kwargs)

    def __getattr__(self, attr_name: str):
        if attr_name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{attr_name}' is prohibited")
        return getattr(self.channel, attr_name)
    
    def sample(self, *args, **kwargs):
        return self.channel.sample(*args, **kwargs)
    
    def get_channel_data(self):
        return self.channel.get_channel_data()
    
    def perturb_locs(self, *args, **kwargs):
        return self.channel.perturb_locs(*args, **kwargs)
    
    def perturb_angles(self, *args, **kwargs):
        return self.channel.perturb_angles(*args, **kwargs)
    
    def safety_check_large_scale_fading_data(self):
        return self.channel.safety_check_large_scale_fading_data()


























class PerturbedChannel(Channel):
    def __init__(self, m, n, R, tx_rx_locs_H_l = None, channel_gains = None, fast_fading = True, perturb_tx_rx_locs_factor = 0.0, permute_nodes = False):
        super().__init__(m, n, R, tx_rx_locs_H_l, channel_gains, fast_fading)

        self.perturb_tx_rx_locs(perturb_tx_rx_locs_factor=perturb_tx_rx_locs_factor, permute_nodes=permute_nodes)


    @property
    def shadowing(self):
        return 0.
    

    def permute_nodes(self, locTx, locRx, H_l, perm):

        print(f"Permuting the nodes for Channel. Perm = {perm}")

        n = m = len(perm)

        locTx_perm = np.zeros_like(locTx)
        locRx_perm = np.zeros_like(locRx)
        locTx_perm[perm] = locTx
        locRx_perm[perm] = locRx

        locTx = locTx_perm
        locRx = locRx_perm

        D_TxRx = distance.cdist(locTx, locRx, 'euclidean')
        # if np.quantile(D_TxRx, 0) < min_D_TxRx:
        #     continue
            
        L = UDN_PL(D_TxRx) + self.shadowing * np.random.randn(m, n) # Loss matrix in dB
        H_l = np.sqrt(np.power(10, -L / 10)) # large-scale fading matrix

        associations = (H_l == np.max(H_l, axis=0, keepdims=True))

        assert np.min(np.sum(associations, axis=1)) > 0, print("Permuting channel nodes messed up associations...")

        return locTx, locRx, H_l

    
    def perturb_tx_rx_locs(self, perturb_tx_rx_locs_factor, permute_nodes = False):

        if perturb_tx_rx_locs_factor is None:
            if permute_nodes == False:
                return
            else:
                locTx = np.copy(self.tx_rx_locs['tx'])
                locRx = np.copy(self.tx_rx_locs['rx'])
                H_l = np.copy(self.H_l_sqrt)

                perm = np.random.permutation(locTx.shape[0])
                locTx, locRx, H_l = self.permute_nodes(locTx, locRx, H_l, perm)

        else:
            delta_r = self.R * perturb_tx_rx_locs_factor
            print(f'Perturbing tx_rx_locs independently by {delta_r} meters...')

            print('self.tx_locs.shape: ', self.tx_rx_locs['tx'].shape) # [num_nodes, 2]

            n = m = self.tx_rx_locs['tx'].shape[0]

            locTx = np.copy(self.tx_rx_locs['tx'])
            locRx = np.copy(self.tx_rx_locs['rx'])
            H_l = np.copy(self.H_l_sqrt)

            while True:
                phi_rx = 2 * np.pi * np.random.uniform(size = (n,))
                phi_tx = 2 * np.pi * np.random.uniform(size = (n,))
                # r = np.sqrt(np.random.uniform(low=min_D_TxRx ** 2, high=max_D_TxRx ** 2, size=(n,)))
                r = delta_r 
                locTx = np.clip(locTx + np.stack((r * np.cos(phi_tx), r * np.sin(phi_tx)), axis=1), a_min= -self.R / 2, a_max=self.R / 2)
                locRx = np.clip(locRx + np.stack((r * np.cos(phi_rx), r * np.sin(phi_rx)), axis=1), a_min= -self.R / 2, a_max=self.R / 2)

                D_TxRx = distance.cdist(locTx, locRx, 'euclidean')
                # if np.quantile(D_TxRx, 0) < min_D_TxRx:
                #     continue
                    
                L = UDN_PL(D_TxRx) + self.shadowing * np.random.randn(m, n) # Loss matrix in dB
                H_l = np.sqrt(np.power(10, -L / 10)) # large-scale fading matrix

                associations = (H_l == np.max(H_l, axis=0, keepdims=True))
                if min(np.sum(associations, axis=1)) > 0: # each transmitter has at least one associated reciever

                    print('Average perturbation in Hl || H_l - H_l(perturbed)||^2_2: ', ((self.H_l_sqrt - H_l)**2).mean().item())

                    if permute_nodes:
                        perm = np.random.permutation(locTx.shape[0])
                        locTx, locRx, H_l = self.permute_nodes(locTx, locRx, H_l, perm)

                    break

                locTx = np.copy(self.tx_rx_locs['tx'])
                locRx = np.copy(self.tx_rx_locs['rx'])
                H_l = np.copy(self.H_l_sqrt)


        print(f'Updating tx-rx-locs and H_l with epsilon={perturb_tx_rx_locs_factor}-perturbation.')
        tx_rx_locs_H_l = {'locTx': locTx, 'locRx': locRx, 'H_l': H_l}

        self.tx_rx_locs = {'tx': tx_rx_locs_H_l['locTx'],
                           'rx': tx_rx_locs_H_l['locRx']
                           }
        self.H_l_sqrt = tx_rx_locs_H_l['H_l']

        # if channel_gains is None:
        channel_gains = self.sample()
        self.channel_gains = channel_gains



def create_channels(accelerator, m, n, density_mode, base_R, num_channels, channel_load_path,
                    fast_fading = True,
                    channel_save_path = None,
                    use_single_channel = False,
                    channel_seed = None,
                    channel_wrappers = None,
                    max_n_per_subnetwork = 50,
                    channel_kws = None
                    ):
    """
    Instantiate wireless channels. 
    Args:
        accelerator: pytorch lightning accelerator
        m: int - number of transmitters
        n: int - number of receivers
        density_mode: str - density mode for the transmitter and receiver locations
        base_R: float - radius of the network area
        num_channels: int - number of channels to create
        channel_load_path: str - path to load channel data from
        fast_fading: bool - whether to include fast fading
        channel_save_path: str - path to save channel data
        use_single_channel: bool - whether to use the same channel for all samples
        channel_seed: int - seed for the channel
        channel_wrappers: list - list of channel wrappers to apply to the channels

    Returns:
        channels_list: list - list of Channel objects
    """

    factory = ChannelFactory(accelerator=accelerator, m=m, n=n, density_mode=density_mode, base_R=base_R, num_channels=num_channels, channel_load_path=channel_load_path,
                             fast_fading=fast_fading, channel_save_path=channel_save_path, use_single_channel=use_single_channel, channel_seed=channel_seed,
                             max_n_per_subnetwork=max_n_per_subnetwork,
                             channel_kws=channel_kws)
    
    # Create channels
    channels_list = factory.generate_channel_objects(accelerator=accelerator)

    
    if channel_wrappers is not None and len(channel_wrappers):
        if channel_load_path is not None:
            # wrap_loaded_channels = channel_wrappers[0].perturbation_type not in ['none', None]
            wrap_loaded_channels = True
            
            if wrap_loaded_channels:
                accelerator.print(f'Applying channel wrappers to first loaded channel.')
                # channels_list = [channels_list[0] for _ in range(len(channels_list))]
                channels_list = [copy.deepcopy(channels_list[0]) for _ in range(len(channels_list))]  
                
                for wrapper in channel_wrappers:
                    channels_list = [wrapper(channel, channel_id = i) for i, channel in enumerate(channels_list)]
            else:
                accelerator.print(f'Channel wrappers are not applied to loaded wrapped channels.')


        else:
            for wrapper in channel_wrappers:
                channels_list = [wrapper(channel, channel_id = i) for i, channel in enumerate(channels_list)]
    
    # channels_list = [ChannelPerturbationWrapper(channel, perturbation_type='perturb_locs', perturbation_params={'delta_r': {"tx": 0.0, 'rx': 50.0}}) for channel in channels_list]

    # Save channel data
    if channel_save_path is not None:
        data_list = defaultdict(list)
        for channel in channels_list:
            data = channel.get_channel_data()
            for key, value in data.items():
                data_list[key].append(value)
    
        save_data = [data_list[key] for key in ['tx_rx_locs_H_l', 'channel_gains']]

        save_channel_data_to_path(save_data, channel_save_path)
        # torch.save(save_data, channel_save_path)

    return channels_list


# def create_channels(accelerator, m, n, density_mode, base_R, num_channels, channel_load_path,
#                     fast_fading = True,
#                     channel_save_path = None,
#                     use_single_channel = False,
#                     channel_seed = None
#                     ):
#     """
#     Instantiate wireless channels. 
#     """

#     if channel_load_path is not None and os.path.exists(channel_load_path):
#         print(f'Loading channel data from {channel_load_path}.')
#         tx_rx_locs_H_l, channel_gains = torch.load(channel_load_path)
#     else:
#         print(f'No channel data load path found at {channel_load_path}! Creating channel data...')
#         tx_rx_locs_H_l, channel_gains = list(zip(*[(None, None) for _ in range(num_channels)]))

    
#     channels_list = [Channel(m=m, n=n,
#                              R=get_network_area(m, density_mode, base_R = base_R),
#                              tx_rx_locs_H_l = loc_and_h_l,
#                              channel_gains=data,
#                              fast_fading=fast_fading,
#                              seed = channel_seed
#                              ) for loc_and_h_l, data in tqdm(zip(tx_rx_locs_H_l, channel_gains))]
    

#     if use_single_channel:
#         print(f"Repeating Channel 0 {len(channels_list)} times.")
#         channels_list = [channels_list[0] for _ in range(len(channels_list))]

#     # Save channel data
#     if channel_save_path is not None:
#         if accelerator is None or accelerator.is_local_main_process:
#             data_list = defaultdict(list)
#             for channel in channels_list:
#                 data = channel.get_channel_data()
#                 for key, value in data.items():
#                     data_list[key].append(value)
        
#             save_data = [data_list[key] for key in ['tx_rx_locs_H_l', 'channel_gains']]
#             torch.save(save_data, channel_save_path)

#     return channels_list



def create_perturbed_channels(accelerator, m, n, density_mode, base_R, num_channels, channel_load_path,
                              fast_fading = True,
                              channel_save_path = None,
                              perturb_tx_rx_locs_factors = None, permute_nodes = False,
                              use_single_channel = False):
    """
    Instantiate wireless channels. 
    """

    if channel_load_path is not None:
        print(f'Loading channel data from {channel_load_path}.')
        tx_rx_locs_H_l, channel_gains = load_channels_from_path(channel_load_path) # torch.load(channel_load_path)

        while not len(tx_rx_locs_H_l) == num_channels: # repeat the same graph
            print(f"len(tx_rx_locs) = {len(tx_rx_locs_H_l)} \noteq {num_channels} = num_channels. Deepcopying the last channel.")
            tx_rx_locs_H_l.append(copy.deepcopy(tx_rx_locs_H_l[-1]))
            channel_gains.append(copy.deepcopy(channel_gains[-1]))
    else:
        print(f'No channel data load path found! Creating channel data...')
        tx_rx_locs_H_l, channel_gains = list(zip(*[(None, None) for _ in range(num_channels)]))

    if not isinstance(perturb_tx_rx_locs_factors, list):
        perturb_tx_rx_locs_factors = [perturb_tx_rx_locs_factors for _ in range(num_channels)]

    if not isinstance(permute_nodes, list):
        permute_nodes = [permute_nodes for _ in range(num_channels)]

    print("Perturb_tx_rx_locs_factor: ", perturb_tx_rx_locs_factors)
    print("Permute channel nodes: ", permute_nodes)
    
    
    channels_list = [PerturbedChannel(m=m, n=n,
                                      R=get_network_area(m, density_mode, base_R = base_R),
                                      tx_rx_locs_H_l = loc_and_h_l,
                                      channel_gains=data,
                                      fast_fading=fast_fading,
                                      perturb_tx_rx_locs_factor=perturb_tx_rx_locs_factor, permute_nodes=permute_nodes
                                      ) for loc_and_h_l, data, perturb_tx_rx_locs_factor, permute_nodes in tqdm(zip(tx_rx_locs_H_l, channel_gains, perturb_tx_rx_locs_factors, permute_nodes))]

    if use_single_channel:
        print(f"Repeating Channel 0 {len(channels_list)} times.")
        channels_list = [channels_list[0] for _ in range(len(channels_list))]

    # Save channel data
    if channel_save_path is not None:
        if accelerator is None or accelerator.is_local_main_process:
            data_list = defaultdict(list)
            for channel in channels_list:
                data = channel.get_channel_data()
                for key, value in data.items():
                    data_list[key].append(value)

        
            save_data = [data_list[key] for key in ['tx_rx_locs_H_l', 'channel_gains']]

            save_channel_data_to_path(save_data, channel_save_path)
            # torch.save(save_data, channel_save_path)

    return channels_list


# def get_weighted_adjacency_matrices(H, H_l, base_assoc = None):

#     # associations = (H_l == np.max(H_l, axis=1, keepdims=True))

#     # if not min(np.sum(associations, axis=1)) > 0: # some transmitters don't have any associated recievers
#     #     associations = base_assoc

#     # else:
#     associations = (H_l == np.max(H_l, axis=1, keepdims=True))

#     # reshape the channel matrices to get the weighted adjacency matrices as the basis for GNNs
#     # instantaneous channel
#     num_samples, m, n, T = H.shape
#     A = np.zeros((num_samples, m+n, m+n, T))
#     A[:, :m, m:, :] = np.expand_dims(associations, 3) * H
#     A[:, m:, :m, :] = np.transpose((np.expand_dims((1 - associations), 3) * H), (0, 2, 1, 3))

#     # long-term channel
#     A_l = np.zeros((num_samples, m+n, m+n))
#     A_l[:, :m, m:] = associations * H_l
#     A_l[:, m:, :m] = np.transpose(((1 - associations) * H_l), (0, 2, 1))

#     return A, A_l, associations


def get_weighted_adjacency_matrices(H, H_l, base_assoc = None):
    """
    Get the weighted adjacency matrices for the instantaneous and long-term channels.
    Args:
        H: np.array of shape (num_samples, m, n, T) - instantaneous channel gains
        H_l: np.array of shape (num_samples, m+n, m+n) - long-term channel gains
        base_assoc: np.array of shape (num_samples, m, n) - base associations for the long-term channel gains
    
    Returns:
        A: np.array of shape (num_samples, m+n, m+n, T) - weighted adjacency matrices for the instantaneous channel gains
        A_l: np.array of shape (num_samples, m+n, m+n) - weighted adjacency matrices for the long-term channel gains
        associations: np.array of shape (num_samples, m, n) - associations for the long-term channel
    """
    
    
    associations = (H_l == np.max(H_l, axis=1, keepdims=True))
    
    if np.min(np.sum(associations, axis=-1)) == 0 and base_assoc is not None:

        if np.min(np.sum(base_assoc, axis=-1)) > 0:
            associations = base_assoc
            print("Some transmitters don't have any associated recievers. Using base associations which might be suboptimal for some tx-rx pairs.")
        else:
            print("Some transmitters don't have any associated recievers.")

    # reshape the channel matrices to get the weighted adjacency matrices as the basis for GNNs
    # instantaneous channel
    num_samples, m, n, T = H.shape
    A = np.zeros((num_samples, m+n, m+n, T))
    A[:, :m, m:, :] = np.expand_dims(associations, 3) * H
    A[:, m:, :m, :] = np.transpose((np.expand_dims((1 - associations), 3) * H), (0, 2, 1, 3))

    # long-term channel
    A_l = np.zeros((num_samples, m+n, m+n))
    A_l[:, :m, m:] = associations * H_l
    A_l[:, m:, :m] = np.transpose(((1 - associations) * H_l), (0, 2, 1))

    return A, A_l, associations



def get_node_features(num_nodes, feature_type = 'ones', gg = None, snr = None):
    if feature_type == 'ones':
        y = np.ones((num_nodes, 1))
    elif feature_type == 'log-channel-gains':
        y = np.log(np.diagonal(gg) * snr).reshape(-1, 1)
        y = y / normalize_matrix(y, ord=float('inf'), axis=None)
    elif feature_type == 'log-one-plus-channel-gains':
        y = np.log(1 + np.diagonal(gg) * snr).reshape(-1, 1)
        y = y / normalize_matrix(y, ord=float('inf'), axis=None)
    else:
        raise NotImplementedError
    
    return y


def get_avg_graph(data_list):

    # Batch the graphs
    batch = Batch.from_data_list(data_list)

    # Average edge attributes (if edge indices are the same)
    edge_weight_l = batch.edge_weight_l.view(len(data_list), -1).mean(dim=0)

    # The edge_index remains the same, so we can take it from any of the graphs
    edge_index_l = data_list[0].edge_index_l

    # Create a new averaged graph
    avg_graph = Data(edge_index_l=edge_index_l, edge_weight_l=edge_weight_l)

    return avg_graph


# def create_channel_dataset(accelerator,
#                            channels_list, P_max_dBm, BW, noise_PSD_dBm, r_min, loggers = None,
#                            normalization = "spectral", eval_baselines = None, avg_graph_data = None,
#                            edge_sparsity = 1., edge_threshold = None, channel_conversion_method = None, channel_normalization_method = 'matrix-norm', permute_graph = False):

#     NODE_FEATURE_TYPE = 'ones' # 'log-one-plus-channel-gains'

#     CHANNEL_CONVERSION_METHOD = channel_conversion_method
#     SPARSIFY_EDGES = True if edge_sparsity < 1. or edge_threshold is not None else False

#     CHANNEL_NORMALIZATION_METHOD = channel_normalization_method

#     if normalization == "spectral":
#         CHANNEL_CONVERSION_METHOD = 'log' if CHANNEL_CONVERSION_METHOD is None else CHANNEL_CONVERSION_METHOD # 'correlation'
#         norm = 'sym'
#     else:
#         # CHANNEL_CONVERSION_METHOD = 'log' # 'correlation'
#         # CHANNEL_CONVERSION_METHOD = 'log-one-plus'
#         CHANNEL_CONVERSION_METHOD = 'correlation' if CHANNEL_CONVERSION_METHOD is None else CHANNEL_CONVERSION_METHOD

#     H = []
#     H_l = []
#     Assocs = []
#     Locs = []
#     for channel in channels_list:
#         base_assoc = channel.tx_rx_associations if hasattr(channel, 'tx_rx_associations') else None
#         tx_rx_locs = channel.tx_rx_locs
#         h, h_l = channel.channel_gains.values()
#         H.append(h)
#         H_l.append(h_l)
#         Assocs.append(base_assoc)
#         Locs.append(tx_rx_locs)

#     H = np.stack(H, axis = 0)
#     H_l = np.stack(H_l, axis = 0)
#     Assocs = np.stack(Assocs, axis = 0)
#     Locs = np.stack(Locs, axis = 0)

#     A, A_l, associations = get_weighted_adjacency_matrices(H=H, H_l=H_l, base_assoc=Assocs)

#     num_samples, m, n, T = H.shape

#     ############################### create pyg graphs ###############################
#     data_list = []
#     # y = torch.ones(n, 1)

#     avg_sparsification = 0.

#     P_max, noise_var = convert_P_max_and_noise_PSD(P_max_dBm, BW, noise_PSD_dBm)
#     snr = P_max / noise_var
#     for i in tqdm(range(num_samples), desc="Creating pyg graphs"):
#         a, a_l, h, h_l = A[i], A_l[i], H[i], H_l[i]

#         pos = {}
#         pos['tx'] = 0.5 + torch.from_numpy(channels_list[i].tx_rx_locs['tx']).to(torch.float32) / channels_list[i].R # normalized range [0, 1]
#         pos['rx'] = 0.5 + torch.from_numpy(channels_list[i].tx_rx_locs['rx']).to(torch.float32) / channels_list[i].R # normalized range [0, 1]

#         # serving_transmitters = torch.Tensor(np.argmax(h_l, axis=0)).to(torch.long)
#         # serving_transmitters_new = torch.Tensor(np.argmax(associations[i], axis=0)).to(torch.long)
#         # print("[serving_transmitters, serving_transmitters_new] = ", torch.stack([serving_transmitters, serving_transmitters_new], dim = -1))
#         serving_transmitters = torch.Tensor(np.argmax(associations[i], axis=0)).to(torch.long)

#         weighted_adjacency = torch.Tensor(a).unsqueeze(0)
#         weighted_adjacency_l = torch.Tensor(a_l).unsqueeze(0)
#         gg = ((1 - associations[i]) * h_l)[serving_transmitters] + np.eye(n) * h_l[serving_transmitters]
#         normalized_log_channel_matrix, channel_matrix_norm = convert_channels(gg, snr, conversion=CHANNEL_CONVERSION_METHOD, normalization=CHANNEL_NORMALIZATION_METHOD, return_normalization_factor=True)
#         y_l = torch.from_numpy(get_node_features(num_nodes=n, feature_type=NODE_FEATURE_TYPE, gg = gg, snr = snr)).to(torch.float32)
#         edge_index_l, edge_weight_l = from_scipy_sparse_matrix(sparse.csr_matrix(normalized_log_channel_matrix))

#         if SPARSIFY_EDGES:
#             # Sparsify the graph for better learning
#             num_edges = len(edge_weight_l)
#             if edge_threshold is None:
#                 avg_degree = int(n * edge_sparsity)
#                 edge_index_l, edge_weight_l = GDC().sparsify_sparse(edge_index=edge_index_l, edge_weight=edge_weight_l,
#                                                                     num_nodes=n, method="threshold", avg_degree = avg_degree)
            
#             else:
#                 eps = edge_threshold / channel_matrix_norm
#                 edge_index_l, edge_weight_l = GDC().sparsify_sparse(edge_index=edge_index_l, edge_weight=edge_weight_l,
#                                                                     num_nodes=n, method="threshold", eps=eps)
            
#             edge_weight_l = edge_weight_l / torch.linalg.vector_norm(edge_weight_l)

#             avg_sparsification += (1 - len(edge_weight_l) / num_edges)

#         if normalization == 'spectral':
#             edge_index_l, edge_weight_l = get_laplacian(edge_index=edge_index_l, edge_weight=edge_weight_l, normalization=norm)
#             edge_weight_l[torch.isnan(edge_weight_l)] = 0.

#         elif normalization == 'gcn':
#             add_self_loops = True
#             improved = add_self_loops
#             edge_index_l, edge_weight_l = gcn_norm(  # yapf: disable
#                                 edge_index_l, edge_weight_l, n,
#                                 improved=improved, add_self_loops=add_self_loops)
#             accelerator.print("GCN normalization applied to edge_weight_l.")
        
#         else:
#             pass
        
#         all_edge_indices = []
#         all_edge_weights = []
#         all_y = []

#         for t in range(T):
#             if t < channels_list[i].T_warmup:
#                 p = P_max * torch.ones(m)
#                 gamma = torch.zeros(n)
#                 selected_rxs = []

#                 for tx in range(m):
#                     associated_receivers = np.where(weighted_adjacency[0, tx , m:, 0].detach().cpu().numpy() > 0)[0]
#                     selected_receiver = associated_receivers[t % len(associated_receivers)]
#                     selected_rxs.append(selected_receiver)
#                 selected_rxs = np.array(selected_rxs)
                
#                 gamma[selected_rxs] = 1
#                 sampled_gamma = gamma
#                 rates = calc_rates(p, sampled_gamma, weighted_adjacency[:, :, :, t], noise_var)

#             else:
#                 gg = ((1 - associations[i]) * h[:, :, t])[serving_transmitters] + np.eye(n) * h[:, :, t][serving_transmitters]
#                 normalized_log_channel_matrix, channel_matrix_norm = convert_channels(gg, snr, conversion=CHANNEL_CONVERSION_METHOD, normalization=CHANNEL_NORMALIZATION_METHOD, return_normalization_factor=True)
#                 y_t = torch.from_numpy(get_node_features(num_nodes=n, feature_type=NODE_FEATURE_TYPE, gg = gg, snr = snr)).to(torch.float32)
#                 edge_index_t, edge_weights = from_scipy_sparse_matrix(sparse.csr_matrix(normalized_log_channel_matrix))

#                 if SPARSIFY_EDGES:
#                     # Sparsify the graph for better learning
#                     if edge_threshold is None:
#                         avg_degree = int(n * edge_sparsity)
#                         edge_index_t, edge_weights = GDC().sparsify_sparse(edge_index=edge_index_t, edge_weight=edge_weights,
#                                                                            num_nodes=n, method="threshold", avg_degree = avg_degree)
#                     else:
#                         eps = edge_threshold / channel_matrix_norm
#                         edge_index_t, edge_weights = GDC().sparsify_sparse(edge_index=edge_index_t, edge_weight=edge_weights,
#                                                                            num_nodes=n, method="threshold", eps=eps)

#                     edge_weights = edge_weights / torch.linalg.vector_norm(edge_weights)

#                 if normalization == 'spectral':
#                     edge_index_t, edge_weights = get_laplacian(edge_index=edge_index_t, edge_weight=edge_weights, normalization=norm)
#                     edge_weights[torch.isnan(edge_weights)] = 0.

#                 elif normalization == 'gcn':
#                     add_self_loops = True
#                     improved = add_self_loops
#                     edge_index_t, edge_weights = gcn_norm(  # yapf: disable
#                                         edge_index_t, edge_weights, n,
#                                         improved=improved, add_self_loops=add_self_loops)
#                     accelerator.print("GCN normalization applied to edge_weights.")

#                 else:
#                     pass

#                 all_edge_indices.append(edge_index_t)
#                 all_edge_weights.append(edge_weights.float())
#                 all_y.append(y_t.float())

#         data_list.append(Data_modTxIndex(pos=pos['tx'],
#                                          y=all_y,
#                                          y_l = y_l.float(),
#                                          edge_index_l=edge_index_l,
#                                          edge_weight_l=edge_weight_l.float(),
#                                          edge_index=all_edge_indices,
#                                          edge_weight=all_edge_weights,
#                                          weighted_adjacency=weighted_adjacency,
#                                          weighted_adjacency_l=weighted_adjacency_l,
#                                          transmitters_index=serving_transmitters,
#                                          num_nodes=n,
#                                          m=m,
#                                          )
#                         )

#     accelerator.print(f'Edge-sparsity: {edge_sparsity}\tEdge-thresh: {edge_threshold}\tEps: {eps}\t Avg graph sparsification rate: {avg_sparsification / num_samples}.')
        
#     if permute_graph:
#         accelerator.print("Applying random permutations to the graphs.")

#         for data in data_list:
#             data_perm, perm = permute_pygraph(data.clone())
#             data_perm.perm = perm 
#             data = data_perm
        

#     ### save average training graph weights for gnn-conditional-gnn-backbone ###
#     if False:
#         if avg_graph_data is None:
#             avg_graph_data = get_avg_graph(data_list=data_list) # get average training graph
#         else:
#             pass # copy the average training graph to other phases
        
#         avg_graph_edge_index_l, avg_graph_edge_weight_l = avg_graph_data.edge_index_l, avg_graph_data.edge_weight_l

#         assert_same_dtype_and_size(avg_graph_edge_index_l, data_list[0].edge_index_l)
#         assert_same_dtype_and_size(avg_graph_edge_weight_l, data_list[0].edge_weight_l)

#         for data in data_list:
#             data.avg_graph_edge_index_l = avg_graph_edge_index_l.clone()
#             data.avg_graph_edge_weight_l = avg_graph_edge_weight_l.clone()

    
#     ############################### Evaluate heuristic baselines ###############################
        
#     if eval_baselines is None or not len(eval_baselines):
#         baseline_metrics = None

#     else:
        
#         baseline_metrics = defaultdict(list)
#         for alg in eval_baselines:
#             # print(alg)

#             warmup_steps = min([channels_list[i].T_warmup for i in range(len(channels_list))])
            
#             # all_rates = torch.zeros((len(H), T - warmup_steps, n), dtype = torch.float32)
#             # all_Ps = torch.zeros((len(H), T - warmup_steps, n), dtype = torch.float32)
#             # all_rates = [torch.zeros((T - warmup_steps, n), dtype = torch.float32)] * len(H)
#             # all_Ps = [torch.zeros((T - warmup_steps, n), dtype = torch.float32)] * len(H)
#             all_rates_all_graphs = []
#             all_Ps_all_graphs = []

#             for i in tqdm(range(len(H)), desc=f"Evaluating {alg} baseline"):
#                 all_Ps = torch.zeros((T - warmup_steps, n), dtype = torch.float32)
#                 all_rates = torch.zeros((T - warmup_steps, n), dtype = torch.float32)
                
#                 a = A[i]
#                 weighted_avg_rates = 1e-10 * np.ones(n)
#                 mean_rates = np.zeros(n)
#                 for t in range(T):
#                     current_S = P_max * np.sum(a[:m, m:, t], axis=0)
#                     current_I = P_max * np.sum(a[m:, :m, t], axis=1)
#                     current_rates = np.log2(1 + current_S / (noise_var + current_I))
#                     PFs = current_rates / weighted_avg_rates
#                     selected_rxs = []
#                     for tx in range(m):
#                         if t < warmup_steps:
#                             associated_receivers = np.where(associations[i][tx, :] > 0)[0]
#                             selected_receiver = associated_receivers[t % len(associated_receivers)]
#                         else:
#                             masked_PFs = (associations[i][tx, :] > 0) * PFs
#                             selected_receiver = np.argmax(masked_PFs)
#                         selected_rxs.append(selected_receiver)
#                     h = H[i][:, selected_rxs, t]

#                     if t < warmup_steps:
#                         p = P_max * np.ones(m)
                    
#                     else:
#                         if alg == 'ITLinQ':
#                             p = ITLinQ(h, P_max, noise_var, PFs[selected_rxs])
#                         elif alg == 'WMMSE':
#                             p = wmmse(np.expand_dims(h, 0), P_max, noise_var)[0]
#                         elif alg == 'FR':
#                             p = P_max * np.ones(m)
#                         elif alg == 'UR': # uniform random
#                             p = P_max * np.random.rand(m)
#                         else:
#                             raise Exception

#                     h_power_adjusted = np.expand_dims(p, 1) * h
#                     S = np.diag(h_power_adjusted)
#                     I = np.sum(h_power_adjusted, axis=0) - S
#                     rates = np.zeros(n)
#                     rates[selected_rxs] = np.log2(1 + S / (noise_var + I))
#                     if t >= warmup_steps:
#                         mean_rates += rates
#                         all_rates[t-warmup_steps] = torch.from_numpy(rates)
#                         all_Ps[t-warmup_steps] = torch.from_numpy(p)
#                         # all_rates[i][t-warmup_steps] = torch.from_numpy(rates)
#                         # all_Ps[i][t-warmup_steps] = torch.from_numpy(p)
#                 mean_rates /= (T - warmup_steps)
#                 baseline_metrics[alg, 'mean_rates'].extend(mean_rates.tolist())

#                 all_rates_all_graphs.append(all_rates)
#                 all_Ps_all_graphs.append(all_Ps)
            
#             # baseline_metrics[alg, 'rates'] = torch.permute(all_rates, dims = (1, 0, 2))
#             # baseline_metrics[alg, 'Ps'] = torch.permute(all_Ps, dims = (1, 0, 2))
#             baseline_metrics[alg, 'rates'] = all_rates_all_graphs # torch.permute(all_rates, dims = (1, 0, 2))
#             baseline_metrics[alg, 'Ps'] = all_Ps_all_graphs # torch.permute(all_Ps, dims = (1, 0, 2))


#             accelerator.print(f'*****************************************\nBaseline alg. {alg} ergodic rates evaluated over {len(all_rates_all_graphs)} graphs, {all_rates_all_graphs[0].shape[0]} timesteps.')
#             accelerator.print(f"Alg = {alg}\tAvg. min. ergodic rate = {np.mean([rate.mean(dim = 0).min().item() for rate in baseline_metrics[alg, 'rates']])}\tAbsolute min. ergodic rate = {np.min([rate.mean(dim = 0).min().item() for rate in baseline_metrics[alg, 'rates']])}\n")
#             q = 1
#             # print(f"{np.mean([np.percentile(rate.mean(dim = 0), q = q).item() for rate in baseline_metrics[alg, 'rates']]).item()}")
#             # print(f"{np.percentile(np.array([rate.mean(dim = 0).detach().cpu().numpy() for rate in baseline_metrics[alg, 'rates']]), q = q).item()}")
#             accelerator.print(f"Alg = {alg}\tAvg. {q}% ergodic rate = {np.mean([np.percentile(rate.mean(dim = 0), q = q).item() for rate in baseline_metrics[alg, 'rates']]).item()}\tAbsolute {q}% ergodic rate = {np.percentile(np.array([rate.mean(dim = 0).detach().cpu().numpy() for rate in baseline_metrics[alg, 'rates']]), q = q).item()}\n")
#             accelerator.print(f"Alg = {alg}\tAvg. {int(5*q)}% ergodic rate = {np.mean([np.percentile(rate.mean(dim = 0), q = int(5 * q)).item() for rate in baseline_metrics[alg, 'rates']]).item()}\tAbsolute {int(5*q)}% ergodic rate = {np.percentile(np.array([rate.mean(dim = 0).detach().cpu().numpy() for rate in baseline_metrics[alg, 'rates']]), q = int(5 * q)).item()}\n*****************************************")
        

#     ############################### Create channel dataset loggers ###############################
#     if loggers is not None:
#         # if accelerator is None or accelerator.is_local_main_process:
#         for logger in loggers:
#             if logger.log_metric in ['rates']:
#                 if logger.network_id is not None:
#                     avg_rates = [baseline_metrics[alg, 'rates'][logger.network_id].mean(dim = 0, keepdim = True) for alg in eval_baselines]
#                     logger.barplot_opt_problem(avg_rates = avg_rates, metric_names = eval_baselines) if logger.network_id < 2 else None
#                 else:
#                     avg_rates = [torch.stack(baseline_metrics[alg, 'rates'], dim = 0).mean(dim = 1).view(1, -1) for alg in eval_baselines]
#                     logger.barplot_opt_problem(avg_rates = avg_rates, metric_names = eval_baselines)

#                 # for alg in eval_baselines:
#                 #     avg_rates = baseline_metrics[alg, 'rates'][logger.network_id].mean(dim = 0, keepdim = True)
#                 #     logger.barplot_opt_problem(avg_rates=avg_rates, metric_names = [alg])
#             elif logger.log_metric in ['tx-rx-locs']:
#                 logger.update_data({'tx-rx-locs': [{"tx": loc['tx'], "rx": loc['rx'],
#                                                     "associations": assoc, "H_l": h_l,
#                                                     "P_max": P_max, "noise_var": noise_var, "r_min": r_min,
#                                                     } for loc, assoc, h_l in zip(Locs, Assocs, H_l)]})
#                 logger()
#             else:
#                 logger.update_data(data_list)
#                 logger()
        

#     # import torch_geometric
#     # import networkx as nx
#     # import matplotlib.pyplot as plt
#     # # from torch_geometric.explain import Explanation
#     # # exp = Explanation(data, edge_index=data.edge_index_l, edge_mask = None)
#     # # exp.visualize_graph(path='./data/example_graph.png',
#     # #                     backend='networkx'
#     # #                     )
#     # data = torch_geometric.data.Data(x=data_list[-1].y, edge_index=data_list[-1].edge_index_l)

#     # fig = plt.figure(figsize=(12,12))
#     # g = torch_geometric.utils.to_networkx(data, to_undirected=True)
#     # nx.draw(g)
#     # plt.savefig('./data/example_graph_nx.png', dpi = 300)

#     return data_list, baseline_metrics, avg_graph_data




def create_channel_dataset(accelerator,
                           channels_list, P_max_dBm, BW, noise_PSD_dBm, r_min, loggers = None,
                           normalization = "spectral", eval_baselines = None, avg_graph_data = None,
                           edge_sparsity = 1., edge_threshold = None,
                           channel_conversion_method = None, channel_normalization_method = 'matrix-norm', permute_graph = False,
                           channel_features = ('FR', 'rates')
                           ):
    
    print("Creating channel dataset...")

    NODE_FEATURE_TYPE = 'ones' # 'log-one-plus-channel-gains'

    CHANNEL_CONVERSION_METHOD = channel_conversion_method
    SPARSIFY_EDGES = True if edge_sparsity < 1. or edge_threshold is not None else False

    CHANNEL_NORMALIZATION_METHOD = channel_normalization_method

    if normalization == "spectral":
        CHANNEL_CONVERSION_METHOD = 'log' if CHANNEL_CONVERSION_METHOD is None else CHANNEL_CONVERSION_METHOD # 'correlation'
        norm = 'sym'
    else:
        # CHANNEL_CONVERSION_METHOD = 'log' # 'correlation'
        # CHANNEL_CONVERSION_METHOD = 'log-one-plus'
        CHANNEL_CONVERSION_METHOD = 'correlation' if CHANNEL_CONVERSION_METHOD is None else CHANNEL_CONVERSION_METHOD

    H = []
    H_l = []
    Assocs = []
    Locs = []
    # for channel in channels_list:
    #     h, h_l = channel.channel_gains.values()
    #     H.append(h)
    #     H_l.append(h_l)
    # H = np.stack(H, axis = 0)
    # H_l = np.stack(H_l, axis = 0)
    accelerator.print("Appending channel matrices...")
    for channel in channels_list:
        base_assoc = channel.tx_rx_associations if hasattr(channel, 'tx_rx_associations') else None
        tx_rx_locs = channel.tx_rx_locs
        h, h_l = channel.channel_gains.values()
        H.append(h)
        H_l.append(h_l)
        Assocs.append(base_assoc)
        Locs.append(tx_rx_locs)

    accelerator.print("Stacking channel matrices...")
    H = np.stack(H, axis = 0)
    H_l = np.stack(H_l, axis = 0)
    accelerator.print("H and H_l matrices are stacked...")
    Assocs = np.stack(Assocs, axis = 0)
    Locs = np.stack(Locs, axis = 0)
    accelerator.print("Assocs and Locs matrices are stacked...")

    A, A_l, associations = get_weighted_adjacency_matrices(H=H, H_l=H_l, base_assoc=Assocs)
    num_samples, m, n, T = H.shape

    print("Weighted adjacency matrices are created...")
    print("Num_samples: ", num_samples)
    ############################### create pyg graphs ###############################
    data_list = []
    # y = torch.ones(n, 1)

    avg_sparsification = 0.

    P_max, noise_var = convert_P_max_and_noise_PSD(P_max_dBm, BW, noise_PSD_dBm)
    snr = P_max / noise_var

    print("P_max: ", P_max)
    print("Noise variance: ", noise_var)
    print("SNR: ", snr)


    # for i in tqdm(range(num_samples), desc="Creating pyg graphs"):
    for i in range(num_samples):
        print(f"i: {i} / {num_samples}")
        a, a_l, h, h_l = A[i], A_l[i], H[i], H_l[i]

        pos = {}
        pos['tx'] = 0.5 + torch.from_numpy(channels_list[i].tx_rx_locs['tx']).to(torch.float32) / channels_list[i].R # normalized range [0, 1]
        pos['rx'] = 0.5 + torch.from_numpy(channels_list[i].tx_rx_locs['rx']).to(torch.float32) / channels_list[i].R # normalized range [0, 1]

        # serving_transmitters = torch.Tensor(np.argmax(h_l, axis=0)).to(torch.long)
        serving_transmitters = torch.Tensor(np.argmax(associations[i], axis=0)).to(torch.long)

        weighted_adjacency = torch.Tensor(a).unsqueeze(0)
        weighted_adjacency_l = torch.Tensor(a_l).unsqueeze(0)
        gg = ((1 - associations[i]) * h_l)[serving_transmitters] + np.eye(n) * h_l[serving_transmitters]
        normalized_log_channel_matrix, channel_matrix_norm = convert_channels(gg, snr, conversion=CHANNEL_CONVERSION_METHOD, normalization=CHANNEL_NORMALIZATION_METHOD, return_normalization_factor=True)
        y_l = torch.from_numpy(get_node_features(num_nodes=n, feature_type=NODE_FEATURE_TYPE, gg = gg, snr = snr)).to(torch.float32)
        edge_index_l, edge_weight_l = from_scipy_sparse_matrix(sparse.csr_matrix(normalized_log_channel_matrix))

        if SPARSIFY_EDGES:
            # Sparsify the graph for better learning
            num_edges = len(edge_weight_l)
            if edge_threshold is None:
                avg_degree = int(n * edge_sparsity)
                edge_index_l, edge_weight_l = GDC().sparsify_sparse(edge_index=edge_index_l, edge_weight=edge_weight_l,
                                                                    num_nodes=n, method="threshold", avg_degree = avg_degree)
            
            else:
                eps = edge_threshold / channel_matrix_norm
                edge_index_l, edge_weight_l = GDC().sparsify_sparse(edge_index=edge_index_l, edge_weight=edge_weight_l,
                                                                    num_nodes=n, method="threshold", eps=eps)
            
            edge_weight_l = edge_weight_l / torch.linalg.vector_norm(edge_weight_l)

            avg_sparsification += (1 - len(edge_weight_l) / num_edges)

        if normalization == 'spectral':
            edge_index_l, edge_weight_l = get_laplacian(edge_index=edge_index_l, edge_weight=edge_weight_l, normalization=norm)
            edge_weight_l[torch.isnan(edge_weight_l)] = 0.

        elif normalization == 'gcn':
            add_self_loops = True
            improved = add_self_loops
            edge_index_l, edge_weight_l = gcn_norm(  # yapf: disable
                                edge_index_l, edge_weight_l, n,
                                improved=improved, add_self_loops=add_self_loops)
            print("GCN normalization applied to edge_weight_l.")
        
        else:
            pass
        
        all_edge_indices = []
        all_edge_weights = []
        all_y = []

        for t in range(T):
            if t < channels_list[i].T_warmup:
                p = P_max * torch.ones(m)
                gamma = torch.zeros(n)
                selected_rxs = []

                for tx in range(m):
                    associated_receivers = np.where(weighted_adjacency[0, tx , m:, 0].detach().cpu().numpy() > 0)[0]
                    selected_receiver = associated_receivers[t % len(associated_receivers)]
                    selected_rxs.append(selected_receiver)
                selected_rxs = np.array(selected_rxs)
                
                gamma[selected_rxs] = 1
                sampled_gamma = gamma
                rates = calc_rates(p, sampled_gamma, weighted_adjacency[:, :, :, t], noise_var)

            else:
                gg = ((1 - associations[i]) * h[:, :, t])[serving_transmitters] + np.eye(n) * h[:, :, t][serving_transmitters]
                normalized_log_channel_matrix, channel_matrix_norm = convert_channels(gg, snr, conversion=CHANNEL_CONVERSION_METHOD, normalization=CHANNEL_NORMALIZATION_METHOD, return_normalization_factor=True)
                y_t = torch.from_numpy(get_node_features(num_nodes=n, feature_type=NODE_FEATURE_TYPE, gg = gg, snr = snr)).to(torch.float32)
                edge_index_t, edge_weights = from_scipy_sparse_matrix(sparse.csr_matrix(normalized_log_channel_matrix))

                if SPARSIFY_EDGES:
                    # Sparsify the graph for better learning
                    if edge_threshold is None:
                        avg_degree = int(n * edge_sparsity)
                        edge_index_t, edge_weights = GDC().sparsify_sparse(edge_index=edge_index_t, edge_weight=edge_weights,
                                                                           num_nodes=n, method="threshold", avg_degree = avg_degree)
                    else:
                        eps = edge_threshold / channel_matrix_norm
                        edge_index_t, edge_weights = GDC().sparsify_sparse(edge_index=edge_index_t, edge_weight=edge_weights,
                                                                           num_nodes=n, method="threshold", eps=eps)

                    edge_weights = edge_weights / torch.linalg.vector_norm(edge_weights)

                if normalization == 'spectral':
                    edge_index_t, edge_weights = get_laplacian(edge_index=edge_index_t, edge_weight=edge_weights, normalization=norm)
                    edge_weights[torch.isnan(edge_weights)] = 0.

                elif normalization == 'gcn':
                    add_self_loops = True
                    improved = add_self_loops
                    edge_index_t, edge_weights = gcn_norm(  # yapf: disable
                                        edge_index_t, edge_weights, n,
                                        improved=improved, add_self_loops=add_self_loops)
                    print("GCN normalization applied to edge_weights.")

                else:
                    pass

                all_edge_indices.append(edge_index_t)
                all_edge_weights.append(edge_weights.float())
                all_y.append(y_t.float())

        data_list.append(Data_modTxIndex(pos=pos['tx'],
                                         network_id = i,
                                         y=all_y,
                                         y_l = y_l.float(),
                                         edge_index_l=edge_index_l,
                                         edge_weight_l=edge_weight_l.float(),
                                         edge_index=all_edge_indices,
                                         edge_weight=all_edge_weights,
                                         weighted_adjacency=weighted_adjacency,
                                         weighted_adjacency_l=weighted_adjacency_l,
                                         transmitters_index=serving_transmitters,
                                         num_nodes=n,
                                         m=m,
                                         )
                        )

    print(f'Edge-sparsity: {edge_sparsity}\tEdge-thresh: {edge_threshold}\tEps: {eps}\t Avg graph sparsification rate: {avg_sparsification / num_samples}.')
        
    if permute_graph and False:
        accelerator.print("Applying random permutations to the graphs.")

        for data in data_list:
            data_perm, perm = permute_pygraph(data.clone())
            data_perm.perm = perm 
            data = data_perm
        
        

    ### save average training graph weights for gnn-conditional-gnn-backbone ###
    if False:
        if avg_graph_data is None:
            avg_graph_data = get_avg_graph(data_list=data_list) # get average training graph
        else:
            pass # copy the average training graph to other phases
        
        avg_graph_edge_index_l, avg_graph_edge_weight_l = avg_graph_data.edge_index_l, avg_graph_data.edge_weight_l

        assert_same_dtype_and_size(avg_graph_edge_index_l, data_list[0].edge_index_l)
        assert_same_dtype_and_size(avg_graph_edge_weight_l, data_list[0].edge_weight_l)

        for data in data_list:
            data.avg_graph_edge_index_l = avg_graph_edge_index_l.clone()
            data.avg_graph_edge_weight_l = avg_graph_edge_weight_l.clone()

    
    ############################### Evaluate heuristic baselines ###############################
    print("Evaluating heuristic baselines...")
    if (eval_baselines in [None, "none", "None"] or not len(eval_baselines)) or (isinstance(eval_baselines, list) and eval_baselines[0] in [None, "none", "None"]): # or True
        baseline_metrics = None
        print("No heuristic baselines to evaluate.")

        for data in data_list:
            # Set dummy attributes for compatibility

            y_l = torch.zeros(size = (data.num_nodes,), dtype = torch.float32)
            y = [torch.zeros_like(y_l) for t in range(len(data.edge_index))]

            X_l = torch.zeros(size = (data.num_nodes, 1), dtype = torch.float32)
            X = [torch.zeros(size = (data.num_nodes, 1), dtype = torch.float32) for t in range(len(data.edge_index))]

            data.x_l = X_l
            data.x = X
            data.y_l = y_l
            data.y = y

        return data_list, baseline_metrics, avg_graph_data

    else:
        
        baseline_metrics = defaultdict(list)
        baseline_datasets_log_dict = {}
        for alg in eval_baselines:
            accelerator.print(alg)

            # # Log the ergodic rates for each graph 
            # if alg.lower() == 'fr':
            #     dummy_sampler = FRLambdaSampler(device = "cpu", r_min = r_min, n_lambdas = n, lambdas_max=None, T_0 = T, eval_sa_learner_fn=None)
            # elif alg.lower() == 'ur':
            #     dummy_sampler = URLambdaSampler(device = "cpu", r_min = r_min, n_lambdas = n, lambdas_max=None, T_0 = T, eval_sa_learner_fn=None)
            # elif alg.lower() == 'itlinq':
            #     pass
            #     # dummy_sampler = ITLinQLambdaSampler(device = "cpu", r_min = r_min)
            # elif alg.lower() == 'wmmse':
            #     dummy_sampler = WMMSELambdaSampler(device = "cpu", r_min = r_min, n_lambdas = n, lambdas_max=None, T_0 = T, eval_sa_learner_fn=None)
            # else:
            #     raise ValueError(f"Unknown algorithm: {alg}")

            warmup_steps = min([channels_list[i].T_warmup for i in range(len(channels_list))])
            
            # all_rates = torch.zeros((len(H), T - warmup_steps, n), dtype = torch.float32)
            # all_Ps = torch.zeros((len(H), T - warmup_steps, n), dtype = torch.float32)
            # all_rates = [torch.zeros((T - warmup_steps, n), dtype = torch.float32)] * len(H)
            # all_Ps = [torch.zeros((T - warmup_steps, n), dtype = torch.float32)] * len(H)
            all_rates_all_graphs = []
            all_Ps_all_graphs = []
            all_signal_powers_all_graphs = []
            all_interference_powers_all_graphs = []

            for i in tqdm(range(len(H)), desc=f"Evaluating {alg} baseline"):
                all_Ps = torch.zeros((T - warmup_steps, n), dtype = torch.float32)
                all_rates = torch.zeros((T - warmup_steps, n), dtype = torch.float32)

                all_signal_powers = torch.zeros((T - warmup_steps, n), dtype = torch.float32)
                all_interference_powers = torch.zeros((T - warmup_steps, n), dtype = torch.float32)
                
                a = A[i]
                weighted_avg_rates = 1e-10 * np.ones(n)
                mean_rates = np.zeros(n)
                for t in range(T):
                    # print(f"t = {t}")
                    current_S = P_max * np.sum(a[:m, m:, t], axis=0)
                    current_I = P_max * np.sum(a[m:, :m, t], axis=1)
                    current_rates = np.log2(1 + current_S / (noise_var + current_I))
                    PFs = current_rates / weighted_avg_rates
                    selected_rxs = []
                    for tx in range(m):
                        if t < warmup_steps:
                            associated_receivers = np.where(associations[i][tx, :] > 0)[0]
                            selected_receiver = associated_receivers[t % len(associated_receivers)]
                        else:
                            masked_PFs = (associations[i][tx, :] > 0) * PFs
                            selected_receiver = np.argmax(masked_PFs)
                        selected_rxs.append(selected_receiver)
                    h = H[i][:, selected_rxs, t]

                    if t < warmup_steps:
                        p = P_max * np.ones(m)
                    
                    else:
                        if alg.lower() == 'itlinq':
                            p = ITLinQ(h, P_max, noise_var, PFs[selected_rxs])
                        elif alg.lower() == 'wmmse':
                            wmmse_num_iters = 10 # 100
                            p = wmmse(np.expand_dims(h, 0), P_max, noise_var, num_iters = wmmse_num_iters)[0]
                        elif alg.lower() == 'fr':
                            p = P_max * np.ones(m)
                        elif alg.lower() == 'ur': # uniform random
                            p = P_max * np.random.rand(m)
                        else:
                            raise Exception

                    h_power_adjusted = np.expand_dims(p, 1) * h
                    S = np.diag(h_power_adjusted)
                    I = np.sum(h_power_adjusted, axis=0) - S
                    rates = np.zeros(n)
                    rates[selected_rxs] = np.log2(1 + S / (noise_var + I))
                    if t >= warmup_steps:
                        mean_rates += rates
                        all_rates[t-warmup_steps] = torch.from_numpy(rates)
                        all_Ps[t-warmup_steps] = torch.from_numpy(p)
                        # all_rates[i][t-warmup_steps] = torch.from_numpy(rates)
                        # all_Ps[i][t-warmup_steps] = torch.from_numpy(p)

                        all_signal_powers[t - warmup_steps] = torch.from_numpy(S.copy())
                        # all_interference_powers[t - warmup_steps] = torch.from_numpy(I.copy() + noise_var)
                        all_interference_powers[t - warmup_steps] = torch.from_numpy(I.copy() / (0 + 1 * noise_var))

                        if t <= (warmup_steps + 2):
                            # Print selected receivers for the {t - warmup_steps}th timestep
                            accelerator.print(f"Selected receivers for the {t - warmup_steps}th timestep: {selected_rxs}")
                            # Print the normalized interference powers for the {t - warmup_steps}th timestep
                            accelerator.print(f"Normalized interference powers for the {t - warmup_steps}th timestep: {all_interference_powers[t - warmup_steps].numpy()}")


                # Compute rate based on long-term channel only now for FR baseline
                if alg == "FR":
                    h_l = H_l[i][:, selected_rxs] # use last selected receivers
                    p_l = P_max * np.ones(m)
                    h_l_power_adjusted = np.expand_dims(p_l, 1) * h_l

                    S_l = np.diag(h_l_power_adjusted)
                    I_l = np.sum(h_l_power_adjusted, axis=0) - S_l
                    rates_l = np.zeros(n)
                    rates_l[selected_rxs] = np.log2(1 + S_l / (noise_var + I_l))

                    # baseline_metrics[alg, "long-term_rates"].extend(rates_l.tolist())
                    baseline_metrics[alg, "long-term_rates"].append(rates_l.tolist())


                mean_rates /= (T - warmup_steps)
                baseline_metrics[alg, 'mean_rates'].extend(mean_rates.tolist())

                # for i in range(MAX_LOGGED_NETWORKS):
                #     dummy_sampler.update_log_dict(key = dummy_sampler.make_log_dict_key('ergodic-rates'), value = mean_rates, idx = i)
                #     dummy_sampler.update_log_dict(key = dummy_sampler.make_log_dict_key('rates'), value = all_rates, idx = i)
                #     dummy_sampler.update_log_dict(key = dummy_sampler.make_log_dict_key('Ps'), value = all_Ps / P_max, idx = i)
                #     dummy_sampler.update_log_dict(key = dummy_sampler.make_log_dict_key('interference-powers'), value = torch.log10(1e-10 + all_interference_powers), idx = i)

                all_rates_all_graphs.append(all_rates)
                all_Ps_all_graphs.append(all_Ps)
                all_signal_powers_all_graphs.append(all_signal_powers)
                all_interference_powers_all_graphs.append(all_interference_powers)


            # baseline_datasets_log_dict.update({f"{alg}-Dataset/{k}": v for k,v in dummy_sampler.log_dict.items()})
            
            # baseline_metrics[alg, 'rates'] = torch.permute(all_rates, dims = (1, 0, 2))
            # baseline_metrics[alg, 'Ps'] = torch.permute(all_Ps, dims = (1, 0, 2))
            baseline_metrics[alg, 'rates'] = all_rates_all_graphs # torch.permute(all_rates, dims = (1, 0, 2))
            baseline_metrics[alg, 'Ps'] = all_Ps_all_graphs # torch.permute(all_Ps, dims = (1, 0, 2))
            baseline_metrics[alg, 'signal_powers'] = all_signal_powers_all_graphs # torch.permute(all_signal_powers, dims = (1, 0, 2))
            baseline_metrics[alg, 'interference_powers'] = all_interference_powers_all_graphs

            accelerator.print(f'*****************************************\nBaseline alg. {alg} ergodic rates evaluated over {len(all_rates_all_graphs)} graphs, {all_rates_all_graphs[0].shape[0]} timesteps.')
            accelerator.print(f"Alg = {alg}\tAvg. min. ergodic rate = {np.mean([rate.mean(dim = 0).min().item() for rate in baseline_metrics[alg, 'rates']])}\tAbsolute min. ergodic rate = {np.min([rate.mean(dim = 0).min().item() for rate in baseline_metrics[alg, 'rates']])}\n")
            q = 1
            # print(f"{np.mean([np.percentile(rate.mean(dim = 0), q = q).item() for rate in baseline_metrics[alg, 'rates']]).item()}")
            # print(f"{np.percentile(np.array([rate.mean(dim = 0).detach().cpu().numpy() for rate in baseline_metrics[alg, 'rates']]), q = q).item()}")
            accelerator.print(f"Alg = {alg}\tAvg. {q}% ergodic rate = {np.mean([np.percentile(rate.mean(dim = 0), q = q).item() for rate in baseline_metrics[alg, 'rates']]).item()}\tAbsolute {q}% ergodic rate = {np.percentile(np.array([rate.mean(dim = 0).detach().cpu().numpy() for rate in baseline_metrics[alg, 'rates']]), q = q).item()}\n")
            accelerator.print(f"Alg = {alg}\tAvg. {int(5*q)}% ergodic rate = {np.mean([np.percentile(rate.mean(dim = 0), q = int(5 * q)).item() for rate in baseline_metrics[alg, 'rates']]).item()}\tAbsolute {int(5*q)}% ergodic rate = {np.percentile(np.array([rate.mean(dim = 0).detach().cpu().numpy() for rate in baseline_metrics[alg, 'rates']]), q = int(5 * q)).item()}\n*****************************************")

        accelerator.log(baseline_datasets_log_dict)

    # channel_features = ('FR', 'rates')
    # features = ('FR', 'interference_powers')
    # features = ('ITLinQ', 'Ps')
    #### Add regression features to the data_list here from FR baseline. ####
    for i, data in enumerate(data_list):

        if ('FR', 'long_term_rates') in baseline_metrics.keys():
        
            X_l = torch.tensor(baseline_metrics['FR', 'long-term_rates'][i], dtype = torch.float32).view(-1)
            r_min_tensor = torch.ones_like(X_l) * r_min
            X_l = torch.cat((X_l.view(-1, 1), r_min_tensor.view(-1, 1)), dim = 1) # add min rate as a constant feature.

            accelerator.print("X_l shape: ", X_l.shape)
            accelerator.print("r_min_tensor shape: ", r_min_tensor.shape)

            accelerator.print("baseline_metrics['FR', 'rates'][i][0].shape: ", baseline_metrics['FR', 'rates'][i][0].shape)

            X = [torch.cat([baseline_metrics['FR', 'rates'][i][t].clone().detach().view(-1, 1), r_min_tensor.view(-1, 1)], dim = 1) for t in range(len(data.edge_index))]

        else:
            X_l = torch.zeros(size = (data.num_nodes,), dtype = torch.float32)
            r_min_tensor = torch.ones_like(X_l) * r_min
            X_l = torch.cat((X_l.view(-1, 1), r_min_tensor.view(-1, 1)), dim = 1) # add min rate as a constant feature.

            X = [torch.cat([torch.zeros(size = (data.num_nodes,), dtype = torch.float32).view(-1, 1), r_min_tensor.view(-1, 1)], dim = 1) for t in range(len(data.edge_index))]

        
        y_l = torch.zeros(size = (data.num_nodes,), dtype = torch.float32)
        y = [torch.zeros_like(y_l) for t in range(len(data.edge_index))]


        if channel_features == ('FR', 'rates'):
            pass
        
        elif any([channel_features[0] == algo and channel_features[1] == 'interference_powers' for algo in eval_baselines]):
            # Use interference powers as features
            log_normalization = True
            for algo in eval_baselines:
                if channel_features[0] == algo and channel_features[1] == 'interference_powers':
                    accelerator.print(f"Using {algo} interference powers as features.")
                    X = [baseline_metrics[algo, 'interference_powers'][i][t].clone().detach().view(-1, 1) for t in range(len(data.edge_index))]
                    y = [baseline_metrics[algo, 'rates'][i][t].clone().detach().view(-1, 1) for t in range(len(data.edge_index))]

                    if log_normalization:
                        accelerator.print(f"Using log10-normalized interference powers as diffusion-learning features.")
                        X = [torch.log10(temp) for temp in X]

                    break
                
                else:
                    continue


        elif channel_features == ('ITLinQ', 'Ps'):
            X = [baseline_metrics['ITLinQ', 'Ps'][i][t].clone().detach().view(-1, 1) / P_max - 1/2 for t in range(len(data.edge_index))]
            y = [baseline_metrics['ITLinQ', 'rates'][i][t].clone().detach().view(-1, 1) for t in range(len(data.edge_index))]

        elif channel_features == ('WMMSE', 'Ps'):
            X = [baseline_metrics['WMMSE', 'Ps'][i][t].clone().detach().view(-1, 1) / P_max - 1/2 for t in range(len(data.edge_index))]
            y = [baseline_metrics['WMMSE', 'rates'][i][t].clone().detach().view(-1, 1) for t in range(len(data.edge_index))]

        else:
            raise ValueError(f"Unknown features: {channel_features}. Expected ('FR', 'rates'), ('FR', 'interference_powers'), ('ITLinQ', 'Ps') or ('WMMSE', 'Ps').")
            X = None

    
        if i == 0:
            accelerator.print(f"X_l shape: {X_l.shape}\tX[0].shape: {X[0].shape}")
            accelerator.print("data.edge_index_l.shape: ", data.edge_index_l.shape)

        if channel_features == ('FR', 'rates'):
            pass

        else:
            X_l = torch.stack(X, dim = 0).mean(0).view(-1, 1) # .view(-1)
        # elif any([channel_features[0] == algo and channel_features[1] == 'interference_powers' for algo in eval_baselines]):
        #     # Use interference powers as features
        #     X_l = torch.stack(X, dim = 0).mean(0) # .view(-1)

        # elif channel_features == ('ITLinQ', 'Ps'):
        #     # Use ITLinQ Ps as features
        #     X_l = torch.stack(X, dim = 0).mean(0) # .view(-1)

        # elif channel_features == ('WMMSE', 'Ps'):
        #     # Use WMMSE Ps as features
        #     X_l = torch.stack(X, dim = 0).mean(0) # .view(-1)

        # else:
        #     raise ValueError(f"Unknown features: {channel_features}. Expected ('FR', 'rates'), ('FR', 'interference_powers'), ('ITLinQ', 'Ps') or ('WMMSE', 'Ps').")
        #     X_l = None

        accelerator.print("Average X_l: ", X_l.mean().item())

        data.x_l = X_l
        data.x = X
        data.y_l = y_l
        data.y = y


    accelerator.print("data_list[0]: ", data_list[0])
    accelerator.print("data_list[0].x_l: ", data_list[0].x_l)
    accelerator.print("data_list[0].x[::10][node = 14-15]", [data_list[0].x[t * len(data.edge_index) // 10][14:16] for t in range(10)])


    fig, ax = plt.subplots(1, 1, figsize = (24, 8))
    X_over_time = np.stack([data.x[t][14:16, 0].cpu().numpy() for t in range(len(data.x))], axis = 0)
    # X_l_over_time = np.stack([data.x_l[14:16, 0].cpu().numpy() for t in range(len(data.x))], axis = 0)
    accelerator.print("X_over_time shape: ", X_over_time.shape)

    ax.plot(np.arange(len(data.edge_index)), X_over_time, linestyle = '--', marker = 'X')
    # ax.plot(np.arange(len(data.edge_index)), X_l_over_time, linestyle = '-', linewidth = 1.0)
    ax.set_xlabel(r"Time step ($t$)")
    ax.set_ylabel(r"$X(t)$")
    ax.grid(True)


    if channel_features == ('FR', 'rates'):
        ax.set_title(f"Delta_t: {DELTA_T} - FR rates over time")
        plt.savefig(f'./example_fr_throughputs_over_time_delta_t_{DELTA_T}.png', dpi = 300)
        plt.close(fig)

    elif any([channel_features[0] == algo and channel_features[1] == 'interference_powers' for algo in eval_baselines]):
        for algo in eval_baselines:
            if channel_features[0] == algo and channel_features[1] == 'interference_powers':
                ax.set_title(f"Delta_t: {DELTA_T} - {algo.upper()} Interference powers over time")
                plt.savefig(f'./example_{algo}_interference_powers_over_time_delta_t_{DELTA_T}.png', dpi = 300)
                plt.close(fig)
                
                break

            else:
                continue


    elif channel_features == ('ITLinQ', 'Ps'):
        ax.set_title(f"Delta_t: {DELTA_T} - ITLinQ Ps over time")
        plt.savefig(f'./example_itlinq_ps_over_time_delta_t_{DELTA_T}.png', dpi = 300)
        plt.close(fig)

    elif channel_features == ('WMMSE', 'Ps'):
        ax.set_title(f"Delta_t: {DELTA_T} - WMMSE Ps over time")
        plt.savefig(f'./example_wmmse_ps_over_time_delta_t_{DELTA_T}.png', dpi = 300)
        plt.close(fig)

    else:
        raise ValueError(f"Unknown features: {channel_features}. Expected ('FR', 'rates') or ('FR', 'interference_powers').")


    # Plot incoming interference over time
    fig, ax = plt.subplots(1, 1, figsize = (24, 8))
    y_over_time = np.stack([data.y[t][14:16].cpu().numpy().reshape(-1) for t in range(len(data.y))], axis = 0)
    accelerator.print("y_over_time shape: ", y_over_time.shape)

    ax.plot(np.arange(len(data.edge_index)), y_over_time, linestyle = '--', marker = 'd')
    ax.set_xlabel(r"Time step ($t$)")
    ax.set_ylabel(r"$y(t)$")
    ax.grid(True)


    if channel_features == ('FR', 'rates'):
        ax.set_title(f"Delta_t: {DELTA_T} - FR rates over time")
        plt.savefig(f'./example_fr_throughputs_over_time_delta_t_{DELTA_T}.png', dpi = 300)
        plt.close(fig)

    elif any([channel_features[0] == algo and channel_features[1] == 'interference_powers' for algo in eval_baselines]):
        for algo in eval_baselines:
            if channel_features[0] == algo and channel_features[1] == 'interference_powers':
                ax.set_title(f"Delta_t: {DELTA_T} - {algo.upper()} interference powers over time")
                plt.savefig(f'./example_{algo}_interference_powers_over_time_delta_t_{DELTA_T}.png', dpi = 300)
                plt.close(fig)
                break

    elif channel_features == ('ITLinQ', 'Ps'):
        ax.set_title(f"Delta_t: {DELTA_T} - ITLinQ rates over time")
        plt.savefig(f'./example_itlinq_rates_over_time_delta_t_{DELTA_T}.png', dpi = 300)
        plt.close(fig)

    elif channel_features == ('WMMSE', 'Ps'):
        ax.set_title(f"Delta_t: {DELTA_T} - WMMSE rates over time")
        plt.savefig(f'./example_wmmse_rates_over_time_delta_t_{DELTA_T}.png', dpi = 300)
        plt.close(fig)

    else:
        raise ValueError(f"Unknown features: {channel_features}. Expected ('FR', 'rates') or ('FR', 'interference_powers').")



    #### Add regression features to the dataset here from FR baseline. ####


    ############################### Create channel dataset loggers ###############################
    print("Creating channel dataset loggers...")
    # if loggers is not None:
    #     if accelerator is None or accelerator.is_local_main_process:
    #         for logger in loggers:
    #             if logger.log_metric in ['rates']:
    #                 if logger.network_id is not None:
    #                     avg_rates = [baseline_metrics[alg, 'rates'][logger.network_id].mean(dim = 0, keepdim = True) for alg in eval_baselines]
    #                     logger.barplot_opt_problem(avg_rates = avg_rates, metric_names = eval_baselines) if logger.network_id < MAX_LOGGED_NETWORKS else None
    #                 else:
    #                     avg_rates = [torch.stack(baseline_metrics[alg, 'rates'], dim = 0).mean(dim = 1).view(1, -1) for alg in eval_baselines]
    #                     logger.barplot_opt_problem(avg_rates = avg_rates, metric_names = eval_baselines)

    #                 # for alg in eval_baselines:
    #                 #     avg_rates = baseline_metrics[alg, 'rates'][logger.network_id].mean(dim = 0, keepdim = True)
    #                 #     logger.barplot_opt_problem(avg_rates=avg_rates, metric_names = [alg])

                
    #             elif logger.log_metric in ['tx-rx-locs']:
    #                 logger.update_data({'tx-rx-locs': [{"tx": loc['tx'], "rx": loc['rx'],
    #                                                     "associations": assoc, "H_l": h_l,
    #                                                     "P_max": P_max, "noise_var": noise_var, "r_min": r_min,
    #                                                     } for loc, assoc, h_l in zip(Locs, Assocs, H_l)]})
    #                 logger()
                
    #             else:
    #                 logger.update_data(data_list)
    #                 logger()



    return data_list, baseline_metrics, avg_graph_data

    