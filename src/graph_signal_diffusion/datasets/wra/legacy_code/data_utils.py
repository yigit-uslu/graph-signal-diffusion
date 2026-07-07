from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset, TensorDataset, WeightedRandomSampler
import torch, random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import abc
from torch_scatter import scatter
import os, pickle


def permute_pygraph(data):
    # perm = torch.randperm(data.x.size(0))  # Random permutation of node indices

    perm = torch.randperm(data.y_l.size(0)) # Random permutation of node indices
    
    # Permute long-term node features
    y_l_perm = torch.zeros_like(data.y_l)
    y_l_perm[perm] = data.y_l
    data.y_l = y_l_perm

    # Permute short-term node features
    for i in range(len(data.y)):
        yy_perm = torch.zeros_like(data.y[i])
        yy_perm[perm] = data.y[i]
        data.y[i] = yy_perm

    
    # Permute edge indices (the edges are pairs of node indices, so permute each element of the pairs)
    data.edge_index_l = perm[data.edge_index_l]

    for i in range(len(data.edge_index)):
        data.edge_index[i] = perm[data.edge_index[i]]
    
    return data, perm



def batch_from_data_list(data_list, batch_size=None, exclude_keys=None):
    
    if exclude_keys is None:
        exclude_keys = []

    # Create a batch from the data list
    batch = Batch.from_data_list(data_list, exclude_keys=exclude_keys)

    # If batch_size is provided, set the batch size
    if batch_size is not None:
        batch.batch_size = batch_size

    return batch



def create_data_list_from_batch(data, keys_to_iterate=None, keys_to_copy=None, batch_size = 1, batch_dim = 0,
                                batch_size_overwrite_num_iterable_keys = False):
    """
    Create a list of Data objects from a single Data object where:
    - For keys in keys_to_iterate: iterate over the first dimension
    - For keys in keys_to_copy: copy the same value to all created Data objects
    
    Args:
        data: PyG Data object
        keys_to_iterate: List of keys to iterate over first dimension
        keys_to_copy: List of keys to simply copy to all data objects
        batch_size: Batch size for the returned data_list
        batch_dim: (Iteration) dimension to use for iterable keys (default is 0)
        batch_size_overwrite_num_iterable_keys: If True, override the number of items 
            in the returned data_list to be batch_size (if batch_size is smaller than 
            the number of items to iterate over) (default is False)
    
    Returns:
        List of Data objects
    """

    if keys_to_iterate is None:
        keys_to_iterate = []
    if keys_to_copy is None:
        keys_to_copy = []
    if keys_to_copy == 'all':
        try:
            keys_to_copy = [key for key in data.keys() if key not in keys_to_iterate]
        except:
            keys_to_copy = [key for key in data.keys if key not in keys_to_iterate]
    
    # Determine the number of data objects to create based on first dimension of keys_to_iterate
    num_items = None
    for key in keys_to_iterate:
        if hasattr(data, key):
            tensor = getattr(data, key)
            if num_items is None:
                num_items = tensor.shape[batch_dim]
            else:
                assert tensor.shape[batch_dim] == num_items, f"Inconsistent first dimension for {key}: {tensor.shape[batch_dim]} != {num_items}"
    

    if num_items is None:
        print(f"No valid Keys_to_iterate were found. Returning a list with {batch_size} Data objects.")
        num_items = batch_size
        # raise ValueError("No valid keys_to_iterate were found")

    else:
        if batch_size_overwrite_num_iterable_keys:
            if batch_size <= num_items:
                print(f"{num_items} valid Keys_to_iterate were found. Still returning a list with {batch_size} Data objects.")
                num_items = batch_size
            else:
                print(f"{num_items} valid Keys_to_iterate were found but asked batch_size = {batch_size} is larger. Returning a list with all the unique {num_items} Data objects.")
                
    
    # Create list of Data objects
    data_list = []
    for i in range(num_items):
        # new_data = Data()
        # new_data = Data_modTxIndex()

        # new_data = type(data)(**data.__dict__)  # Copy the data object structure
        # Initialize new data object with the same attributes as data
        new_data = data.clone()  # Clone the data object to avoid modifying the original
        # print("new_data is of type: ", type(new_data), "\tof class: ", new_data.__class__.__name__)
        # print("New data.batch: ", new_data.batch)
        
        # Iterate over specified keys
        for key in keys_to_iterate:
            if hasattr(data, key):
                tensor = getattr(data, key)
                # setattr(new_data, key, tensor[i])

                print(f"key = {key}, i = {i}, tensor.shape = {tensor.shape}, batch_dim = {batch_dim}")
                setattr(new_data, key, 
                        torch.index_select(tensor, batch_dim, torch.tensor([i], device=tensor.device)) #.squeeze(batch_dim)
                        )

                # Get a slice of a tensor in batch_dim dimension and at index i
                # new_data[key] = tensor.index_select(batch_dim, torch.tensor([i], device=tensor.device))
        
        # Copy remaining keys
        for key in keys_to_copy:
            if hasattr(data, key):
                setattr(new_data, key, getattr(data, key))

        # # Setting number of graphs manually
        # if hasattr(data, "batch"):
        #     print(f"Setting new_data.num_graphs from {new_data.
        # } to {int(data.batch.max()) + 1}")
        #     new_data.num_graphs = int(data.batch.max()) + 1

        # print(f'i = {i}\tnew_data: ', new_data)
        
        data_list.append(new_data)
    
    return data_list


class Data_modTxIndex(Data):
    def __init__(self,
                 pos = None,
                 network_id = None,
                 x_l = None, # long-term dual-regression features
                 x = None, # instantaneous dual-regression features not used in the paper
                 y=None,
                 y_l=None, # dual-regression outputs, i.e., initial dual multipliers
                 y_target = None, # used for dual regression only
                 edge_index_l=None,
                 edge_weight_l=None,
                 edge_index=None,
                 edge_weight=None,
                 avg_graph_edge_index_l=None,
                 avg_graph_edge_weight_l=None,
                 weighted_adjacency=None,
                 weighted_adjacency_l=None,
                 transmitters_index=None,
                 init_long_term_avg_rates=None,
                 num_nodes=None,
                 m=None):
        super().__init__()
        self.pos = pos,
        self.network_id = network_id
        self.x_l = x_l
        self.x = x
        self.y = y
        self.y_l = y_l
        self.y_target = y_target
        self.edge_index_l = edge_index_l
        self.edge_weight_l = edge_weight_l
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.weighted_adjacency = weighted_adjacency
        self.weighted_adjacency_l = weighted_adjacency_l
        self.transmitters_index = transmitters_index
        self.init_long_term_avg_rates = init_long_term_avg_rates
        self.num_nodes = num_nodes
        self.m = m

                
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'transmitters_index':
            if isinstance(self.m, torch.Tensor):
                # print(f"Incrementing transmitters_index = [{value.shape}] by self.m.sum(): ", self.m.sum().item())
                return self.m.sum()
            else:
                # print(f"Incrementing transmitters_index = [{value.shape}] by m: ", self.m)
                return self.m
            
        elif key in ['y_index']:
            return 0 # we do not increment y_index.
            
        elif key in ['x_index', 'xgen_index']:
            return 0
            # if isinstance(self.x, list):
            #     return len(self.network_id) * len(self.x) # increment by the number of total samples in the dataset.
            # else:
            #     return len(self.x)

        else:
            return super().__inc__(key, value, *args, **kwargs)
        

    def __cat_dim__(self, key, value, *args, **kwargs):
        # Default behavior for specific keys
        if key in ['x', 'y', 'xgen']:
            return 1  # concatenate along the node dimension
        
        # Fall back to parent class behavior
        return super().__cat_dim__(key, value, *args, **kwargs)
        

    # def __getitem__(self, key):
    #     if key == 'x':
    #         x_temp = super().__getitem__(key)
    #         return transform(x_temp)

    #     else:
    #         return super().__getitem__(key)

# def transform(x):
#     return x / 10000



class Data_Lambdas(Data):
    def __init__(self, y = None, y_l = None):
        super().__init__()
        self.y = y
        self.y_l = y_l


    def __inc__(self, key, value, *args, **kwargs):
        if key == 'transmitters_index':
            return self.m
        else:
            return super().__inc__(key, value, *args, **kwargs)
        

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'y':
            return 1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)
        

    def __getitem__(self, key):
        if key == 'x' and self.transform_x is not None:
            x_temp = super().__getitem__(key)
            return self.transform_x(x_temp)

        else:
            return super().__getitem__(key)
        


class Data_PowerAllocPolicy(Data):
    def __init__(self,
                 x = None,
                 channel_id = None,
                 transmitters_index=None,
                 num_nodes=None,
                 m=None,
                 transform_x = None,
                 weights = None):
        super().__init__()
        self.x = x
        self.transform_x = transform_x
        self.channel_id = channel_id
        self.transmitters_index = transmitters_index
        self.num_nodes = num_nodes
        self.m = m
        self.weights = weights
                

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'transmitters_index':
            return self.m
        else:
            return super().__inc__(key, value, *args, **kwargs)
        

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'x':
            return 1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)
        

    def __getitem__(self, key):
        if key == 'x' and self.transform_x is not None:
            x_temp = super().__getitem__(key)
            return self.transform_x(x_temp)

        else:
            return super().__getitem__(key)
        

    # def __getitem__(self, idx = None):
    #     if idx is None:
    #         return self.x
        
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()
    #     x_batch = self.x[idx]

    #     return x_batch



class LambdasBuffer(abc.ABC):
    def __init__(self, buffer_size = 1000, lambdas_shape = (1,)):
        self.buffer_size = buffer_size
        self.lambdas_shape = lambdas_shape
        self.lambdas = torch.zeros(buffer_size, *self.lambdas_shape)
        self.pos = 0
        self.buffer_adds = 0
        self.full = False


    def reset(self):
        self.lambdas = torch.zeros(self.buffer_size, *self.lambdas_shape)
        self.pos = 0
        self.full = False


    def add(self, lambdas):

        if isinstance(lambdas, list):
            for lam in lambdas:
                self.add(lam)
            return
        
        if isinstance(lambdas, np.ndarray):
            lambdas = torch.from_numpy(lambdas)
            self.add(lambdas)
            return

        if isinstance(lambdas, torch.Tensor) and lambdas.shape != self.lambdas_shape:
            # Reshape if neeeded
            lambdas = lambdas.view(-1, *self.lambdas_shape)
            self.add([lam for lam in lambdas])
            return

        # Copy to avoid modification by reference
        self.lambdas[self.pos] = lambdas.detach().clone()

        self.pos += 1
        self.buffer_adds += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    
    def get_batch_indices(self, batch_size, **kwargs):
        if not isinstance(batch_size, tuple):
            batch_size = (batch_size,)

        if self.full:
            batch_inds = (torch.randint(low = 1, high = self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = torch.randint(low = 0, high = self.pos, size=batch_size)

        return batch_inds
            

    def sample(self, batch_size, return_indices = False, **kwargs):
        batch_inds = self.get_batch_indices(batch_size, **kwargs)
        print(f"batch_inds: {batch_inds.shape}")
        print(f"lambdas: {self.lambdas.shape}")
        print(f"lambdas[batch_inds]: {self.lambdas[batch_inds].shape}")

        if return_indices:
            return self.lambdas[batch_inds], batch_inds
        else:
            return self.lambdas[batch_inds]
        


class LambdasBufferWithImportanceWeights(LambdasBuffer):
    def __init__(self, buffer_size = 1000, lambdas_shape = (1,), importance_weights_func = None):
        super(LambdasBufferWithImportanceWeights, self).__init__(buffer_size=buffer_size, lambdas_shape=lambdas_shape)
        self.importance_weights = torch.zeros(buffer_size)
        self.lagrangians = torch.zeros(buffer_size)
        self.lagrangians_history = []
        self.infos_history = []
        self.set_weights_func(importance_weights_func)

    def add(self, lambdas, data, *args, **kwargs):
        print("First adding lambdas using Parent Buffer's add method.")
        # super().add(self, lambdas)

        if isinstance(lambdas, list):
            for lam in lambdas:
                super().add(lam)
            # return
        
        if isinstance(lambdas, np.ndarray):
            lambdas = torch.from_numpy(lambdas)
            super().add(lambdas)
            # return

        if isinstance(lambdas, torch.Tensor) and lambdas.shape != self.lambdas_shape:
            # Reshape if neeeded
            lambdas = lambdas.view(-1, *self.lambdas_shape)
            for lam in lambdas:
                super().add(lam)
            # super().add(self, [lam for lam in lambdas])
            # return

        if data is not None:
            self.set_weights(data, *args, **kwargs)
        else:
            self.importance_weights[:self.pos] = 1


    def set_weights_func(self, func):
        def thunk(lambdas, data, *args, **kwargs):
            weights, lagrangians, infos = func(lambdas, data, *args, **kwargs)

            if not self.full:
                weights[self.pos:] = 0.

            return weights, lagrangians, infos

        self.importance_weights_func = thunk
        # self.importance_weights_func = importance_weights_func

    def set_weights(self, data, *args, **kwargs):
        if self.importance_weights_func is not None:
            # try:
            weights, lagrangians, infos = self.importance_weights_func(self.lambdas, data, *args, **kwargs)
            print("weights: ", weights)
            print("lagrangians: ", lagrangians)
            assert torch.all(weights >= 0), "Importance weights must be non-negative."
            assert torch.all(torch.isfinite(weights)), "Importance weights must be finite."
            assert torch.sum(weights) > 0, "Importance weights must sum to a positive value."

            self.importance_weights = weights
            self.record_lagrangian(lagrangians)
            self.record_infos(infos)

            # except:
            #     print("Error in setting importance weights. Skipping weight update.")
            #     pass


    def record_lagrangian(self, lagrangians):
        self.lagrangians = lagrangians
        # self.lagrangians_history.append(lagrangians.mean().item())

    def record_infos(self, infos):
        for name, value in infos.items():
            setattr(self, name, value)


    def get_batch_indices(self, batch_size, **kwargs):
        if isinstance(batch_size, tuple):
            batch_size = int(batch_size[0])

        # Draw samples from the buffer with probability proportional to the importance weights.
        probs = self.importance_weights / self.importance_weights.sum()

        if self.importance_weights.sum() == 0:
            print("Importance weights sum to zero. Sampling uniformly.")
            probs = torch.ones_like(probs) / len(probs)

        batch_inds = torch.multinomial(probs, batch_size, replacement=True)

        # if self.full:
        #     batch_inds = (torch.randint(low = 1, high = self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        # else:
        #     batch_inds = torch.randint(low = 0, high = self.pos, size=batch_size)
        return batch_inds
    

    def sample(self, batch_size, return_indices=False, return_lagrangians = False, **kwargs):
        return_indices = True if return_lagrangians is True else return_indices
        temp = super().sample(batch_size, return_indices, **kwargs)
        
        if return_lagrangians:
            return *temp, self.lagrangians[temp[1]]
        else:
            return temp
            
        
        
# Old WirelessDataset class
class BaseWirelessDataset(Dataset):
    def __init__(self, data_list, **kwargs):
        super().__init__(**kwargs)
        self.data_list = data_list
        # self.__dict__.update(kwargs)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx], idx
    


class WirelessDataset(BaseWirelessDataset):
    def __init__(self, data_list, **kwargs):
        super().__init__(data_list, **kwargs)
        # self.original_dataset = original_dataset
        
    def len(self):
        return super().len()
        
    def get(self, idx):
        data, graph_idx = super().get(idx)
        
        # # Add time sample here (during dataset access, not batching)
        # if hasattr(data, 'edge_index') and isinstance(data.edge_index, list) and len(data.edge_index) > 0:
        #     t = torch.randint(0, len(data.edge_index), (1,)).item()
        #     data.edge_index_t = data.edge_index[t]
        #     data.edge_weight_t = data.edge_weight[t] if hasattr(data, 'edge_weight') else None
        #     data.t = t
            
        return data, graph_idx
    


class LambdasDataset(BaseWirelessDataset):
    def __init__(self, data_list, lambdas_sampler, n_lambdas, **kwargs):
        super().__init__(data_list, **kwargs)
        self.lambdas_sampler = lambdas_sampler
        # self.lambda_transform = lambda x, direction: x / self.lambdas_sampler.lambdas_max - 0.5 if direction == 'forward' else torch.clamp_min((x + 0.5) * self.lambdas_sampler.lambdas_max, min = 0)
        self.n_lambdas = n_lambdas
        self.set_operation_mode(kwargs.get('operation_mode', 'primal'))


    def get(self, idx):
        print("Getting data from LambdasDataset at index:", idx)
        data, graph_idx = super().get(idx)
  
        # Add randomly sampled lambdas here and standardize them to the diffusion space
        if self.lambdas_sampler is not None:

            # print("Batch size of lambdas: ", data.num_graphs)
            print(f"Calling lambdas_sampler to sample {self.n_lambdas} lambdas.")
            lambdas = self.lambdas_sampler.sample(n_samples=self.n_lambdas, data = data, sampler_probs = self.sampler_probs) # same mechanism for all graphs
            
            if isinstance(lambdas, dict):
                data.y = lambdas["lambdas"].clone()
                data.y_l = lambdas["lambdas_star"].clone()

            else:
            
                print(f"Called {self.lambdas_sampler.__class__.__name__} ended up sampling {lambdas.shape[0]} lambdas with shape {lambdas.shape}.")
                assert lambdas.shape[0] == self.n_lambdas, f"Expected {self.n_lambdas} lambdas, but got {lambdas.shape[0]}."
                # data.lambdas = lambdas
                # data.y = self.lambda_transform(lambdas, direction = 'forward')
                data.y = lambdas.clone()

                # print("data.num_graphs: ", data.num_graphs)
                print("data.y.shape: ", data.y.shape)
                print("data.network_id: ", data.network_id)
                # print(f"lambdas: {lambdas}")

                # else:
                #     print("Data.y is not None. Using it as lambdas.")

        return data, graph_idx
    

    def set_operation_mode(self, operation_mode):
        if operation_mode not in ['primal', 'dual']:
            raise ValueError(f"Unknown operation mode: {operation_mode}")
        self.operation_mode = operation_mode
        print(f"Operation mode set to {self.operation_mode}.")
    

    @property
    def sampler_probs(self):
        if not hasattr(self.lambdas_sampler, 'probs'):
            return None
        

        probs = self.lambdas_sampler.probs
        if self.operation_mode == 'primal':
            # In primal step, use probabilities from all samplers
            return probs


        elif self.operation_mode == 'dual':
            # In dual step, mask out the probabilities of the dual sampler
            # masked_probs = [0.0 if type(sampler) == DiffusionStateAugmentedLambdaSampler else prob 
            #                 for prob, sampler in zip(probs, self.lambdas_sampler.lambda_samplers_list)]
            
            
            masked_probs = probs.clone()
            masked_probs[-1] = 0.0
            
            masked_probs = masked_probs / masked_probs.sum()
            print(f"In dual step, masking out the probabilities of the dual sampler from {probs} to mask {masked_probs}.")

            return masked_probs


        else:
            raise ValueError(f"Unknown operation mode: {self.operation_mode}")
            

    


class standardize_data(abc.ABC):
    ''' 
    A standardization function wrapped in a class object.
    '''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, invert = False):
        ''' 
        If invert = FALSE, the forward transform standardizes the input data x to have zero mean and unit variance.
        If invert = TRUE, the forward transform is reversed.
        '''

        if invert:
            return x * self.std + self.mean
        else:
            return (x - self.mean) / self.std



class ZippedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        # Return the length of the shorter dataset
        return min([len(dataset) for dataset in self.datasets])

    def __getitem__(self, idx):
        items = [dataset[idx] for dataset in self.datasets]
        return (*items,)


def importance_sampler(X_0, Y_0 = None, weights = None, batch_size = 1000, replacement = True):
    X_dataset = TensorDataset(X_0)
    if Y_0 is not None:
        Y_dataset = TensorDataset(Y_0)
        dataset = ZippedDataset([X_dataset, Y_dataset])
    else:
        dataset = X_dataset

    weights = torch.ones_like(len(X_dataset)) if weights is None else weights
    sampler = WeightedRandomSampler(weights, len(weights), replacement = replacement)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             sampler=sampler,
                                             batch_size = batch_size,
                                             drop_last = False
                                             )
    return dataloader

                
    

def create_channel_dataloader(data_list, batch_size, shuffle = False):
    dataloader = DataLoader(WirelessDataset(data_list), batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

def get_T_range(t_range_list, T_repeats = 1, shuffle = True):
    T_range = []
    for _ in range(T_repeats): # to avoid having to save more channel state matrices to the disk and sample from the stationary distribution of dual dynamics, resample shuffled channel states.
        if shuffle:
            random.shuffle(t_range_list)
        T_range.extend(t_range_list)

    return T_range





def collate_with_time_sample(data_list):
    """
    Custom collate function that adds edge_index_t and edge_weight_t attributes
    by randomly sampling one timestep per batch.
    """
    print("Collating data with time sampling...")
    # Process each data object to add the time-sampled attributes
    for data in data_list:
        # Check if we have temporal edge data
        if hasattr(data, 'edge_index') and isinstance(data.edge_index, list) and len(data.edge_index) > 0:
            # Randomly sample a timestep
            print("Randomly sampling a timestep...")
            t = torch.randint(0, len(data.edge_index), (1,)).item()
            
            # Create new attributes with the sampled timestep data
            data.edge_index_t = data.edge_index[t]
            data.edge_weight_t = data.edge_weight[t] if hasattr(data, 'edge_weight') else None
            
            # Optionally add the sampled timestep for reference
            data.t = t

    # Use PyG's default batching after adding our custom attributes
    return Batch.from_data_list(data_list)



def create_power_alloc_policy_dataset(accelerator,
                                      args, channel_dataloader, expert_policy, sa_model, device, policy_name = 'state-augmented', loggers = None,
                                      tau = 1.0,
                                      lr_resilient = 0.0,
                                      num_channel_rollouts = 20,
                                      num_expert_dataset_samples = 1000
                                      ):

    
    sa_model = sa_model.to(device)

    print(f'Creating a dataset of {policy_name} expert-policy allocations.')
    p_dataset = defaultdict(list)
    channel_p_joint_dataset = defaultdict(list)
    diffusion_dataloader = defaultdict(list)

    p_mean = 0
    p_std = 0

    #### Retrive optimal state-augmented power allocations. ####
    T_0 = 5
    if hasattr(expert_policy.config, "T_0") and expert_policy.config.T_0 is not None:
        T_0 = expert_policy.config.T_0

    T_repeats = num_channel_rollouts # rollout the channel for longer to sample from the stationary distribution.
    for phase in channel_dataloader:
        
        ### state-augmented policy ###
        if policy_name == 'state-augmented':
            sa_model.eval()
            Ps_all = []
            rates_all = []
            lambdas_all = []
            lagrangians_all = []
            sample_idx_all = []
            lambdas_init = torch.zeros((len(channel_dataloader[phase].dataset), args.n)).to(dtype = torch.float32, device=device)
            for data, sample_idx in tqdm(channel_dataloader[phase]):

                with torch.no_grad():
                    sample_idx = sample_idx.to(device)
                    # T_range = []
                    t_range = list(range(len(data.edge_index)))
                    # for _ in range(T_repeats): # to avoid having to save more channel state matrices to the disk and sample from the stationary distribution of dual dynamics, resample shuffled channel states.
                    #     random.shuffle(t_range)
                    #     T_range.extend(t_range)
                    T_range = get_T_range(t_range_list=t_range, T_repeats=T_repeats, shuffle=False)
                    eval_metrics = expert_policy.eval(model=sa_model,
                                                data=data.to(device),
                                                lambdas=lambdas_init[sample_idx],
                                                lr_dual=expert_policy.config.lr_dual,
                                                T_0 = T_0,
                                                lr_resilient=lr_resilient, # 0.05
                                                T_range = T_range,
                                                lambda_thresh_resilient = expert_policy.lambda_sampler.lambdas_max / 2,
                                                lambda_clamp_max = 2. * expert_policy.lambda_sampler.lambdas_max
                                                )
                    
                    eval_Ps = eval_metrics['Ps'] # [T * t_repeats, B*n, 1]
                    Ps = eval_Ps.view(eval_Ps.shape[0], -1, args.n)
                    # Ps = Ps[-(Ps.shape[0] // 2):] # don't sample from the transient phase
                    Ps = Ps[-num_expert_dataset_samples:] # don't sample from the transient phase
                    Ps = torch.moveaxis(Ps, source=0, destination=-2).unsqueeze(-1) # [B, TT, n, 1]

                    eval_rates = eval_metrics['rates']
                    rates = eval_rates.view(eval_rates.shape[0], -1, args.n)
                    # rates = rates[-(rates.shape[0] // 2):] # don't sample from the transient phase
                    rates = rates[-num_expert_dataset_samples:] # don't sample from the transient phase
                    rates = torch.moveaxis(rates, source=0, destination=-2).unsqueeze(-1) # [B, TT, n, 1]
                    
                    eval_lambdas = eval_metrics['lambdas'] # [K * t_repeats, B, n]
                    lambdas = eval_lambdas.view(eval_lambdas.shape[0], -1, args.n)

                    # lambdas = lambdas[-(lambdas.shape[0] // 2):] # don't sample from the transient phase
                    lambdas = lambdas[-(num_expert_dataset_samples // T_0):] # don't sample from the transient phase
                    lambdas = torch.moveaxis(lambdas, source = 0, destination=-2).unsqueeze(-1) # [B, KK, n, 1]

                    dual_weighted_slacks = expert_policy.constraints(eval_rates) * eval_lambdas.repeat(T_0, *([1] * len(eval_lambdas.shape[1:])))[:-T_0]
                    lagrangian_over_time = (expert_policy.obj(eval_rates) + torch.sum(dual_weighted_slacks, dim = -1)) / args.n
                    lagrangian_over_iterations = lagrangian_over_time.reshape(T_0, -1,*(lagrangian_over_time.shape[1:])).mean(dim = 0)
                    lagrangian_over_iterations = torch.moveaxis(lagrangian_over_iterations, source=0, destination=-1).unsqueeze(-1) # [B, KK, 1]

                    Ps_all.append(Ps.detach().cpu())
                    rates_all.append(rates.detach().cpu())
                    lambdas_all.append(lambdas.detach().cpu())
                    lagrangians_all.append(lagrangian_over_iterations.detach().cpu())
                    sample_idx_all.append(sample_idx.detach().cpu())


            sample_idx_all = torch.cat(sample_idx_all, dim = 0)
            Ps_all = torch.cat(Ps_all, dim = 0) / sa_model.P_max # normalized to [0, 1] range
            lambdas_all = torch.cat(lambdas_all, dim = 0)
            rates_all = torch.cat(rates_all, dim = 0)
            lagrangians_all = torch.cat(lagrangians_all, dim = 0)

            print(f"sa_model.P_max: {sa_model.P_max}\tchannel_config.P_max={expert_policy.channel_config.P_max}")

            # print(f"Ps_all.min() / P_max before scattter: ", Ps_all.min().item())
            # print(f"Ps_all.shape before scattter: ", Ps_all.shape)

            Ps_all = scatter(src=Ps_all, index=sample_idx_all, dim=0, reduce="mean")

            try:
                print(f"Ps_all.min() / P_max after scattter: ", Ps_all.quantile(q=0.01).item())
                print(f"Ps_all.max() / P_max after scattter: ", Ps_all.quantile(q=0.99).item())
                print(f"Ps_all.median() / P_max after scattter: ", Ps_all.quantile(q=0.50).item())
            except:
                print("Printing quantiles failed possibly because input tensor was too large.")
            
            print(f"Ps_all.shape after scattter: ", Ps_all.shape)

            rates_all = scatter(src=rates_all, index=sample_idx_all, dim=0, reduce="mean")
            lambdas_all = scatter(src=lambdas_all, index=sample_idx_all, dim=0, reduce="mean")
            lagrangians_all = scatter(src=lagrangians_all, index=sample_idx_all, dim=0, reduce="mean")

            ergodic_rates_all = torch.mean(rates_all, dim = 1)
            print(f"1-percentile-ergodic-rates (avg): ", ergodic_rates_all.squeeze(-1).quantile(q=0.01, dim = -1).mean().item())
            print(f"1-percentile-ergodic-rates (best): ", ergodic_rates_all.squeeze(-1).quantile(q=0.01, dim = -1).max().item())
            print(f"1-percentile-ergodic-rates (worst): ", ergodic_rates_all.squeeze(-1).quantile(q=0.01, dim = -1).min().item())
            print(f"10-percentile-ergodic-rates (avg): ", ergodic_rates_all.squeeze(-1).quantile(q=0.1, dim = -1).mean().item())
            print(f"10-percentile-ergodic-rates (best): ", ergodic_rates_all.squeeze(-1).quantile(q=0.1, dim = -1).max().item())
            print(f"10-percentile-ergodic-rates (worst): ", ergodic_rates_all.squeeze(-1).quantile(q=0.1, dim = -1).min().item())



            ### Save ergodic rates to a file. ###
            save_dir = f"./expert-ergodic-rates-{phase}"
            save_dir = None 
            if loggers['train'] is not None:
                for logger in loggers['train']:
                    if logger.log_metric == 'rates':
                        save_dir = f"{logger.log_path}/ergodic-rates/{phase}/"

            save_dir = f"./expert-ergodic-rates-{phase}" if save_dir is None else save_dir

            os.makedirs(save_dir, exist_ok = True)
            with open(save_dir + 'sa-ergodic-rates.pickle', 'wb') as f:
                pickle.dump(ergodic_rates_all.detach().cpu().numpy(), f, pickle.HIGHEST_PROTOCOL)

            with open(save_dir + 'sa-rates.pickle', 'wb') as f:
                pickle.dump(rates_all.detach().cpu().numpy(), f, pickle.HIGHEST_PROTOCOL)


            log_variables = {'Ps': torch.moveaxis(Ps_all.squeeze(-1), source = 0, destination=1),
                             'rates': torch.moveaxis(rates_all.squeeze(-1), source=0, destination=1),
                             'lambdas': torch.moveaxis(lambdas_all.squeeze(-1), source = 0, destination=1),
                             'lagrangian-over-iterations': torch.moveaxis(lagrangians_all.squeeze(-1), source = 0, destination = 1),
                             'opt-problem': [(ergodic_rates_all[i].squeeze(-1),) for i in range(rates_all.shape[0])], 
                            }

            # expert_policy.log(epoch = -1, loggers = [logger for logger in loggers['test'] if logger.log_metric in log_variables.keys()], all_variables = log_variables, model = sa_model, n = args.n)
            if accelerator is None or accelerator.is_local_main_process and loggers[phase] is not None: # log once
                try:
                    expert_policy.log(epoch = -1, loggers = [logger for logger in loggers[phase] if logger.log_metric in log_variables.keys()], all_variables = log_variables, model = sa_model, n = args.n)
                except:
                    expert_policy.log(phase = phase, epoch = -1, loggers = [logger for logger in loggers[phase] if logger.log_metric in log_variables.keys()], all_variables = log_variables, model = sa_model, n = args.n)


            if tau is None:
                sampler_weights = None
            else:
                ### Sample Ps proportional to the Lagrangian values. ###
                # tau = 1.0
                # tau = args.tau
                n_samples = Ps_all.shape[1]

                # print('n_samples: ', n_samples)
                print("lagrangian_over_iterations: ", lagrangians_all.shape)

                exp_lagrangians = torch.exp(tau * (lagrangians_all - lagrangians_all.max(dim = 1, keepdim = True).values))

                print('exp_lagrangians.shape: ', exp_lagrangians.shape) 
                sampler_weights = torch.repeat_interleave(exp_lagrangians, repeats=n_samples // lagrangians_all.shape[1], dim = 1).squeeze(-1)
                sampler_weights = torch.nn.functional.normalize(sampler_weights, p=1, dim=1) * sampler_weights.shape[1]
                # sampler_weights = exp_lagrangians.squeeze(-1)
                print('sampler_weights.shape: ', sampler_weights.shape)

                print("Lagrangians per graph before I.S.:\n")
                for i in range(len(Ps_all)):
                    weighted_lagrangian = torch.mean(sampler_weights[i] * torch.repeat_interleave(lagrangians_all[i].squeeze(-1), repeats=n_samples // lagrangians_all[i].shape[0], dim = 0))
                    print(f"Graph {i}\tLagrangian (before I.S.): {lagrangians_all[i].mean().item()}\tLagrangian (with I.S.): {weighted_lagrangian.mean().item()}")

                
                ### Perform importance sampling per mini-batch of graph signals for each graph ###
                n_samples = n_samples // T_repeats
                temp_Ps_all = []
                for i in range(len(Ps_all)):
                    Ps_temp = []
                    for j in range(T_repeats):
                        b_idx = torch.arange(j*n_samples, (j+1)*n_samples)
                        x_0 = Ps_all[i][b_idx]
                        weights = sampler_weights[i][b_idx]
                
                        dataloader = importance_sampler(X_0=x_0, # Ps_orig
                                                        Y_0=None,
                                                        weights=weights, # importance sampling weights
                                                        batch_size=n_samples,
                                                        replacement=True
                                                        )
                    
                        x = next(iter(dataloader))[0]
                        Ps_temp.append(x)
                    Ps_temp = torch.cat(Ps_temp, dim = 0)
                    temp_Ps_all.append(Ps_temp)

                temp_Ps_all = torch.stack(temp_Ps_all, dim = 0)
                print('temp_Ps_all.shape: ', temp_Ps_all.shape)

                Ps_all = temp_Ps_all
                print(f"Updated dataset of state-augmented policy samples with KL-regularized Importance-Sampler with tau = {tau}.")



            # sampler_weights = torch.ones_like(Ps_all)

            ### Sample Ps proportional to the Lagrangian values. ###


        ### uniform random policy ###
        elif policy_name == 'uniform-random':
            num_graphs = len(channel_dataloader[phase])
            T = T_repeats * len(channel_dataloader[phase].dataset[0][0].edge_index)
            n = args.n 
            Ps_all = torch.rand((num_graphs, T, n, 1), device = device) # U[0, 1]


        ### ITLinQ or WMMSE policy ###
        elif policy_name in ['ITLinQ', 'WMMSE']:
            Ps_all = []
            rates_all = []
            sample_idx_all = []
            for data, sample_idx in tqdm(channel_dataloader[phase]):
                sample_idx = sample_idx.to(device)

                t_range = list(range(len(data.edge_index)))
                T_range = get_T_range(t_range_list=t_range, T_repeats=T_repeats, shuffle=False)

                eval_metrics = expert_policy.eval(data=data.to(device),
                                                  T_range = T_range,
                                                  phase = phase
                                                  )
                
                Ps = eval_metrics['Ps'] # [T * t_repeats, B*n, 1]
                Ps = Ps.view(Ps.shape[0], -1, args.n)
                Ps = Ps[-(Ps.shape[0] // 2):] # don't sample from the transient phase
                Ps = torch.moveaxis(Ps, source=0, destination=-2).unsqueeze(-1) # [B, TT, n, 1]

                rates = eval_metrics['rates'] # [T * t_repeats, B*n, 1]
                rates = rates.view(rates.shape[0], -1, args.n)
                rates = rates[-(rates.shape[0] // 2):] # don't sample from the transient phase
                rates = torch.moveaxis(rates, source=0, destination=-2).unsqueeze(-1) # [B, TT, n, 1]

                rates_all.append(rates)
                # lambdas_all.append(lambdas)
                sample_idx_all.append(sample_idx)

            sample_idx_all = torch.cat(sample_idx_all, dim = 0)
            Ps_all = torch.cat(Ps_all, dim = 0) / expert_policy.channel_config.P_max # normalized to [0, 1] range
            rates_all = torch.cat(rates_all, dim = 0)

            Ps_all = scatter(src=Ps_all, index=sample_idx_all, dim=0, reduce="mean")
            rates_all = scatter(src=rates_all, index=sample_idx_all, dim=0, reduce="mean")
            
            log_variables = {'Ps': torch.moveaxis(Ps_all.squeeze(-1), source = 0, destination=1),
                             'rates': torch.moveaxis(rates_all.squeeze(-1), source = 0, destination=1)
                            # 'lambdas': torch.moveaxis(lambdas_all.squeeze(-1), source = 0, destination=1)
                            }

            if accelerator is None or accelerator.is_local_main_process and loggers[phase] is not None:
                # expert_policy.log(loggers = [logger for logger in loggers['test'] if logger.log_metric in log_variables.keys()], all_variables = log_variables, n = args.n) 
                expert_policy.log(loggers = [logger for logger in loggers[phase] if logger.log_metric in log_variables.keys()], all_variables = log_variables, n = args.n) 

        else:
            ### we can add other policies here ###
            raise NotImplementedError
        

        ### Move the dataset to cpu() ###
        Ps_all = Ps_all.to("cpu")

        if phase == 'train': # standardizing the dataset helps with training a diffusion model.
            p_mean, p_std = Ps_all.mean().item(), Ps_all.std().item()
            print('Dataset mean: ', p_mean, '\tDataset std: ', p_std)
            # The preceeding standardization is sensitive to dataset size and value range.
            
            # The following standardization scales the normalized tx powers to [-1, 1] range.
            p_mean = 0.5 
            p_std = .5
            print(f'Using standardization method x <- (x - {p_mean} / {p_std}).')
            # p_std = p_std * data.num_nodes**0.5
            
        channel_idx = range(len(channel_dataloader[phase].dataset))

        data_list = [Data_PowerAllocPolicy(x=Ps_all[i],
                                           channel_id=channel_dataloader[phase].dataset[i][1],
                                           transmitters_index=channel_dataloader[phase].dataset[i][0].transmitters_index,
                                           num_nodes=channel_dataloader[phase].dataset[i][0].num_nodes,
                                           m=channel_dataloader[phase].dataset[i][0].m,
                                           transform_x=standardize_data(mean=p_mean, std=p_std),
                                           weights=sampler_weights
                                           ) for i in channel_idx
                                           ]
        
        p_dataset[phase] = WirelessDataset(data_list=data_list, p_mean = p_mean, p_std = p_std)

        ## Update channel dataset with power alloc policy dataset and the standardization method. ##
        for (channel_data, channel_id), (p_data, _) in zip(channel_dataloader[phase].dataset, p_dataset[phase]):
            channel_data.x = p_data.x
            channel_data.transform_x = p_data.transform_x
            # channel_data.y = p_data.y

        # p_dataloader[phase] = DataLoader(dataset=p_dataset[phase],
        #                                  batch_sampler=BatchSampler(RandomSampler(p_dataset[phase]),
        #                                                             num_samples=40,
        #                                                             replacement=True
        #                                                             ),
        #                                  batch_size=channel_dataloader[phase].batch_size,
        #                                  shuffle = (phase == 'train')
        #                                 )

        # Although unused, one can return a dataloader of concatenation of channel and policy datasets. 
        channel_p_joint_dataset[phase] = ConcatDataset([channel_dataloader[phase].dataset, p_dataset[phase]])


        # Write a custom collate_fn that returns also edge_index_t and edge_weight_t as edge_index[t] and edge_weight[t] for a uniform randomly picked t.




        diffusion_dataloader[phase] = DataLoader(dataset = channel_dataloader[phase].dataset,
                                                 batch_size=args.batch_size_diffusion,
                                                 shuffle=(phase == 'train'),
                                                #  collate_fn=collate_with_time_sample,
                                                 )

    return p_dataset, diffusion_dataloader