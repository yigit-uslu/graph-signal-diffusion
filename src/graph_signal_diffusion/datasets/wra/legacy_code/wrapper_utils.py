from .channel import ChannelPerturbationWrapper
import numpy as np
# from core.config import MAX_LOGGED_NETWORKS

MAX_LOGGED_NETWORKS = 10


def increment_perturb_seed(args, base_seed, increment, phase = None):
    """
    Increment the perturbation seed for the given phase.

    Args:
        current_seed: int, current seed value.
        increment: int, increment value.
        phase: str, 'train', 'val', 'test'

    Returns:
        new_seed: int, new seed value.
    """

    phase_seeds = {'train': 0, 'val': args.num_channels['train'], 'test': args.num_channels['train'] + args.num_channels['val']}
    new_seed = base_seed + increment
    new_seed = new_seed + phase_seeds[phase] if phase in ['train', 'val', 'test'] else new_seed

    return new_seed
  


def make_channel_perturbation_wrapper(args, perturb_args, phase = 'train', save_rendered_perturb_path = None):
    """
    Create a channel perturbation wrapper.

    Args:
        args: Namespace object containing the arguments.
        perturb_args: Namespace object containing the perturbation arguments.
        phase: str, 'train', 'val', 'test'

    Returns:
        thunk: A function that takes a channel object and returns a perturbed channel object.
    """
    
    if perturb_args.perturbation_mode not in ['none', None]:

        perturbation_type_arr = ['none'] + [perturb_args.perturbation_mode] * (args.num_channels[phase] - 1)


        # Handle perturbation seeding
        if perturb_args.perturbation_seed is not None and not perturb_args.fix_channel_perturbation_seed:
            perturbation_seed_list = [increment_perturb_seed(args, base_seed=perturb_args.perturbation_seed, increment=i, phase = phase) for i in range(1, args.num_channels[phase])]
        else:
            perturbation_seed_list = [perturb_args.perturbation_seed] * (args.num_channels[phase] - 1)


        if perturb_args.perturbation_mode in ['perturb_locs']:
            perturbation_params_arr = [{'delta_r': {"tx": 0.0, "rx": 0.0}, 'seed': perturb_args.perturbation_seed}] \
                + [{'delta_r': {"tx": 0.0, "rx": perturb_args.perturbation_strength * perturb_args.max_rx_loc_perturbation},
                    'seed': perturbation_seed_list[i]} for i in range(args.num_channels[phase] - 1)]
            # perturbation_params_arr = [{'delta_r': {"tx": 0.0, "rx": 0.0}}] + [{'delta_r': {"tx": 0.0, "rx": perturb_args.max_rx_loc_perturb * i / (args.num_channels[phase] - 1)}} for i in range(1, args.num_channels[phase])]

        elif perturb_args.perturbation_mode in ['perturb_angles']:
            perturbation_params_arr = [{'delta_r': {"tx": 0.0, "rx": 0.0}, 'delta_phi': 0.0, 'seed': perturb_args.perturbation_seed}] \
                + [{'delta_r': {"tx": 0.0, "rx": 0.0}, 'delta_phi': perturb_args.perturbation_strength * np.pi,
                    'seed': perturbation_seed_list[i]} for i in range(args.num_channels[phase] - 1)]
            

        elif perturb_args.perturbation_mode in ['resample_shadowing_loss']:
            perturbation_params_arr = [{'delta_r': {"tx": 0.0, "rx": 0.0}, 'seed': perturb_args.perturbation_seed}] \
                + [{'delta_r': {"tx": 0.0, "rx": 0.0}, 'seed': perturbation_seed_list[i]} for i in range(args.num_channels[phase] - 1)]

        else:
            raise ValueError(f"Invalid perturbation mode: {perturb_args.perturbation_mode}")
        
    else:
        perturbation_type_arr = ['none']
        perturbation_params_arr = [{'delta_r': {"tx": 0.0, "rx": 0.0}, 'delta_phi': 0.0, 'seed': perturb_args.perturbation_seed}]



    def thunk(channel, **kwargs):
        channel_id = kwargs.get("channel_id", 0)

        perturb_type = perturbation_type_arr[channel_id % len(perturbation_type_arr)]
        perturb_params = perturbation_params_arr[channel_id % len(perturbation_params_arr)]
        render_perturbation = False
        
        if len(perturbation_params_arr) > 0:
            render_perturbation = (1 + channel_id) % max(1, len(perturbation_params_arr) // MAX_LOGGED_NETWORKS) == 0 or channel_id == 0
        return ChannelPerturbationWrapper(channel, perturbation_type=perturb_type, perturbation_params=perturb_params,
                                          render_perturbation=render_perturbation,
                                          save_path=f"{save_rendered_perturb_path}/perturbations/network_{channel_id}" if save_rendered_perturb_path is not None else None)
    
    return thunk