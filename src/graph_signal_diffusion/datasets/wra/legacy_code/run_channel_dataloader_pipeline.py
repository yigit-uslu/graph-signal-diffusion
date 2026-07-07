from collections import defaultdict
import copy
import io
from PIL import Image
from matplotlib import pyplot as plt
import wandb
import torch
import numpy as np
# from utils.logger_utils import make_channel_loggers
from .channel import create_channels, create_channel_dataset, create_perturbed_channels
from .data_utils import create_channel_dataloader

# from core.config import PHASES
from .wrapper_utils import make_channel_perturbation_wrapper

PHASES = ['train', 'val', 'test']

def baseline_log(accelerator, baseline_metrics, log_dict = {}, r_min = None):


    #### Log baseline metrics to Wandb #### 
    if wandb.run is not None and baseline_metrics is not None:

        keys = ["1st-percentile rate", "5th-percentile rate", "mean rate"]

        eval_baselines = [k[0] for k in baseline_metrics.keys() if k[1] == 'rates']

        for key in keys:

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))  

            for alg in eval_baselines:

                rates = torch.stack(baseline_metrics[alg, 'rates'], dim = 0).detach().cpu()
                T_max = rates.shape[1]
                timesteps = np.array([T_max // 10, T_max // 5, int(0.4 * T_max), T_max])
                # timesteps = np.array([50, 100, 200, 500])

                ergodic_rates = []
                for t in timesteps:
                    if key == "mean rate":
                        value = torch.mean(rates[:, :t], dim = 1).mean().item()
                    elif key == "1st-percentile rate":
                        value = torch.mean(rates[:, :t], dim = 1).quantile(q = 0.01).item()
                    elif key == "5th-percentile rate":
                        value = torch.mean(rates[:, :t], dim = 1).quantile(q = 0.05).item()

                    ergodic_rates.append(value)

                ergodic_rates = np.array(ergodic_rates)

                ax.plot(timesteps, ergodic_rates, linestyle = '-', marker = 'x', label = f"{alg}")

            if r_min is not None:
                ax.axhline(xmin = 0, xmax = 1, y = r_min, color = 'r', linestyle = '--', label = r'$f_{\min}$')

            ax.set_xlabel(r"Timesteps")
            ax.set_ylabel(r"Ergodic rates (bps/Hz)")
            ax.legend(loc = 'best')
            ax.grid(True)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            # wandb.log({"Figures/Xgen_hist": wandb.Image(Image.open(buf)), "chkpt_step": chkpt_step}, step=chkpt_step)
            plt.close(fig)
        
            log_item = wandb.Image(Image.open(buf))

            log_dict.update({f"{key}": log_item})

    return log_dict





def run_channel_dataloader_pipeline(args, arg_groups, accelerator,
                                    experiment_name,
                                    log_prefix = "Baselines",
                                    channel_save_path_override = False,
                                    phases = None):

    ############ Create/load wireless channel dataset ################
    channels = defaultdict(list)
    channel_dataset = defaultdict(list)
    baseline_metrics = defaultdict(list)
    channel_dataloader = defaultdict(list)
    channel_wrappers = defaultdict(list)

    phases = phases if phases is not None else PHASES
    for phase in phases:

        if arg_groups['Channel-Perturbation'].perturbation_mode not in ['none', None]:
            channel_wrappers[phase].append(make_channel_perturbation_wrapper(
                args=args, perturb_args=arg_groups['Channel-Perturbation'],
                phase=phase, save_rendered_perturb_path=f'{args.experiment_name}/channel-logs/{phase}')
                )
        
        else:
            print(f"No channel perturbation wrapper is initialized for {phase} phase.")

        if arg_groups['RRM'].num_channels[phase] in [0, None]:
            accelerator.print(f"Skipping {phase} phase as num_channels is set to {arg_groups['RRM'].num_channels[phase]}.")
            continue

        print(f"Creating channels for {phase} phase with num nodes = {arg_groups['RRM'].n}, num channels = {arg_groups['RRM'].num_channels[phase]}...")
        # Create CHANNEL object instances
        channels[phase] = create_channels(accelerator=accelerator,
                                          m=arg_groups['RRM'].n,
                                          n=arg_groups['RRM'].n,
                                          density_mode=arg_groups['RRM'].density_mode,
                                          base_R = arg_groups['RRM'].base_R,
                                          num_channels=arg_groups['RRM'].num_channels[phase], # if phase == 'train' else arg_groups['RRM'].batch_size_channels,
                                          channel_load_path=arg_groups['RRM'].load_channel_path[phase],
                                          # channel_save_path=None,
                                          channel_save_path=None if channel_save_path_override is True or arg_groups['RRM'].load_channel_path[phase] not in [None, 'None', 'none'] else f'{args.experiment_name}/data/{phase}/channel_data_{experiment_name}.json',
                                          fast_fading=arg_groups['RRM'].fast_fading,
                                          channel_seed=args.random_seed if arg_groups['RRM'].fix_channel_gen_seed and phase in ['train', 'val', 'test'] else None,
                                          channel_wrappers=channel_wrappers[phase],
                                          max_n_per_subnetwork=arg_groups['RRM'].max_n_per_subnetwork,
                                          channel_kws = vars(arg_groups['Channel']) if arg_groups['Channel'].use_args else None,
                                          )

        if args.train_on_single_network_diffusion:
            if phase in ['val', 'test']: 
                channels[phase] = copy.deepcopy(channels['train'])
            channels_list = channels[phase][0:1] # pick first network arbitrarily
            print('Training on single graph.')
            
        else:
            channels_list = channels[phase]
        
        avg_graph_data = None
        channel_dataset[phase], baseline_metrics[phase], avg_graph_data = create_channel_dataset(accelerator=accelerator,
                                                                                                channels_list=channels_list,
                                                                                                P_max_dBm=arg_groups['RRM'].P_max_dBm,
                                                                                                BW=arg_groups['RRM'].BW,
                                                                                                noise_PSD_dBm=arg_groups['RRM'].noise_PSD_dBm,
                                                                                                r_min=arg_groups['SA-train-algo'].r_min,
                                                                                                loggers=None, # channel_loggers[phase],
                                                                                                normalization="spectral" if arg_groups['RRM'].use_graph_laplacian else "gcn" if arg_groups['RRM'].use_gcn_norm else None,
                                                                                                eval_baselines = args.eval_baselines, # ['FR', 'UR', 'WMMSE'], # ['ITLinQ', 'WMMSE', 'FR', 'UR'], # if args.expert_policy in ['ITLinQ', 'WMMSE'] else None,
                                                                                                avg_graph_data = avg_graph_data, # will be filled up for the first (training) phase
                                                                                                edge_sparsity=arg_groups['RRM'].edge_sparsity, edge_threshold=arg_groups['RRM'].edge_threshold,
                                                                                                channel_conversion_method=arg_groups['RRM'].channel_conversion_method,
                                                                                                channel_normalization_method=arg_groups['RRM'].channel_normalization_method,
                                                                                                permute_graph=False,
                                                                                                channel_features = ('WMMSE', 'interference_powers') if args.diffusion_policy == 'interference-power' else ('FR', 'rates'), # ('FR', 'rates') if args.diffusion_policy == 'state-augmented-power-allocation' else None,
                                                                                                )
        

        if wandb.run is not None and phase == 'test' and log_prefix is not None:
            # Log the baseline metrics to wandb
            log_dict = baseline_log(accelerator=accelerator,
                                    baseline_metrics=baseline_metrics[phase],
                                    log_dict={},
                                    r_min=arg_groups['SA-train-algo'].r_min
                                    )
            
            accelerator.log({f"{log_prefix}/{k}": v for k, v in log_dict.items()})


        # Create a dataloader of CHANNEL datasets to train the state-augmented algo.
        channel_dataloader[phase] = create_channel_dataloader(data_list=channel_dataset[phase], 
                                                                    #   data_list=[channel_dataset['train'][0]] * 1024 if phase == 'train' else [channel_dataset['train'][0]] * 64, # len(channel_dataset[phase]),
                                                                batch_size=arg_groups['RRM'].batch_size_channels,
                                                                shuffle=(phase == 'train')
                                                                )

        accelerator.print(f"Saving baseline metrics for {phase} phase to disk...")
        torch.save(baseline_metrics[phase], f'{args.experiment_name}/data/{phase}/baseline_metrics.pth')

        
    # accelerator.print("Waiting for all processes to finish...")
    # accelerator.wait_for_everyone()
    accelerator.print('Channels, channel datasets and dataloaders are created successfully...')

    channel_loggers = None

    return channel_dataloader, channel_dataset, channel_loggers, baseline_metrics




def run_perturbed_channel_dataloader_pipeline(args, arg_groups, accelerator,
                                                experiment_name):

    ############ Create/load wireless channel dataset ################
    channels = defaultdict(list)
    channel_dataset = defaultdict(list)
    baseline_metrics = defaultdict(list)
    channel_dataloader = defaultdict(list)

    # channel_loggers = make_channel_loggers(args = args, log_path = f'{args.experiment_name}/channel-logs')

    # with accelerator.main_process_first():
    for phase in PHASES:
        # Create CHANNEL object instances

        save_unperturbed_channel_path = f'{args.experiment_name}/data/train/channel_data_{experiment_name}.json'

        if phase == 'train':
            channels[phase] = create_perturbed_channels(accelerator=accelerator,
                                                        m=arg_groups['RRM'].n,
                                                        n=arg_groups['RRM'].n,
                                                        density_mode=arg_groups['RRM'].density_mode,
                                                        base_R = arg_groups['RRM'].base_R,
                                                        num_channels=arg_groups['RRM'].num_channels[phase], # if phase == 'train' else arg_groups['RRM'].batch_size_channels,
                                                        channel_load_path=None,
                                                        # channel_save_path=None,
                                                        channel_save_path=save_unperturbed_channel_path,
                                                        fast_fading=arg_groups['RRM'].fast_fading,
                                                        use_single_channel=True,
                                                        permute_nodes = False
                                                        )
            
        elif phase == 'val':
            channels[phase] = create_perturbed_channels(accelerator=accelerator,
                                                        m=arg_groups['RRM'].n,
                                                        n=arg_groups['RRM'].n,
                                                        density_mode=arg_groups['RRM'].density_mode,
                                                        base_R = arg_groups['RRM'].base_R,
                                                        num_channels=arg_groups['RRM'].num_channels[phase], # if phase == 'train' else arg_groups['RRM'].batch_size_channels,
                                                        channel_load_path=save_unperturbed_channel_path,
                                                        # channel_save_path=None,
                                                        channel_save_path=f'{args.experiment_name}/data/{phase}/channel_data_{experiment_name}.json',
                                                        fast_fading=arg_groups['RRM'].fast_fading,
                                                        perturb_tx_rx_locs_factors=[1e-4 * 2**(i-1) if i > 2 else 0.0 for i in range(arg_groups['RRM'].num_channels[phase])],
                                                        permute_nodes = [True if i > 0 else False for i in range(arg_groups['RRM'].num_channels[phase])]
                                                        )
        else:
            channels[phase] = copy.deepcopy(channels['val'])

        if args.train_on_single_network_diffusion:
            if phase in ['val', 'test']: 
                channels[phase] = copy.deepcopy(channels['train'])
            channels_list = channels[phase][0:1] # pick first network arbitrarily
            accelerator.print('Training on single graph.')
            
        else:
            channels_list = channels[phase]
        
        avg_graph_data = None
        channel_dataset[phase], baseline_metrics[phase], avg_graph_data = create_channel_dataset(accelerator=accelerator,
                                                                                                    channels_list=channels_list,
                                                                                                    P_max_dBm=arg_groups['RRM'].P_max_dBm,
                                                                                                    BW=arg_groups['RRM'].BW,
                                                                                                    noise_PSD_dBm=arg_groups['RRM'].noise_PSD_dBm,
                                                                                                    loggers=None, #channel_loggers[phase],
                                                                                                    spectral_normalization="spectral" if arg_groups['RRM'].use_graph_laplacian else "gcn" if arg_groups['RRM'].use_gcn_norm else None,
                                                                                                    # eval_baselines = [args.expert_policy] if args.expert_policy in ['ITLinQ', 'WMMSE'] else None,
                                                                                                    eval_baselines = ['ITLinQ', 'WMMSE', 'FR', 'UR'],
                                                                                                    avg_graph_data = avg_graph_data, # will be filled up for the first (training) phase
                                                                                                    edge_sparsity=arg_groups['RRM'].edge_sparsity, edge_threshold=arg_groups['RRM'].edge_threshold,
                                                                                                    channel_conversion_method=arg_groups['RRM'].channel_conversion_method,
                                                                                                    permute_graph=False
                                                                                                    )
        

        # Create a dataloader of CHANNEL datasets to train the state-augmented algo.
        n_lambdas = {'train': 1, 'val': 1, 'test': 1} # it may be helpful to repeat the same graph in the dataset to sample multiple lambdas per graph in each batch.
        channel_dataloader[phase] = create_channel_dataloader(data_list=channel_dataset[phase] * n_lambdas[phase], #
                                                                    #   data_list=[channel_dataset['train'][0]] * 1024 if phase == 'train' else [channel_dataset['train'][0]] * 64, # len(channel_dataset[phase]),
                                                                batch_size=arg_groups['RRM'].batch_size_channels * n_lambdas[phase],
                                                                shuffle=(phase == 'train')
                                                                )
        
    accelerator.wait_for_everyone()
    accelerator.print('Channels, channel datasets and dataloaders are created successfully...')

    return channel_dataloader, channel_dataset, channel_loggers := None, baseline_metrics