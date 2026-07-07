#!/usr/bin/env python3
"""
Collect consecutive samples from a trained primal-dual model.

This script loads a trained model checkpoint and collects M samples from
the final model state, representing consecutive epochs. This provides samples
that match the M-averaged stochastic policy metrics reported in epoch_summaries.jsonl.

Usage:
    python scripts/collect_consecutive_samples.py --checkpoint-dir <path>
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graph_signal_diffusion.models.power_allocation_gnn import PowerAllocationGNN
from graph_signal_diffusion.trainers.dual_optimizer import DualOptimizer
from graph_signal_diffusion.datasets.wra.primal_dual_dataset import PrimalDualDataset
from graph_signal_diffusion.utils.rate_calculator import compute_ergodic_rates
from torch_geometric.loader import DataLoader


def main():
    parser = argparse.ArgumentParser(description='Collect consecutive samples from trained model')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory containing trained model checkpoint')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of consecutive samples to collect (default: 100)')
    parser.add_argument('--checkpoint-name', type=str, default='pre_collection_checkpoint.pt',
                        help='Name of checkpoint file to load (default: pre_collection_checkpoint.pt)')
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)
    
    checkpoint_path = checkpoint_dir / args.checkpoint_name
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"{'='*70}")
    print(f"Collecting Consecutive Samples")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Number of samples: {args.num_samples}")
    print(f"{'='*70}\n")
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    
    # Extract configuration from checkpoint
    config = checkpoint.get('config', {})
    system_params = checkpoint['system_params']
    
    # Load dataset (to get channel data)
    print("Loading dataset...")
    dataset_path = checkpoint.get('dataset_path', None)
    if dataset_path is None:
        print("Error: Dataset path not found in checkpoint")
        sys.exit(1)
    
    dataset = PrimalDualDataset.load(dataset_path)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # Initialize model
    print("Initializing model...")
    model = PowerAllocationGNN(
        in_channels=config.get('in_channels', 3),
        hidden_channels=config.get('hidden_channels', 64),
        out_channels=1,
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.0)
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Initialize dual optimizer (needed for r_min)
    dual_optimizer = DualOptimizer(
        num_networks=len(dataset),
        num_receivers_per_network=[data.num_nodes for data in dataset],
        r_min=checkpoint['dual_optimizer_state']['r_min'],
        dual_lr=checkpoint['dual_optimizer_state']['dual_lr'],
        momentum=checkpoint['dual_optimizer_state'].get('momentum', 0.0)
    )
    dual_optimizer.dual_multipliers = checkpoint['dual_optimizer_state']['dual_multipliers']
    
    # Collect consecutive samples
    print(f"\nCollecting {args.num_samples} consecutive samples...")
    samples = {}
    
    with torch.no_grad():
        for sample_idx in tqdm(range(args.num_samples), desc="Samples"):
            for batch in dataloader:
                batch = batch.to(device)
                batch_size = batch.num_graphs
                ptr = batch.ptr
                
                for idx in range(batch_size):
                    net_id = batch.network_id[idx].item()
                    
                    if net_id not in samples:
                        samples[net_id] = {}
                    
                    # Extract graph data
                    start_idx = ptr[idx]
                    end_idx = ptr[idx + 1]
                    
                    x_graph = batch.x[start_idx:end_idx]
                    edge_mask = (batch.batch[batch.edge_index[0]] == idx)
                    edge_index_graph = batch.edge_index[:, edge_mask] - start_idx
                    edge_weight_graph = batch.edge_weight[edge_mask]
                    
                    H_inst = batch.H_instantaneous[idx].to(device)
                    associations = batch.associations[idx].to(device)
                    
                    # Forward pass
                    power_per_receiver = model(x_graph, edge_index_graph, edge_weight_graph)
                    power = associations @ power_per_receiver
                    
                    # Compute rates
                    ergodic_rates = compute_ergodic_rates(
                        power_allocation=power,
                        H_instantaneous=H_inst,
                        associations=associations,
                        noise_var=system_params['noise_var']
                    )
                    
                    # Initialize on first sample
                    if len(samples[net_id]) == 0:
                        samples[net_id] = {
                            'H_instantaneous': H_inst.cpu().numpy(),
                            'associations': associations.cpu().numpy(),
                            'power_samples': [],
                            'rate_samples': [],
                        }
                        if hasattr(batch, 'network_seed') and batch.network_seed is not None:
                            samples[net_id]['network_seed'] = batch.network_seed[idx].item()
                    
                    # Append sample
                    samples[net_id]['power_samples'].append({
                        'power': power.cpu().numpy(),
                        'rates': ergodic_rates.cpu().numpy(),
                        'sum_rate': ergodic_rates.sum().item(),
                        'min_rate': ergodic_rates.min().item(),
                    })
    
    # Compute M-averaged statistics
    print(f"\n{'='*70}")
    print(f"M-Averaged Statistics (M={args.num_samples})")
    print(f"{'='*70}")
    
    for net_id in sorted(samples.keys()):
        network_data = samples[net_id]
        rates_all = np.array([s['rates'] for s in network_data['power_samples']])  # (M, n)
        rates_averaged = rates_all.mean(axis=0)  # (n,)
        
        print(f"\nNetwork {net_id}:")
        print(f"  M-averaged min rate:  {rates_averaged.min():.4f} bits/s/Hz")
        print(f"  M-averaged 5th %-ile: {np.percentile(rates_averaged, 5):.4f} bits/s/Hz")
        print(f"  M-averaged mean rate: {rates_averaged.mean():.4f} bits/s/Hz")
        print(f"  Absolute min (worst): {rates_all.min():.4f} bits/s/Hz")
    
    # Save samples
    output_path = checkpoint_dir / "collected_samples_consecutive.npz"
    samples_dict = {}
    
    for net_id, network_data in samples.items():
        if 'network_seed' in network_data:
            samples_dict[f'network_{net_id}_seed'] = network_data['network_seed']
        
        samples_dict[f'network_{net_id}_H_instantaneous'] = network_data['H_instantaneous']
        samples_dict[f'network_{net_id}_associations'] = network_data['associations']
        
        for sample_idx, sample in enumerate(network_data['power_samples']):
            samples_dict[f'network_{net_id}_power_{sample_idx}'] = sample['power']
            samples_dict[f'network_{net_id}_rates_{sample_idx}'] = sample['rates']
    
    np.savez(output_path, **samples_dict)
    
    print(f"\n{'='*70}")
    print(f"Samples saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
