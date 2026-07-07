import numpy as np
import torch
from graph_signal_diffusion.models.components.embeddings import TimeEmbedding, AdaptiveSinusoidalEmbedding


def test_embedding_separation(time_embed, num_timesteps):
    """
    Test if embeddings are well-separated across timesteps.
    
    Args:
        time_embed: TimeEmbedding module
        num_timesteps: Total number of timesteps
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # Generate embeddings for all timesteps
    timesteps = torch.arange(num_timesteps)
    embeddings = time_embed(timesteps).detach().cpu().numpy()
    
    # 1. Check nearest neighbor distances
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(embeddings, metric='cosine'))
    
    # For each timestep, find nearest neighbor (excluding itself)
    np.fill_diagonal(dist_matrix, np.inf)
    min_distances = dist_matrix.min(axis=1)
    
    print(f"Mean nearest neighbor distance: {min_distances.mean():.4f}")
    print(f"Min nearest neighbor distance: {min_distances.min():.4f}")
    print(f"Max nearest neighbor distance: {min_distances.max():.4f}")
    
    # Good embeddings should have:
    # - Adjacent timesteps close (distance < 0.1)
    # - Distant timesteps far (distance > 0.5)
    
    # 2. Visualize with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                c=timesteps, cmap='viridis', s=10)
    plt.colorbar(label='Timestep')
    plt.title('Time Embeddings (t-SNE visualization)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(f'./plot_embeddings/time_embeddings_tsne_{time_embed.__class__.__name__}_embed_dim_{embeddings.shape[1]}.png', dpi=150)
    plt.close()
    
    print(f"✅ Visualization saved to './plot_embeddings/time_embeddings_tsne_{time_embed.__class__.__name__}_embed_dim_{embeddings.shape[1]}.png'")
    
    # 3. Check if embeddings form a smooth manifold
    adjacent_distances = []
    for t in range(num_timesteps - 1):
        dist = np.linalg.norm(embeddings[t] - embeddings[t+1])
        adjacent_distances.append(dist)
    
    print(f"\nAdjacent timestep distances:")
    print(f"  Mean: {np.mean(adjacent_distances):.4f}")
    print(f"  Std:  {np.std(adjacent_distances):.4f}")
    
    # Should be relatively uniform (low std)
    if np.std(adjacent_distances) < 0.01:
        print("✅ Embeddings form a smooth manifold")
    else:
        print("⚠️  Embeddings may have discontinuities")



if __name__ == "__main__":
  
    # Test with your embedding
    embed_dim, num_timesteps = 512, 1000

    # time_embed = TimeEmbedding(embed_dim=embed_dim, num_timesteps=num_timesteps)
    time_embed = AdaptiveSinusoidalEmbedding(embed_dim=embed_dim, num_timesteps=num_timesteps)

    test_embedding_separation(time_embed, num_timesteps=num_timesteps)