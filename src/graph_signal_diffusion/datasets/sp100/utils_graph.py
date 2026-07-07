import torch
import torch_geometric
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Arc, FancyArrowPatch, PathPatch
import numpy as np
import seaborn as sns


# ---------------------------------------------------------------------------
# Paper-figure rcParams — quarantined (Phase 2 of plot-style cleanup).
# See docstring in sp500/utils_graph.py for context.
# ---------------------------------------------------------------------------
_W = 469 / 72
_LABEL_SCALE = 2.25
_TICK_SCALE = 2.25
_LEGEND_SCALE = 2.0

PAPER_FIGURE_RC_PARAMS: dict = {
    "figure.figsize": (_W, _W * 2 / 3),
    "lines.markersize": 10,
    "axes.titlesize": 10 * _LABEL_SCALE,
    "axes.labelsize": 10 * _LABEL_SCALE,
    "xtick.labelsize": 10 * _TICK_SCALE,
    "ytick.labelsize": 10 * _TICK_SCALE,
    "legend.fontsize": 10 * _LEGEND_SCALE,
    "legend.title_fontsize": 10 * _LEGEND_SCALE,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsfonts, lmodern}",
    "font.family": "lmodern",
    "axes.grid": True,
    "grid.alpha": 0.75,
    "grid.linestyle": "--",
    "grid.color": "gray",
    "figure.constrained_layout.use": True,
    "legend.fancybox": True,
    "legend.framealpha": 0.5,
    "legend.edgecolor": "black",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Bitstream Vera Sans",
    "mathtext.it": "Bitstream Vera Sans:italic",
    "mathtext.bf": "Bitstream Vera Sans:bold",
}



def draw_stocks_graph_by_sector(stocks_graph, save_path):
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize=(12, 5))

        nx.draw(stocks_graph, ax=ax, with_labels=True, node_size=500, node_color='skyblue', font_size=8, font_weight='bold', font_color='black', pos=nx.spring_layout(stocks_graph, k=0.4))
        ax.set_title('Stocks Graph by Sector')
        ax.grid(True)

    plt.savefig(save_path, dpi = 300, bbox_inches='tight')
    plt.close(fig)


def draw_traffic_graph_by_road_connection(traffic_graph, save_path):
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize=(16, 7))
        nx.draw(traffic_graph, ax=ax, with_labels=True, node_size=500, node_color='skyblue', font_size=8, font_weight='bold', font_color='black', pos=nx.spring_layout(traffic_graph, k=.5))
        ax.set_title('Graph of Traffic Detectors by Road Connection')
        ax.grid(True)

    plt.savefig(save_path, dpi = 300, bbox_inches='tight')
    plt.close(fig)


def draw_stocks_graph_by_correlation(fundamentals_corr_graph, save_path):
    with matplotlib.rc_context(PAPER_FIGURE_RC_PARAMS), sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize=(12, 5))
        nx.draw(fundamentals_corr_graph, ax=ax, with_labels=True, node_size=500, node_color='skyblue', font_size=8, font_weight='bold', font_color='black', pos=nx.spring_layout(fundamentals_corr_graph))
        ax.set_title('Stocks Graph by Correlation')
        ax.grid(True)

    plt.savefig(save_path, dpi = 300, bbox_inches='tight')
    plt.close(fig)


def draw_merged_stocks_graph(graph, save_path):

    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize=(12, 5))
        nx.draw(graph, ax=ax, with_labels=True, node_size=500, node_color='skyblue', font_size=8, font_weight='bold', font_color='black', pos=nx.spring_layout(graph, k=.5))
        ax.set_title(f'Merged Stock Relation Graph (Mean degree: {np.mean([degree for node, degree in graph.degree]):.2f}, Density: {nx.density(graph):.4f})')
        ax.grid(True)

    plt.savefig(save_path, dpi = 300, bbox_inches='tight')
    plt.close(fig)


# utility function
def rescale_edge_weights(edge_weight, max_edge_weight = 1., scale = 'linear'):
    min_w, max_w = edge_weight.min(), edge_weight.max()
    if scale == 'linear':
        edge_weight = edge_weight * max_edge_weight / (max_w - min_w)
    elif scale == 'exponential':
        edge_weight = max_edge_weight * edge_weight.exp() / edge_weight.exp().max()
    else:
        raise NotImplementedError
    
    edge_weight = edge_weight - edge_weight.min()
    edge_weight.data.clamp_(min = 0.01, max = max_edge_weight)

    return edge_weight


def curved_edges(G, pos, dist_ratio=0.05):
    """
    Generate curved edge positions for visualization, including self-loops.
    
    Parameters:
    - G (networkx.Graph): The graph.
    - pos (dict): Node positions (output from a layout algorithm).
    - dist_ratio (float): Distance ratio for how far the curve is from the straight line.
    
    Returns:
    - edge_pos (dict): Dictionary where keys are edges and values are lists of points defining the curve.
    """
    edge_pos = {}
    for edge in G.edges():
        start, end = pos[edge[0]], pos[edge[1]]
        if edge[0] == edge[1]: # handle self-loops
            # # Self-loop: generate a circular arc around the node
            # rad = np.pi / 4
            # offset = dist_ratio
            # control1 = (start[0] + offset, start[1] + offset)
            # control2 = (start[0] - offset, start[1] + offset)
            # edge_pos[edge] = [start, control1, control2, start]

            # Self-loop: generate a circular arc around the node
            rad = np.pi / 4
            offset = 1 # 0.15
            theta = np.linspace(0, 2*np.pi, 100)
            x = start[0] + offset * (1 + np.cos(theta))
            y = start[1] + offset * (1 + np.sin(theta))
            edge_pos[edge] = list(zip(x, y))


        else:
            rad = np.arctan2(end[1] - start[1], end[0] - start[0])
            control = (
                (start[0] + end[0]) / 2 + dist_ratio * np.cos(rad + np.pi/2),
                (start[1] + end[1]) / 2 + dist_ratio * np.sin(rad + np.pi/2)
            )
            edge_pos[edge] = [start, control, end]
    return edge_pos


def visualize_graph(data, ax, node_color='blue', edge_color='black', with_labels=True, node_size=50, font_size = 4, curved=True, margin = 0.03):
    """
    Visualizes a graph using PyTorch Geometric and Matplotlib with optional curved edges and handling of self-loops.
    
    Parameters:
    - data (torch_geometric.data.Data): The graph data object from PyTorch Geometric.
    - node_color (list or torch.Tensor, optional): Colors for the nodes. Should be of length `num_nodes`.
    - edge_color (list, optional): Colors for the edges.
    - edge_weights (list or torch.Tensor, optional): Weights for the edges.
    - with_labels (bool, optional): Whether to draw labels on the nodes.
    - node_size (int, optional): Size of the nodes.
    - font_size (int, optional): Fontsize
    - curved (bool, optional): Whether to draw curved edges.
    - margin (float, optional): Margin to add between arrow tips and target nodes.
    """

    # edge_linewidth = 0.1

    # Convert the PyTorch Geometric data object to a NetworkX graph
    G = to_networkx(data, to_undirected=False, remove_self_loops=False)
    
    # Extract the node positions from the data object (if they exist)
    if 'pos' in data:
        pos = {i: coord for i, coord in enumerate(data.pos.cpu().numpy())}
    else:
        pos = nx.spring_layout(G)  # Generate a layout if node positions are not provided
    
    # Extract edge weights if provided
    if data.edge_weight is not None:
        if isinstance(data.edge_weight, torch.Tensor):
            edge_weights = data.edge_weight.cpu().numpy()
        edge_labels = {(u, v): f'{w:.2f}' for (u, v, w) in zip(data.edge_index[0].tolist(), data.edge_index[1].tolist(), edge_weights)}
    else:
        edge_weights = None
        edge_labels = None
    
    # plt.figure(figsize=figsize, facecolor='black')
    # ax = plt.axes()
    # ax.set_facecolor('#eafff5') 
    
    # Plot nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, cmap=plt.get_cmap('Set2'), ax=ax)
    
    # if with_labels:
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_color='black', ax=ax)
    
    if curved:
        # Generate curved edge positions
        edge_pos = curved_edges(G, pos)

        for (u, v), points in edge_pos.items():

            if u == v:  # Self-loop
                # # Create a circular arc for the self-loop
                # path = Path(points, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY])
                # patch = PathPatch(path, edgecolor=edge_color if edge_color else 'black',
                #                   linewidth=float(edge_labels[(u, v)]) if edge_labels is not None else 1,
                #                   fill=False)
                # plt.gca().add_patch(patch)
                # # Optionally, you can add an arrow in the self-loop arc
                # arrow = FancyArrowPatch(path=path, connectionstyle="arc3,rad=0.5",
                #                         arrowstyle='->', mutation_scale=10, color=edge_color if edge_color else 'black')
                # plt.gca().add_patch(arrow)

                # Create a circular arc for the self-loop
                rad_x = 0.1
                rad_y = 0.1
                center = pos[u] - np.array([rad_x/2, 0.])
                text_pos = 2.6 * center - pos[u]
                arc = Arc(center, rad_x, rad_y, angle=0, theta1=0, theta2=350, color=edge_color if edge_color else 'red', linewidth=float(edge_labels[(u, u)]) if edge_labels is not None else 1, linestyle='-')
                plt.gca().add_patch(arc)
                # Optionally, you can add an arrow in the self-loop arc
                # arrow = FancyArrowPatch(points[0], points[1], connectionstyle="arc3,rad=0.5",
                #                         arrowstyle='->', mutation_scale=10, color=edge_color if edge_color else 'black')
                # plt.gca().add_patch(arrow)

                if edge_labels is not None and with_labels:
                    plt.text(text_pos[0], text_pos[1], f'{float(edge_labels[(u, u)]):.2f}', fontsize=font_size, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.0))
            else:
                end = np.array(points[2])
                control = np.array(points[1])

                direction = (end - control)
                norm = np.linalg.norm(direction)
                if norm != 0:  # Avoid division by zero
                    direction = direction / norm
                new_end = end - direction * margin
                
                # path = Path([points[0], points[1], new_end], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
                # # patch = PathPatch(path, edgecolor=edge_color if edge_color is not None else 'black', linewidth=edge_weights[u][v] if edge_weights is not None else 1)
                # patch = PathPatch(path, edgecolor=edge_color if edge_color is not None else 'black', linewidth=float(edge_labels[(u,v)]) if edge_labels is not None else 1, fill = False)
                # plt.gca().add_patch(patch)


                # Adding arrow at the end of the curve
                path = Path([points[0], points[1], new_end], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
                arrow = FancyArrowPatch(path=path, connectionstyle="arc3,rad=0.2",
                                        arrowstyle='->', mutation_scale=10, color=edge_color if edge_color else 'black', linewidth = float(edge_labels[(u, v)]))
                plt.gca().add_patch(arrow)

                if edge_labels is not None and with_labels:
                    # Calculate the midpoint for the annotation
                    mid_point = ((points[0][0] + points[2][0]) / 2, (points[0][1] + points[2][1]) / 2)
                    mid_point = points[1]
                    plt.text(mid_point[0], mid_point[1], f'{float(edge_labels[(u, v)]):.2f}', fontsize=font_size, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.0))

        edge_label_pos = pos
    else:
        # Draw straight edges
        edge_label_pos = pos
        nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_weights if edge_weights is not None else 1.0, arrowsize=node_size // 5, ax=ax)
    
        if edge_labels is not None and with_labels:
            nx.draw_networkx_edge_labels(G, edge_label_pos, edge_labels=edge_labels, ax=ax)
    
    # plt.tight_layout()
    # plt.show()

# # Example usage:
# from torch_geometric.data import Data

# # Example graph data
# edge_index = torch.tensor([[0, 0, 1, 2, 3, 3], [0, 1, 2, 3, 0, 2]], dtype=torch.long)
# node_pos = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float)
# edge_weights = torch.tensor([1.0, 0.5, 0.8, 1.2, 0.7, -0.4], dtype=torch.float)

# # Creating a PyG Data object
# data = Data(pos=node_pos, edge_index=edge_index, edge_weight = edge_weights)
    
# figsize=(8, 8)
# plt.figure(figsize=figsize, facecolor='black')
# ax = plt.axes()
# ax.set_facecolor('#eafff5') 

# # Visualize the graph with curved edges and edge weights
# visualize_graph(data, ax=ax, curved=True)