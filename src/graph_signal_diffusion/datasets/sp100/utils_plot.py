import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
# from torch_geometric.data import Data
from .dataset import StocksDataDiffusion
from typing import Union

import logging
logger = logging.getLogger(__name__)


def plot_batched_data(data: StocksDataDiffusion,
                      stocks_idx: Union[np.ndarray, list, None], 
                      plot_target: str = "close_price",
                      save_dir: str = None,
                      save_ext: str = ".pdf") -> None:
    """
    Test batched Data and plot time evolution of closing prices for 4 example stocks.
    """

    # print("Data.x.shape: ", data.x.shape) # [num_stocks x num_timestamps, num_features, past_window]
    # print("Data.edge_index.shape: ", data.edge_index.shape)
    # print("Data.edge_weight.shape: ", data.edge_weight.shape)
    # print("Data.y.shape: ", data.y.shape)
    # print("Data.close_price.shape: ", data.close_price.shape) # [num_stocks x num_timestamps, past_window]
    # print("Data.close_price_y.shape: ", data.close_price_y.shape) # [num_stocks x num_timestamps, 1]

    # This legacy plotting code expects data.x in shape [num_stocks * num_timestamps, num_features, past_window]
    # Undo the axis swapping done in the dataset __getitem__ method.
    x, y, close_price, close_price_y = data.x, data.y, data.close_price, data.close_price_y
    x = x.permute(0, 2, 1)  # [num_stocks * num_timestamps, past_window, num_features] -> [num_stocks * num_timestamps, num_features, past_window]
    y = y.squeeze(-1)  # [num_stocks * num_timestamps, future_window, 1] -> [num_stocks * num_timestamps, future_window]
    close_price = close_price.squeeze(-1) # [num_stocks * num_timestamps, past_window, 1] -> [num_stocks * num_timestamps, past_window]
    close_price_y = close_price_y.squeeze(-1) # [num_stocks * num_timestamps, future_window, 1] -> [num_stocks * num_timestamps, future_window]


    num_timestamps = len(data.ptr) - 1
    num_stocks = x.shape[0] // num_timestamps
    
    num_features, past_window = x.shape[1], x.shape[2]  # num_features, past_window

    logger.info(f"Number of timestamps in the batch: {num_timestamps}")
    logger.info(f"Number of stocks in the batch: {num_stocks}")
    logger.info(f"Number of features in x: {num_features}")
    logger.info(f"Past window size in x: {past_window}")

    if stocks_idx is None:
        stocks_idx = np.random.choice(num_stocks, 4, replace = False)

    if plot_target == "Closing_Price":
        target = close_price.reshape(num_timestamps, num_stocks, -1)

    elif plot_target == "Closing_Price_y":
        target = close_price_y.reshape(num_timestamps, num_stocks, -1)

    elif plot_target == "y":
        target = y.reshape(num_timestamps, num_stocks, -1)

    elif isinstance(plot_target, tuple) and plot_target[0] == "y":
        target = y.reshape(num_timestamps, num_stocks, -1)
        plot_target = f"{plot_target[1]} (y)"

    elif isinstance(plot_target, tuple) and plot_target[0].startswith('x[') and plot_target[0].endswith(']'):
        try:
            feature_idx = int(plot_target[0][2:-1])  # Extract the index from 'x[idx]'
            if 0 <= feature_idx < num_features:
                target = x[:, feature_idx, :].reshape(num_timestamps, num_stocks, -1)
                plot_target = f"{plot_target[1]} (Feature {feature_idx})"
            else:
                raise ValueError(f"Feature index {feature_idx} is out of range [0, {num_features-1}]")
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError(f"Invalid feature index format in '{plot_target}'. Expected format: 'x[idx]' where idx is an integer.")
            else:
                raise e

    else:
        raise ValueError(f"Invalid plot_target '{plot_target}'. Must be one of: 'Closing_Price', 'Closing_Price_y', 'y', or 'x[idx]' where idx is a feature index.")

    logger.info(f"Target.shape: {target.shape}") # [num_timestamps, num_stocks, target_window]


    if save_dir is not None:
        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))

            for idx, stock_idx in enumerate(stocks_idx):
                ax = axs[idx // 2, idx % 2]
                ax.plot(target[:, stock_idx].detach().numpy(), label=None) 
                ax.set_title(f"Stock {stock_idx}")
                ax.set_xlabel("Timestamp (Date)")
                ax.set_ylabel(plot_target)
                # ax.legend()

            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/batched_data_{plot_target}" + save_ext, dpi=300, bbox_inches='tight')
            plt.close(fig)