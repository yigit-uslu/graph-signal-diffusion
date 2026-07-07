"""
Reusable components for graph neural network architectures.

Usage:
    from graph_signal_diffusion.models.components import (
        TimeEmbedding,
        MLPEmbedding,
        SinusoidalEmbedding,
        AdaptiveSinusoidalEmbedding,
        CascadeGNN,
        EncoderBlock,
        LearnableGraphPool
    )
"""

# Level 0: No dependencies
from .embeddings import (
    MLPEmbedding,
    SinusoidalEmbedding,
    AdaptiveSinusoidalEmbedding,
    TimeEmbedding,
    ConditionalMLPEmbedding,
    ConditionalTemporalConvEmbedding,
    ConditionalTemporalMixerEmbedding,
    HybridConditionalEmbedding,
    CombinedEmbedding,
)

from .normalization import (
    LayerNorm,
    GroupNorm,
    InstanceNorm,
    GraphNorm,
    AdaptiveLayerNorm,
    AdaptiveGroupNorm,
)

# Level 1: PyG dependencies only
from .graph_conv import (
    TAGConvLayer,
    StridedTAGConv,
    ResidualGNNBlock,
    GNN
)

from .pooling import (
    PoolingGNN,
    LearnableGraphPool,
    GraphUnpool,
    GraphMaxPool,
    GraphMaxUnpool,
    NodeSelector,
    NeighborhoodFeaturePooling,
)

# # Level 2: Composite blocks
# from .blocks import (
#     EncoderBlock,
#     DecoderBlock,
#     BottleneckBlock,
# )

# Optional
try:
    from .attention import SelfAttention, CrossAttention
except ImportError:
    pass

__all__ = [
    # Embeddings
    "TimeEmbedding",
    "MLPEmbedding",
    "SinusoidalEmbedding",
    "AdaptiveSinusoidalEmbedding",
    "ConditionalMLPEmbedding",
    "ConditionalTemporalConvEmbedding",
    "ConditionalTemporalMixerEmbedding",
    "HybridConditionalEmbedding",
    "CombinedEmbedding",
    # Normalization
    "LayerNorm",
    "GroupNorm",
    "InstanceNorm",
    "GraphNorm",
    "AdaptiveLayerNorm",
    "AdaptiveGroupNorm",
    # Graph convolutions
    "TAGConvLayer",
    "StridedTAGConv",
    "ResidualGNNBlock",
    "GNN",
    # Pooling
    "PoolingGNN",
    "LearnableGraphPool",
    "GraphUnpool",
    "GraphMaxPool",
    "GraphMaxUnpool",
    "NodeSelector",
    "NeighborhoodFeaturePooling",
    # Attention
    # Blocks
    # "EncoderBlock",
    # "DecoderBlock",
    # "BottleneckBlock",
]
