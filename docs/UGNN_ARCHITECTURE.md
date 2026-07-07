# UGNN Architecture: Complete U-Net with Encoder-Decoder

## Overview

Complete U-Net style Graph Neural Network for diffusion models with hierarchical multi-scale processing.

```
INPUT: x (B, T, N, F_in)
  │
  ├─── timesteps (B,) ──→ TimeEmbedding ──→ time_emb (B, 128) [shared]
  │
  └─── cond (B, T, N, F_cond) ──→ ConditionalTemporalConvEmbedding ──→ cond_emb (B, N, 64) [shared]
  
  ↓ input_proj: F_in → base_channels * mult[0]
```

---

## Encoder (Contracting Path) - Progressive Downsampling

### LEVEL 0 
**[stride_pre=1, N active nodes, channels=base×mult[0]]**

```
┌─────────────────────────────────────────────────────────────────┐
│ EncoderBlock:                                                    │
│   1. Fuse: [x, time_emb, cond_emb] → fusion_proj               │
│   2. GNN: stride_pre=1 neighborhoods → 0, 1, 2, 3 hops         │
│   3. Pool: stride_post=2 aggregation → 0, 2, 4 hops           │
│           Select N/2 nodes with highest aggregated values       │
│ Output: x_pooled (N/2 active), active_mask_0                    │
└─────────────────────────────────────────────────────────────────┘
  │                                       ↓
  │                                  skip_features[0] ←─────┐
  │                                  active_masks[0]        │
  ↓ [stride_post=2, N/2 active]                            │
```

### LEVEL 1
**[stride_pre=2, N/2 active nodes, channels=base×mult[1]]**

```
┌─────────────────────────────────────────────────────────────────┐
│ EncoderBlock:                                                    │
│   1. Fuse: [x, time_emb, cond_emb] → fusion_proj               │
│   2. GNN: stride_pre=2 neighborhoods → 0, 2, 4, 6 hops         │
│   3. Pool: stride_post=4 aggregation → 0, 4, 8 hops           │
│           Select N/4 nodes with highest aggregated values       │
│ Output: x_pooled (N/4 active), active_mask_1                    │
└─────────────────────────────────────────────────────────────────┘
  │                                       ↓
  │                                  skip_features[1] ←─────┐
  │                                  active_masks[1]        │
  ↓ [stride_post=4, N/4 active]                            │
```

### LEVEL 2
**[stride_pre=4, N/4 active nodes, channels=base×mult[2]]**

```
┌─────────────────────────────────────────────────────────────────┐
│ EncoderBlock:                                                    │
│   1. Fuse: [x, time_emb, cond_emb] → fusion_proj               │
│   2. GNN: stride_pre=4 neighborhoods → 0, 4, 8, 12 hops        │
│   3. Pool: stride_post=8 aggregation → 0, 8, 16 hops          │
│           Select N/8 nodes with highest aggregated values       │
│ Output: x_pooled (N/8 active), active_mask_2                    │
└─────────────────────────────────────────────────────────────────┘
  │                                       ↓
  │                                  skip_features[2] ←─────┐
  │                                  active_masks[2]        │
  ↓ [stride_post=8, N/8 active]                            │
```

---

## Bottleneck
**[stride=8, N/8 active nodes, channels=base×mult[2]]**

```
┌─────────────────────────────────────────────────────────┐
│ Bottleneck GNN (optional, num_bottleneck_layers=1):    │
│   - Process at coarsest resolution                      │
│   - Global context aggregation                          │
│   - stride=8 neighborhoods → 0, 8, 16, 24 hops         │
└─────────────────────────────────────────────────────────┘
  ↓ x_bottleneck (N/8 active)
```

---

## Decoder (Expanding Path) - Progressive Upsampling

### LEVEL 2→1
**[stride_pre=4, channels: base×mult[2] → base×mult[1]]**

```
┌────────────────────────────────────────────────────────┐
│ DecoderBlock:                                          │
│   1. Upsample: Restore N/4 active nodes (zero-fill)   │
│      Newly active nodes initialized to zeros           │
│   2. Skip Fusion: concat[x_upsampled, skip_features[2]]│←──── skip_features[2]
│      → skip_proj                                       │
│   3. Fuse: [x, time_emb, cond_emb] → fusion_proj     │
│   4. GNN: stride_pre=4 neighborhoods → 0, 4, 8, 12 hops│
│ Output: x_decoded (N/4 active)                         │
└────────────────────────────────────────────────────────┘
  ↓ [stride=4, N/4 active]
```

### LEVEL 1→0
**[stride_pre=2, channels: base×mult[1] → base×mult[0]]**

```
┌────────────────────────────────────────────────────────┐
│ DecoderBlock:                                          │
│   1. Upsample: Restore N/2 active nodes (zero-fill)   │
│      Newly active nodes initialized to zeros           │
│   2. Skip Fusion: concat[x_upsampled, skip_features[1]]│←──── skip_features[1]
│      → skip_proj                                       │
│   3. Fuse: [x, time_emb, cond_emb] → fusion_proj     │
│   4. GNN: stride_pre=2 neighborhoods → 0, 2, 4, 6 hops │
│ Output: x_decoded (N/2 active)                         │
└────────────────────────────────────────────────────────┘
  ↓ [stride=2, N/2 active]
```

### LEVEL 0
**[stride_pre=1, channels: base×mult[0] → base×mult[0]]**

```
┌────────────────────────────────────────────────────────┐
│ DecoderBlock:                                          │
│   1. Upsample: Restore N active nodes (zero-fill)     │
│      Newly active nodes initialized to zeros           │
│   2. Skip Fusion: concat[x_upsampled, skip_features[0]]│←──── skip_features[0]
│      → skip_proj                                       │
│   3. Fuse: [x, time_emb, cond_emb] → fusion_proj     │
│   4. GNN: stride_pre=1 neighborhoods → 0, 1, 2, 3 hops │
│ Output: x_decoded (N active)                           │
└────────────────────────────────────────────────────────┘
  ↓ output_proj: base×mult[0] → out_channels
```

```
OUTPUT: (B, T, N, F_out)
```

---

## Key Design Principles

### 1. Graph Structure
- **Fixed**: N nodes, edge_index - NEVER changes
- **Pooling** = masking inactive nodes to zero (sparse representation)
- **Unpooling** = restoring previously masked nodes (zero-filling)

### 2. Stride Tracking
Controls neighborhood scales on the original graph:
- **stride_pre[i]**: Spacing of input signal at level i
- **stride_post[i] = stride_pre[i] × gamma[i]**: Spacing of output signal
- **Nested pooling**: stride_post accumulates through levels

### 3. GNN Neighborhoods
Match input signal scale (stride_pre):
- Level i GNN uses `k×stride_pre[i]` hops for `k=0,1,...,K`
- Captures information at appropriate scale for sparse input signal

### 4. Pooling Neighborhoods
Match output signal scale (stride_post):
- Level i pooling aggregates from `k×stride_post[i]` hops for `k=0,1,...,pool_K`
- Ensures proper multi-scale aggregation for output signal spacing
- **⚠️ CRITICAL**: Uses stride_post (not gamma) for nested pooling principle

### 5. Skip Connections
Preserve fine-grained information:
- Encoder stores features BEFORE pooling (`skip_features[i]`)
- Decoder fuses upsampled features with skip connections
- Fusion modes: `'concat'` (default) or `'add'`

### 6. Embeddings
Time + Conditional (optional):
- Computed ONCE at top level, shared across all encoder/decoder blocks
- Each block projects embeddings to its feature space
- Masked for inactive nodes to prevent information leakage

### 7. Active Masks
Track which nodes are active at each level:
- Encoder progressively reduces active nodes
- Decoder restores encoder's active masks in reverse order
- Perfect symmetry: decoder level i has same active nodes as encoder level i

### 8. Upsampling
Zero-filling (extensible to other methods):
- Newly activated nodes start with zeros
- Skip connections provide rich context for reconstruction
- Decoder GNN propagates information from active neighbors

---

## Example Configuration

**3-level UGNN**: γ=2, N=64 nodes, base_channels=32, K=3, pool_K=2

| Level  | Active Nodes | Stride (pre→post) | Channels | GNN Hops (stride_pre) | Pool Hops (stride_post) |
|--------|--------------|-------------------|----------|----------------------|------------------------|
| Enc-0  | 64→32        | 1→2              | 32       | 0, 1, 2, 3          | 0, 2, 4               |
| Enc-1  | 32→16        | 2→4              | 64       | 0, 2, 4, 6          | 0, 4, 8               |
| Enc-2  | 16→8         | 4→8              | 128      | 0, 4, 8, 12         | 0, 8, 16              |
| **Bottleneck** | 8   | stride=8         | 128      | 0, 8, 16, 24        | -                     |
| Dec-2  | 8→16         | 4                | 128      | 0, 4, 8, 12         | (restore N/4)         |
| Dec-1  | 16→32        | 2                | 64       | 0, 2, 4, 6          | (restore N/2)         |
| Dec-0  | 32→64        | 1                | 32       | 0, 1, 2, 3          | (restore N)           |

**Output**: 64 nodes, F_out channels

---

## Nested Pooling Principle

At each level, pooling aggregation neighborhoods match the OUTPUT signal scale:

- **Level 0**: Pool with stride_post=2 → Aggregates 0, 2, 4 hop info  
  Output has stride=2 spacing → ✓ Matches aggregation scale

- **Level 1**: Pool with stride_post=4 → Aggregates 0, 4, 8 hop info  
  Output has stride=4 spacing → ✓ Matches aggregation scale

- **Level 2**: Pool with stride_post=8 → Aggregates 0, 8, 16 hop info  
  Output has stride=8 spacing → ✓ Matches aggregation scale

**This ensures hierarchical consistency**: the pooling operation aggregates information at the same absolute scale as the output signal it produces.

---

## Information Flow Paths

1. **VERTICAL** (Encoder→Bottleneck→Decoder):  
   Progressive abstraction → global context → reconstruction

2. **HORIZONTAL** (Skip connections):  
   Fine-grained features bypass bottleneck → fused in decoder

3. **TEMPORAL** (Time embeddings):  
   Diffusion timestep information → fused at every level

4. **CONDITIONAL** (Optional):  
   Task-specific context → fused at every level

5. **SPATIAL** (Multi-hop GNN):  
   Neighborhood aggregation at appropriate scales for signal density

6. **HIERARCHICAL** (Multi-level):  
   Each level operates at different graph resolution (stride)

---

## Configuration Parameters

```python
@dataclass
class UGNNConfig:
    in_channels: int                          # Input feature dimension
    out_channels: int                         # Output feature dimension
    base_channels: int = 64                   # Base channel count
    channel_multipliers: List[int] = [1,2,4,8]  # Channel scaling per level
    
    gnn_config: GNNConfig                     # GNN layer configuration
    pooling_config: PoolingConfig             # Pooling configuration
    upsampling_config: UpsamplingConfig       # Upsampling configuration
    embedding_config: EmbeddingConfig         # Embedding configuration
    
    num_bottleneck_layers: int = 1            # Bottleneck GNN layers (0=disable)
    skip_connection_mode: str = 'concat'      # 'concat' or 'add'
```

---

## Usage Example

```python
config = UGNNConfig(
    in_channels=1,
    out_channels=1,
    base_channels=64,
    channel_multipliers=[1, 2, 4, 8],
    num_bottleneck_layers=1,
    gnn_config=GNNConfig(K=3, num_layers=2),
    pooling_config=PoolingConfig(gamma=2, pool_K=2, selection_method='learned'),
)

ugnn = UGNN(config=config)

# Forward pass
x = torch.randn(4, 10, 100, 1)         # (B, T, N, F_in)
timesteps = torch.randint(0, 1000, (4,))
edge_index = torch.randint(0, 400, (2, 500))

output = ugnn(x, timesteps, edge_index)  # (B, T, N, F_out)
```
