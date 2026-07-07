# U-Graph Neural Networks (U-GNN)

The framework's denoiser: a graph-domain adaptation of the U-Net that performs multi-resolution
encoder–decoder processing of graph signals, with pooling realized as a learned node selection rather
than an explicit graph coarsening.

![U-GNN architecture with graph-signal side panels](fig1_architecture_side_panels.png)

The denoiser **ε<sub>θ</sub>(x<sub>k</sub>, k; S, u)** spans *B* resolution levels
N = N<sub>1</sub> ≥ … ≥ N<sub>B</sub>, with B−1 encoder–decoder block pairs and a bottleneck at the
coarsest level. Each block stacks a fusion layer **Π<sub>b</sub>**, which merges the signal path with
global embeddings of the node states **u** and the diffusion step *k*, on a GNN module
**Φ<sub>b</sub>** whose graph convolutions are parametrized by the shared shift operator **S**. On the
encoder path, a selector head **Ψ<sub>b</sub>** scores the active nodes and a **TopK** readout yields
the selection matrix **C<sub>b+1</sub>**; the nested composites
**D<sub>b</sub> = C<sub>b</sub> ⋯ C<sub>1</sub>** fix the active node set at every depth. The matched
decoder reuses each **C<sub>b+1</sub>** transposed for zero-padded up-sampling, and skip connections
carry the encoder features **Z<sub>b</sub>** across. Each GNN layer lifts its node-reduced input to
the full vertex set by zero-padding, filters on **S** with a depth-dependent stride that widens the
hop reach, and reduces back to the active set — so filtering stays convolutional on the original graph
at every resolution. The TopK selection is trained end-to-end with the denoiser via a straight-through
estimator. The side insets trace one representative graph signal through all depths: the encoding
(down-sampling) path on the left and the decoding (up-sampling) path on the right, with hollow markers
denoting pooled-out (inactive) nodes.

![U-GNN block interface (wide)](fig2_wide.png)

A single encoder–decoder block pair at depth *b*. *Encoder (left):* the input
**V<sub>b</sub> = C<sub>b</sub>Z<sub>b−1</sub>** passes through the chain **Π<sub>b</sub> →
Φ<sub>b</sub><sup>E</sup>** to produce **Z<sub>b</sub>**, which continues along the skip and also
feeds the selector **Ψ<sub>b</sub> → TopK**, emitting **C<sub>b+1</sub>**. *Decoder (right):* the
skip-projection **Π<sub>b</sub><sup>skip</sup>** combines the up-sampled coarser output
**C<sub>b+1</sub><sup>⊤</sup>Y<sub>b+1</sub>** with the encoder skip **Z<sub>b</sub>**; the result
passes through **Π<sub>b</sub>** and **Φ<sub>b</sub><sup>D</sup>**. In every block, the global
embeddings **[U<sub>0</sub>; K<sub>0</sub>]**, restricted to the depth-*b* active nodes by
**D<sub>b</sub>**, enter **Π<sub>b</sub>** and **Ψ<sub>b</sub>**, and **D<sub>b</sub>** parametrizes
the GNN modules that convolve over **S**.

The default configuration used in the paper spans B = 4 depths with three pooling levels of factor
ρ = 2 (8× reduction overall), uniform 64-channel widths, and strided TAGConv-based layers, shared
across both applications.

See **[docs/UGNN_ARCHITECTURE.md](../../docs/UGNN_ARCHITECTURE.md)** for the full
architecture reference. Figure sources (TeX/TikZ, caption-off renders of `ugnn_fig1b`
and `ugnn_fig2_wide`) are in **[figure-sources/](figure-sources/)**.
