# Dataset release tag — content-addressed bundle id

The reproduction dataset for `rigorous-quoll-131` is published as a GitHub Release
in the shared public **`gsd-dataset`** repo under the tag

```
wra-N400-gsd-85faf506ec70
└──── stem ───┘└ bundle id ┘
```

The **stem** (`wra-N400-gsd`) is human context. The **bundle id** (`85faf506ec70`)
is a content fingerprint of *exactly which* dataset the tag points at. This note
defines that id precisely, so anyone can reproduce and verify it — **offline, from
the checked-in manifest, without the 39 GB present**.

## Why a "bundle" id at all

`rigorous-quoll-131` trains on **four** content-addressed sub-datasets (one per
network density). Each already carries its own hash in its directory name:

```
medium-large_outdoor_ultra-low_density/wrpc_v1_primal_history_k200_h0dd7afd393f9
medium-large_outdoor_low_density/      wrpc_v1_primal_history_k200_h43d4a26a4203
medium-large_outdoor_mid_density/      wrpc_v1_primal_history_k200_hc1f8f7a25432
medium-large_outdoor_high_density/     wrpc_v1_primal_history_k200_ha6c7c432ee13
```

Each `_h<hash>` is `md5[:12]` of that sub-dataset's canonicalized build inputs —
scenario/channel generation + primal-dual trainer + sample collection + dataset
build (see [`datasets/wra/channel_factory.py`](../../src/graph_signal_diffusion/datasets/wra/channel_factory.py)).
So the identity is **four** hashes, not one. To tag the *release* (one string) we
fold the four into a single deterministic **bundle id**.

## Definition (canonical, reproducible)

1. Take the four sub-dataset relpaths from
   [`checksums/dataset_manifest.txt`](checksums/dataset_manifest.txt) (field 1,
   `|`-separated).
2. Extract each trailing `_h<hex>` token → the bare lowercase 12-hex string
   (`0dd7afd393f9`, …). This drops the leading `h`.
3. **Sort** the tokens bytewise with `LC_ALL=C`, so the id depends only on the
   *set* of sub-datasets — not on which density is listed "first", nor on the host
   locale.
4. Form the canonical byte stream: the sorted tokens, **one per line**, each
   terminated by `\n` (a trailing newline after the last — exactly `printf '%s\n'`).
5. `bundle_id = md5(stream)` truncated to the **first 12 hex characters** — the
   same `md5[:12]` convention the per-sub-dataset hashes use.

`release_tag = "<stem>-<bundle_id>"`, stem default `wra-N400-gsd`.

### Reproduce it by hand

```bash
printf '%s\n' 0dd7afd393f9 43d4a26a4203 c1f8f7a25432 a6c7c432ee13 \
  | LC_ALL=C sort | md5sum | cut -c1-12
# -> 85faf506ec70
```

### Reproduce / verify with the helper

```bash
reproduce/wra-rigorous-quoll/dataset_bundle_id.sh          # -> 85faf506ec70
reproduce/wra-rigorous-quoll/dataset_bundle_id.sh --tag    # -> wra-N400-gsd-85faf506ec70
reproduce/wra-rigorous-quoll/dataset_bundle_id.sh --check  # asserts == DATASET_RELEASE_TAG
TOKEN_SOURCE=disk reproduce/wra-rigorous-quoll/dataset_bundle_id.sh --check
#   ^ additionally requires the four sub-dataset dirs to be physically present
```

`dataset_bundle_id.sh` derives the tokens from the manifest (offline) or, with
`TOKEN_SOURCE=disk`, additionally checks the dirs exist under `DATASET_ROOT`.

## Design notes

- **Why `md5`, not `sha256`?** Here md5 is a *content fingerprint*, not a security
  primitive. Using `md5[:12]` matches the existing per-sub-dataset dir-hash
  convention, so the whole scheme reads consistently. (Collision risk over four
  fixed inputs is irrelevant.)
- **Why sort?** The four densities have no canonical order; sorting makes the id a
  function of the *set* of sub-datasets, not their listing order.
- **Why derive from the manifest?** `dataset_manifest.txt` is the checked-in,
  verified identity list (`verify_dataset_manifest.sh` checks presence + file count
  + total bytes against it). Deriving the tag from the same source guarantees the
  tag and the integrity check can never disagree about *which* dataset is meant.

## When the dataset is rebuilt

If any sub-dataset is regenerated with different inputs, its `_h<hash>` changes →
the bundle id changes. Recompute and bump the tag in **one place** (`00_config.sh`):

```bash
reproduce/wra-rigorous-quoll/verify_dataset_manifest.sh --regen   # refresh the manifest
NEWID=$(reproduce/wra-rigorous-quoll/dataset_bundle_id.sh)         # recompute the id
# set DATASET_RELEASE_TAG default in 00_config.sh to  ${DATASET_TAG_STEM}-$NEWID
```

`package_dataset.sh` runs `dataset_bundle_id.sh --check` before creating the
release, so it **refuses to publish under a tag that does not match the dataset's
content**.
