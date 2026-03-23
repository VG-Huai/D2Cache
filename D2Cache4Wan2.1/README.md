# D2Cache for Wan2.1

This folder provides a minimal integration of **D2Cache** with the **Wan2.1** text-to-video diffusion model. For the method, benchmarks, and citation, see the [main project README](../README.md).

## Overview

Under comparable acceleration, D2Cache improves visual quality relative to strong first-order caching (e.g., TeaCache). The qualitative examples in [`../showcases/`](../showcases/) include Wan2.1 at roughly **3.63×** speedup in our experiments.

<video src="../showcases/1.mp4" controls></video>
<video src="../showcases/2.mp4" controls></video>

## Prerequisites

1. Clone and install **[Wan2.1](https://github.com/Wan-Video/Wan2.1)** following its official instructions (dependencies, checkpoints, and directory layout).
2. Copy `delta2cache_generate.py` from this folder into your local Wan2.1 repository so it sits alongside the project’s usual entry scripts.

## Example: text-to-video (1.3B)

Example command for T2V with the **1.3B** model (adjust paths to match your checkpoint location):

```bash
python delta2cache_generate.py \
  --task t2v-1.3B \
  --size 832*480 \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --base_seed 42 \
  --teacache_thresh 0.24 \
  --teacache_mode delta_delta
```

The flag `--teacache_mode delta_delta` selects the D2Cache second-order caching path as described in the paper.

## Citation

Please cite the D2Cache paper if you use this code; see the [main README](../README.md#citation).
