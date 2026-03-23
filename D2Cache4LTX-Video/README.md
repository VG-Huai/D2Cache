# D2Cache for LTX-Video

This folder integrates **D2Cache** with **LTX-Video** and provides scripts for large-scale **VBench**-style generation and evaluation. For the method, paper, and citation, see the [main project README](../README.md).

## Installation

1. **Install LTX-Video** following the official instructions: [https://github.com/Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video)

2. Copy the D2Cache scripts from this folder into your LTX-Video workspace.

## Usage

**Single-GPU VBench sweep**

The script below runs automated generation over VBench-style settings (the default configuration may produce on the order of **4730** videos across settings; confirm disk space and runtime before launching):

```bash
python vbench_gen_d2cache_ltx.py
```

For TeaCache-only or ablation baselines, see companion scripts in this directory (e.g., `vbench_gen_teacache_ltx.py`) if present.

Tune batch size, resolution, and output directories inside the Python files to match your hardware and evaluation protocol.

## Citation

Please cite the D2Cache paper if you use this code; see the [main README](../README.md#citation).
