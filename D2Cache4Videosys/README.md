# D2Cache for VideoSys (Latte & Open-Sora)

This directory contains **VideoSys**-based implementations of D2Cache for **Latte** and **Open-Sora**, including scripts aligned with **VBench** evaluation. For the method, paper, and citation, see the [main project README](../README.md).

## Environment

**Requirements**

- Python ≥ 3.10  
- PyTorch ≥ 1.13 (PyTorch **2.x** recommended)  
- CUDA ≥ 11.6  

We recommend a clean Conda environment:

```shell
conda create -n d2cache python=3.10 -y
conda activate d2cache
```

Install the local package (editable install):

```shell
pip install -e .
```

Follow `requirements.txt` and upstream **VideoSys** / **ColossalAI** notes if you hit dependency issues on your cluster.

## What is included

- `latte_delta2.py` — Latte generation with D2Cache-style caching.  
- `opensora_delta2.py` — Open-Sora generation with the same framework.  
- `vbench/` — Helpers to run VBench metrics on generated videos (see scripts below).

## Qualitative Comparisons

**Latte (≈3.62× speedup)**

![Latte demo 1](../showcases/3.gif)
![Latte demo 2](../showcases/4.gif)

**Open-Sora (≈2.86× speedup)**

![Open-Sora demo 1](../showcases/5.gif)
![Open-Sora demo 2](../showcases/6.gif)

## Evaluation (VBench-oriented workflow)

**1. Generate videos** (from prompts / settings used in your experiments):

```shell
python latte_delta2.py
python opensora_delta2.py
```

**2. Compute VBench scores** (run VBench on the produced folders, then aggregate):

```shell
# Per-metric scores (adjust paths)
python vbench/run_vbench.py --video_path <path_to_videos> --save_path <score_output_dir>

# Aggregate into a final summary
python vbench/cal_vbench.py --score_dir <score_output_dir>
```

Paths `<path_to_videos>` and `<score_output_dir>` should match your local layout. See `vbench/` for any extra configuration expected by those scripts.

## Citation

Please cite the D2Cache paper if you use this code; see the [main README](../README.md#citation).
