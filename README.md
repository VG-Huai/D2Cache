# D2Cache: Second-Order Delta Caching for Higher Video Diffusion Acceleration

Official PyTorch reference implementations for our **CVPR 2026** paper.

**Authors:** [Enhuai Liu](mailto:eliu0719@sydney.edu.au), [Yunke Wang](mailto:yunke.wang@sydney.edu.au), [Changming Sun](mailto:Changming.Sun@data61.csiro.au), [Chang Xu](mailto:c.xu@sydney.edu.au) (corresponding author)  
**Affiliations:** The University of Sydney · CSIRO Data61

---

## Abstract

Video diffusion models are accurate but expensive at inference time because denoising is applied sequentially over many timesteps. Caching methods reuse computations across timesteps and often extrapolate from **first-order residuals** (differences between adjacent predictions). We propose **D2Cache**, a **training-free** plug-in that exploits the smoothness of **second-order residual deltas**—temporal differences between consecutive first-order residuals—to predict skipped steps more accurately and curb error accumulation. We further **adaptively scale** second-order terms using signals from timestep embeddings. Across several video diffusion backbones, D2Cache improves quality–latency trade-offs compared with strong first-order caching baselines (e.g., TeaCache), with larger gains under aggressive acceleration.

A camera-ready PDF is included in this repository: [`d2cache.pdf`](./d2cache.pdf).

---

## Contributions

- **Second-order delta caching** for diffusion inference: reuse and correct predictions using second-order residual structure, beyond standard first-order residual caching.
- **Theory:** analysis of delta caching in diffusion models and guarantees motivating second-order correction.
- **Empirical evaluation** on multiple video diffusion models and benchmarks, showing consistent gains especially under high speedup.

---

## Repository layout

| Path | Description |
|------|-------------|
| [`D2Cache4Wan2.1/`](./D2Cache4Wan2.1/) | Integration with **Wan2.1** (text-to-video); drop-in generation script. |
| [`D2Cache4Videosys/`](./D2Cache4Videosys/) | **VideoSys**-based implementations for **Latte** and **Open-Sora**, plus VBench-oriented evaluation scripts. |
| [`D2Cache4LTX-Video/`](./D2Cache4LTX-Video/) | Integration with **LTX-Video** and large-scale VBench generation. |
| [`showcases/`](./showcases/) | Qualitative comparison videos (default vs. TeaCache-superfast vs. D2Cache-superfast). |

Each subdirectory contains its own `README.md` with environment setup and commands.

---

## Qualitative comparisons

The videos below compare **default inference**, **TeaCache (superfast)**, and **D2Cache (superfast)** under matched caching schedules. Reported speedups are from our paper’s experimental setup; see the paper for full settings and metrics.

### Wan2.1 (≈3.63× speedup)

<video src="./showcases/1.mp4" controls></video>
<video src="./showcases/2.mp4" controls></video>

### Latte (≈3.62× speedup)

<video src="./showcases/3.mp4" controls></video>
<video src="./showcases/4.mp4" controls></video>

### Open-Sora (≈2.86× speedup)

<video src="./showcases/5.mp4" controls></video>
<video src="./showcases/6.mp4" controls></video>

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{liu2026d2cache,
  title     = {D2Cache: Second-Order Delta Caching for Higher Video Diffusion Acceleration},
  author    = {Liu, Enhuai and Wang, Yunke and Sun, Changming and Xu, Chang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

*(Update `pages` / `doi` / `url` in the BibTeX entry once the official proceedings metadata is available.)*

---

## Acknowledgements

This work was supported in part by the Australian Research Council under Projects **DP240101848** and **FT230100549**.

---

## Contact

Questions about the code or paper: open an issue in this repository or email the authors (see addresses above).
