> **Fork note (awschult002):** This repository extends [fastgs/FastGS](https://github.com/fastgs/FastGS) with an experimental **COLMAP-free camera solve** path inspired by [CF-3DGS](https://arxiv.org/abs/2312.07504) (NVIDIA / UCSD / Berkeley). See [docs/CF3DGS_INTEGRATION.md](docs/CF3DGS_INTEGRATION.md) and `train_colmap_free.py`. Upstream FastGS COLMAP training is unchanged.

<div align="center">
<h1>FastGS: Training 3D Gaussian Splatting in 100 Seconds</h1>
<h2>CVPR 2026</h2>

[Homepage](https://fastgs.github.io/) | [Paper](https://arxiv.org/abs/2511.04283) | [Pre-trained model](https://huggingface.co/Goodsleepeverday/fastgs)

</div>

## COLMAP-free mode (CF-3DGS-inspired, this fork)

Skip COLMAP by solving cameras progressively on an ordered image sequence, export a COLMAP-compatible `sparse/0`, then run normal FastGS training:

```bash
python train_colmap_free.py -s /path/to/seq --export_dir /path/to/seq_solved
python train.py -s /path/to/seq_solved -m output/seq_run
```

Design, limitations, and license notes: [docs/CF3DGS_INTEGRATION.md](docs/CF3DGS_INTEGRATION.md). Smoke test: `python -m unittest tests.test_cf3dgs_bridge_smoke`.

Please cite CF-3DGS when using this path:

```
@InProceedings{Fu_2024_CVPR,
  author    = {Fu, Yang and Liu, Sifei and Kulkarni, Amey and Kautz, Jan and Efros, Alexei A. and Wang, Xiaolong},
  title     = {COLMAP-Free 3D Gaussian Splatting},
  booktitle = {CVPR},
  year      = {2024}
}
```

## What Makes FastGS Special?

FastGS is a **general acceleration framework** that supercharges 3D Gaussian Splatting training while maintaining comparable rendering quality:

- Blazing fast training (~100 seconds)
- Easy integration with various 3DGS backbones
- Multi-task ready (dynamic, surface, sparse-view, large-scale, SLAM)

## Quick Start

```bash
git clone https://github.com/awschult002/FastGS.git --recursive
cd FastGS
conda env create --file environment.yml
conda activate fastgs
```

Classic COLMAP training (unchanged):

```bash
bash train_base.sh
# or
python train.py -s <colmap_scene> -m <output>
```

See upstream docs for dataset layout (Mip-NeRF 360, Tanks & Temples, Deep Blending).

## Acknowledgements

Built upon 3DGS, Taming-3DGS, Speedy-Splat, and Abs-GS. COLMAP-free camera solving techniques are inspired by [CF-3DGS](https://github.com/NVlabs/CF-3DGS). Do **not** copy NVlabs CF-3DGS source verbatim (see their LICENSE).

## Citation

```
@article{ren2025fastgs,
  title={FastGS: Training 3D Gaussian Splatting in 100 Seconds},
  author={Ren, Shiwei and Wen, Tianci and Fang, Yongchun and Lu, Biao},
  journal={arXiv preprint arXiv:2511.04283},
  year={2025}
}
```
