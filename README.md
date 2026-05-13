# Simple GNC for Point Set Registration

<p align="center">
  <img src="misc/gnc_tls_path_enhanced.gif" alt="GNC-TLS optimization path" width="640">
</p>

<p align="center">
  Heavy outliers, noisy correspondences, and a compact registration benchmark in one self-contained tutorial.
  <br>
  This repository compares Least Squares, RANSAC, GNC-GM, and GNC-TLS on classic Stanford 3D scan data.
</p>

## Overview

This repository is a small but complete tutorial on outlier-robust point cloud registration.
The main script, `simple_gnc_pcr.py`, takes a `.ply` point cloud, applies a random rigid transform, injects inlier noise and extreme outliers, and then compares several pose estimators under the same synthetic correspondence set.

The goal is simple:

- show how quickly naive least squares breaks under heavy outlier corruption
- compare minimal solvers such as RANSAC against non-minimal robust estimators
- visualize correspondences and registration results in a way that is easy to explain in a talk, class, or GitHub project page
- keep the code path short enough that the full experiment can be read in one sitting

## Highlights

- Single-file experiment pipeline in `simple_gnc_pcr.py`
- Random SE(3) perturbation plus Monte Carlo evaluation over 30 runs
- Truncated Gaussian inlier noise and uniformly sampled spherical outliers
- Side-by-side comparison of `LS`, `RANSAC1K`, `RANSAC10K`, `GNC-GM`, and `GNC-TLS`
- Ready-to-use Stanford Bunny, Armadillo, Dragon, and Happy Buddha assets included in the repository

## Companion Reading

If you want the theory behind the code, I also wrote a short blog series on GNC and robust estimation:

- [Graduated Non-Convexity (1)](https://tars0523.github.io/posts/GNC/): motivation for robust estimation, why outliers break plain least squares, and why non-convex robust costs create local minimum issues
- [Graduated Non-Convexity (2)](https://tars0523.github.io/posts/GNC-2/): a walkthrough of Black-Rangarajan duality and the weighted least-squares view of robust kernels
- [Graduated Non-Convexity (3)](https://tars0523.github.io/posts/GNC-3/): the actual GNC algorithm, `mu` continuation, and how GNC-GM is turned into a practical solver

The posts are a more theory-first companion to this repository, while the code here is meant to stay compact and experiment-focused.

## Polynomial Toy Example

<table>
  <tr>
    <th align="center" width="20%">GT</th>
    <th align="center" width="20%">LS</th>
    <th align="center" width="20%">RANSAC-1K</th>
    <th align="center" width="20%">RANSAC-10K</th>
    <th align="center" width="20%">GNC-TLS</th>
  </tr>
  <tr>
    <td><img src="misc/poly_gt.png" alt="Polynomial ground truth" width="100%"></td>
    <td><img src="misc/poly_ls.png" alt="Polynomial least squares" width="100%"></td>
    <td><img src="misc/poly_rs_1k.png" alt="Polynomial RANSAC 1K" width="100%"></td>
    <td><img src="misc/poly_rs_10k.png" alt="Polynomial RANSAC 10K" width="100%"></td>
    <td><img src="misc/poly_tls_gnc.png" alt="Polynomial GNC-TLS" width="100%"></td>
  </tr>
</table>

## Result Gallery

<table>
  <tr>
    <th align="center" width="25%">Corr</th>
    <th align="center" width="25%">RANSAC</th>
    <th align="center" width="25%">GNC-GM</th>
    <th align="center" width="25%">GNC-TLS</th>
  </tr>
  <tr>
    <td align="center"><strong>Armadillo</strong><br><img src="misc/arma_corr.png" alt="Armadillo correspondences" width="100%"></td>
    <td><img src="misc/arma_ransac.png" alt="Armadillo RANSAC" width="100%"></td>
    <td><img src="misc/arma_gnc_gm.png" alt="Armadillo GNC-GM" width="100%"></td>
    <td><img src="misc/arma_gnc_tls.png" alt="Armadillo GNC-TLS" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><strong>Bunny</strong><br><img src="misc/bunny_corr.png" alt="Bunny correspondences" width="100%"></td>
    <td><img src="misc/bunny_ransac.png" alt="Bunny RANSAC" width="100%"></td>
    <td><img src="misc/bunny_gnc_gm.png" alt="Bunny GNC-GM" width="100%"></td>
    <td><img src="misc/bunny_gnc_tls.png" alt="Bunny GNC-TLS" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><strong>Dragon</strong><br><img src="misc/dragon_corr.png" alt="Dragon correspondences" width="100%"></td>
    <td><img src="misc/dragon_ransac.png" alt="Dragon RANSAC" width="100%"></td>
    <td><img src="misc/dragon_gnc_gm.png" alt="Dragon GNC-GM" width="100%"></td>
    <td><img src="misc/dragon_gnc_tls.png" alt="Dragon GNC-TLS" width="100%"></td>
  </tr>
</table>

The `misc/` folder also contains the animated GNC-TLS trajectory and additional presentation-style figures.

## What The Script Does

`simple_gnc_pcr.py` follows the pipeline below:

1. Load a `.ply` point cloud and normalize it to a consistent scale.
2. Sample a random rotation and translation to create a target point cloud.
3. Generate synthetic correspondences from ground-truth pairs.
4. Add truncated Gaussian noise to inliers.
5. Replace a large fraction of target correspondences with uniformly sampled outliers inside a sphere.
6. Estimate the rigid transform with multiple solvers.
7. Measure translation and rotation error over repeated Monte Carlo trials.
8. Save box-style RMSE summaries as `translation_rmse.png` and `rotation_rmse.png`.

## Methods Compared

- `LS`: vanilla least-squares alignment using Horn's method
- `RANSAC1K`: minimal-set RANSAC with 1,000 iterations
- `RANSAC10K`: minimal-set RANSAC with 10,000 iterations
- `GNC-GM`: Graduated Non-Convexity with the Geman-McClure penalty
- `GNC-TLS`: Graduated Non-Convexity with a truncated least-squares penalty

## Quick Start

Install the Python dependencies:

```bash
pip install numpy open3d pandas seaborn matplotlib tqdm
```

Run the tutorial on one of the bundled point clouds:

```bash
python simple_gnc_pcr.py \
  --ply bunny/reconstruction/bun_zipper_res4.ply \
  --noise_std 0.01 \
  --outlier_ratio 0.99 \
  --voxel 0.03
```

You can also try other included assets:

```bash
python simple_gnc_pcr.py --ply Armadillo_scans/Armadillo.ply --noise_std 0.01 --outlier_ratio 0.95 --voxel 0.03 --visualize
python simple_gnc_pcr.py --ply dragon_recon/dragon_vrip_res4.ply --noise_std 0.01 --outlier_ratio 0.99 --voxel 0.03
python simple_gnc_pcr.py --ply happy_recon/happy_vrip_res4.ply --noise_std 0.01 --outlier_ratio 0.99 --voxel 0.03
```

## Command-Line Options

| Option | Description | Default |
| --- | --- | --- |
| `--ply` | Path to the input `.ply` point cloud | required |
| `--noise_std` | Standard deviation of Gaussian inlier noise | `0.01` |
| `--outlier_ratio` | Fraction of correspondences replaced by outliers | `0.99` |
| `--voxel` | Voxel size used for downsampling | `0.03` |
| `--visualize` | Open Open3D windows for correspondences and registration results | off |
