import argparse
import numpy as np
import open3d as o3d
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
from tqdm import tqdm

def plot_iqr_boxes(RMSE_dict, title="RMSE Results", x_label="RMSE",
                   save_path=None, show_median=True, log_x=False):
    """
    Plot IQR boxes for RMSE results.
    """
    ALG_ORDER = ['LS', 'RANSAC1K', 'RANSAC10K', 'GNC-GM', 'GNC-TLS']

    plot_order = [alg for alg in ALG_ORDER if alg in RMSE_dict]
    
    df_wide = pd.DataFrame({k: RMSE_dict[k] for k in plot_order})

    base_colors = ['blue', 'red', 'green', 'purple', 'brown']
    colors = dict(zip(ALG_ORDER, base_colors))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    BOX_H = 0.55
    EDGE_LW = 3.0
    WHISKER_LW = EDGE_LW * 0.85
    n_alg = len(plot_order)

    for i, alg in enumerate(plot_order):
        y = (n_alg - 1 - i) # 세로 위치 (역순)
        vals = np.asarray(df_wide[alg]).ravel()
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue

        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        q1, q3 = np.percentile(vals, [25, 75])
        median = float(np.median(vals))

        # IQR 박스
        rect = mpatches.Rectangle(
            (q1, y - BOX_H / 2),
            max(q3 - q1, 1e-12),
            BOX_H,
            facecolor='none',
            edgecolor=colors[alg], linewidth=EDGE_LW, joinstyle='miter'
        )
        ax.add_patch(rect)

        # 중앙값
        if show_median:
            ax.plot([median, median], [y - BOX_H / 2, y + BOX_H / 2],
                    color=colors[alg], linewidth=EDGE_LW)

        # 수염 + 캡
        ax.plot([vmin, q1], [y, y], color=colors[alg], linewidth=WHISKER_LW)
        ax.plot([q3, vmax], [y, y], color=colors[alg], linewidth=WHISKER_LW)
        cap_h = BOX_H * 0.6
        ax.plot([vmin, vmin], [y - cap_h / 2, y + cap_h / 2], color=colors[alg], linewidth=WHISKER_LW)
        ax.plot([vmax, vmax], [y - cap_h / 2, y + cap_h / 2], color=colors[alg], linewidth=WHISKER_LW)

    ax.set_yticks([])
    ax.set_ylabel(None)
    ax.set_xlabel(x_label, fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.set_xscale('log')
    ax.set_title(title, fontsize=22, pad=20)

    pad = 0.6
    ax.set_ylim(-pad, n_alg - 1 + pad)

    ax.set_axisbelow(True)
    ax.grid(True, which='major', axis='x', linestyle='-', linewidth=0.8, alpha=0.35)
    ax.grid(True, which='minor', axis='x', linestyle='-', linewidth=0.5, alpha=0.2)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    handles = [
        mpatches.Patch(facecolor=colors[alg], edgecolor=colors[alg],
                       linewidth=EDGE_LW, label=alg, alpha=0.8)
        for alg in plot_order
    ]
    ax.legend(handles=handles, title="Algorithms", frameon=False, fontsize=12,
              bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.0)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # 범례 공간 확보
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_correspondences(
    pcd_src: o3d.geometry.PointCloud,
    pcd_tgt: o3d.geometry.PointCloud,
    P: np.ndarray,
    Q: np.ndarray,
    max_lines: int = 2000,
    src_color=(0.0, 0.7, 1.0),
    tgt_color=(0.0, 0.2, 0.8),
    line_color=(0.2, 0.9, 0.2),
    R_gt=None,
    t_gt=None,
    inlier_threshold=0.02,
    line_width=3.0
):

    if not isinstance(pcd_src, o3d.geometry.PointCloud) or not isinstance(pcd_tgt, o3d.geometry.PointCloud):
        raise TypeError("pcd_src and pcd_tgt must be open3d.geometry.PointCloud")
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.ndim != 2 or Q.ndim != 2 or P.shape[1] != 3 or Q.shape[1] != 3:
        raise ValueError("P and Q must have shape (N, 3)")
    if P.shape[0] != Q.shape[0]:
        raise ValueError(f"P and Q must have the same number of rows, got {P.shape[0]} and {Q.shape[0]}")
    if R_gt is None or t_gt is None:
        raise ValueError("R_gt and t_gt must be provided to classify inliers/outliers")
    R_gt = np.asarray(R_gt, dtype=float).reshape(3, 3)
    t_gt = np.asarray(t_gt, dtype=float).reshape(3,)

    finite_mask = np.isfinite(P).all(axis=1) & np.isfinite(Q).all(axis=1)
    if not finite_mask.all():
        P = P[finite_mask]
        Q = Q[finite_mask]
    N = P.shape[0]
    src = copy.deepcopy(pcd_src)
    tgt = copy.deepcopy(pcd_tgt)
    src.paint_uniform_color(src_color)
    tgt.paint_uniform_color(tgt_color)

    if N == 0:
        o3d.visualization.draw_geometries([src, tgt])
        return src, tgt, None, {"inlier_mask": np.array([], dtype=bool), "errors": np.array([])}

    if N > max_lines:
        idx = np.linspace(0, N - 1, max_lines, dtype=int)
        P_vis, Q_vis = P[idx], Q[idx]
    else:
        P_vis, Q_vis = P, Q
    M = P_vis.shape[0]
    Q_hat = (P_vis @ R_gt.T) + t_gt[None, :]
    errors = np.linalg.norm(Q_vis - Q_hat, axis=1)
    inlier_mask = errors <= float(inlier_threshold)

    points_all = np.vstack([P_vis, Q_vis])
    idx_in = np.nonzero(inlier_mask)[0]
    idx_out = np.nonzero(~inlier_mask)[0]

    def make_lineset(sub_idx, color_rgb):
        if sub_idx.size == 0: return None
        lines = np.column_stack([sub_idx, sub_idx + M]).astype(int)
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_all),
            lines=o3d.utility.Vector2iVector(lines)
        )
        ls.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color_rgb, dtype=float), (lines.shape[0], 1)))
        return ls

    line_set_in = make_lineset(idx_in, (0.2, 0.9, 0.2))
    line_set_out = make_lineset(idx_out, (0.95, 0.2, 0.2))

    all_lines = np.arange(M, dtype=int)
    ls_all = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_all),
        lines=o3d.utility.Vector2iVector(np.column_stack([all_lines, all_lines + M]).astype(int))
    )
    cols = np.zeros((M, 3), dtype=float)
    cols[inlier_mask] = (0.2, 0.9, 0.2)
    cols[~inlier_mask] = (0.95, 0.2, 0.2)
    ls_all.colors = o3d.utility.Vector3dVector(cols)
    
    # ---- draw (prefer the modern renderer to control line width) ----
    drew = False
    try:
        from open3d.visualization.rendering import MaterialRecord
        mat_in = MaterialRecord()
        mat_in.shader = "unlitLine"
        mat_in.line_width = float(line_width)
        mat_out = MaterialRecord()
        mat_out.shader = "unlitLine"
        mat_out.line_width = float(line_width)
        mat_pc = MaterialRecord()
        mat_pc.shader = "defaultUnlit"

        geoms = []
        geoms.append({"name": "src", "geometry": src, "material": mat_pc})
        geoms.append({"name": "tgt", "geometry": tgt, "material": mat_pc})
        if line_set_in is not None:
            geoms.append({"name": "inlier_corr", "geometry": line_set_in, "material": mat_in})
        if line_set_out is not None:
            geoms.append({"name": "outlier_corr", "geometry": line_set_out, "material": mat_out})

        # --- 💡 핵심 수정 사항 ---
        # show_skybox=False 를 추가하여 하늘색 돔 배경을 비활성화
        o3d.visualization.draw(geoms, bg_color=(1.0, 1.0, 1.0, 1.0), show_skybox=False)
        drew = True
    except Exception:
        pass

    if not drew:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1.0, 1.0, 1.0]) # 흰색 배경
        
        vis.add_geometry(src)
        vis.add_geometry(tgt)
        vis.add_geometry(ls_all)
        
        vis.run()
        vis.destroy_window()

    info = {
        "inlier_mask": inlier_mask,
        "errors": errors,
        "num_inliers": int(inlier_mask.sum()),
        "num_outliers": int((~inlier_mask).sum()),
        "threshold": float(inlier_threshold)
    }


def to_pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    """(N,3) numpy -> Open3D PointCloud"""
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("pts must be shape (N, 3)")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64, copy=False))
    return pcd

def _truncated_gaussian_noise(rng, std, beta, size):
    """
    Draw Gaussian noise with std then truncate by Euclidean norm <= beta.
    Vectorized rejection sampling.
    """
    n = size[0]
    out = np.zeros(size, dtype=np.float64)
    remain = np.ones(n, dtype=bool)
    while remain.any():
        k = remain.sum()
        cand = rng.normal(0.0, std, size=(k, 3))
        mask = np.linalg.norm(cand, axis=1) <= beta
        # place accepted
        out[remain][mask] = cand[mask]
        # update remaining
        idx = np.where(remain)[0]
        remain[idx[mask]] = False
    return out

def _uniform_points_in_sphere(rng, n, radius=5.0, center=None):
    """
    Sample n points uniformly inside a 3D sphere of given radius.
    """
    # directions ~ N(0,1), normalize
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    # radii ~ U(0,1)^(1/3) * R  (for uniform volume)
    r = rng.random(n) ** (1.0 / 3.0)
    pts = dirs * (r[:, None] * radius)
    if center is not None:
        pts = pts + center[None, :]
    return pts

def gen_corr(pcd_src, pcd_tgt, noise_std, outlier_ratio, rng,
             M=3000, sphere_radius=3.0, sphere_center="tgt_mean"):
    """
    Generate correspondences (src_corr, tgt_corr) following TEASER-style setup (no scale).
    - Inliers:    tgt = true_tgt + truncated Gaussian noise (‖ε‖ <= β), src = true_src (no noise)
    - Outliers:   replace a fraction of tgt points by random points uniformly sampled inside a sphere
                  (src stays as is). This breaks any rigid relation.

    Args:
        pcd_src: (N,3) source points
        pcd_tgt: (N,3) target points (paired with pcd_src under unknown R,t)
        noise_std: σ for Gaussian noise on inliers (applied to target only)
        outlier_ratio: fraction in [0,1]
        rng: numpy Generator
        M: number of correspondences to sample
        sphere_radius: radius of the outlier sampling sphere (default 5.0 as in paper)
        sphere_center: "tgt_mean" | np.ndarray(3,) | None
                       - "tgt_mean": center the sphere at mean of pcd_tgt
                       - None: center at origin
                       - array: explicit center

    Returns:
        src_corr: (M,3)
        tgt_corr: (M,3)
        inlier_mask: (M,) boolean mask (True for inliers)
    """
    N = pcd_src.shape[0]
    M = min(M, N)
    idx = rng.choice(N, size=M, replace=False)

    src_corr = pcd_src[idx].astype(np.float64, copy=True)  # no noise on source (inliers)
    tgt_corr = pcd_tgt[idx].astype(np.float64, copy=True)  # will add noise / replace for outliers

    # choose outliers
    n_outlier = int(round(M * float(outlier_ratio)))
    if n_outlier > 0:
        outlier_idx = rng.choice(M, size=n_outlier, replace=False)
    else:
        outlier_idx = np.array([], dtype=int)

    # inlier mask
    inlier_mask = np.ones(M, dtype=bool)
    inlier_mask[outlier_idx] = False

    # ----- Inlier noise (truncate by norm <= beta) -----
    # Paper used beta ≈ 5.54 * sigma (for chi^2_3 0.999999 quantile). Use same factor.
    beta = 5.54 * float(noise_std)
    if inlier_mask.any() and noise_std > 0:
        eps = _truncated_gaussian_noise(
            rng, std=float(noise_std), beta=float(beta), size=(inlier_mask.sum(), 3)
        )
        tgt_corr[inlier_mask] += eps

    # ----- Outliers: replace target points by uniform samples in a sphere -----
    if n_outlier > 0:
        if isinstance(sphere_center, str) and sphere_center == "tgt_mean":
            center = pcd_tgt.mean(axis=0)
        elif sphere_center is None:
            center = np.zeros(3, dtype=np.float64)
        else:
            center = np.asarray(sphere_center, dtype=np.float64)
        repl = _uniform_points_in_sphere(
            rng, n_outlier, radius=float(sphere_radius), center=center
        )
        tgt_corr[outlier_idx] = repl

    return src_corr, tgt_corr, inlier_mask

def load_bunny_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    return pcd

def vis_reg(pcd_src, pcd_tgt, R, t):
    pcd_src_transformed = o3d.geometry.PointCloud(pcd_src)
    pts = np.asarray(pcd_src.points)
    pts_transformed = (R @ pts.T).T + t
    pcd_src_transformed.points = o3d.utility.Vector3dVector(pts_transformed)
    pcd_src_transformed.paint_uniform_color([0.3, 0.7, 1.0])  # 연한 하늘색 (Light Sky Blue)
    pcd_tgt.paint_uniform_color([0.0, 0.2, 0.8])              # 진한 파란색 (Deep Blue)
    o3d.visualization.draw_geometries([pcd_src_transformed, pcd_tgt])

def horns_method(P, Q):
    """
    Horn's method for absolute orientation using unit quaternions
    Input:
        P: correspondence from source point cloud (np.ndarray (N,3))
        Q: correspondence from target point cloud (np.ndarray (N,3))
    Output:
        R: (3,3) rotation matrix
        t: (3,) translation vector    
    """

    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    H = P_centered.T @ Q_centered  # (3, 3)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_Q - R @ centroid_P

    return R, t

def ransac_horns(P, Q, max_iterations=1000, distance_threshold=0.01 , rng=None):
    
    best_inliers = []
    best_R = np.eye(3)
    best_t = np.zeros(3)

    n_corr = P.shape[0]

    for it in range(max_iterations):
        sample_indices = rng.choice(n_corr, 3, replace=False)
        R, t = horns_method(P[sample_indices], Q[sample_indices])
        P_transformed = (R @ P.T).T + t
        dists = np.linalg.norm(P_transformed - Q, axis=1) # [K,]
        inliers = np.where(dists < distance_threshold)[0] # [M,]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R = R
            best_t = t
    
    if len(best_inliers) >= 3:
        P = P[best_inliers]
        Q = Q[best_inliers]
        best_R, best_t = horns_method(P, Q)

    return best_R, best_t

def weighted_horns(P, Q, weights):
    """
    Weighted Horn's method for absolute orientation using unit quaternions
    Input:
        P: source point cloud (np.ndarray (N,3))
        Q: target point cloud (np.ndarray (N,3))
        corr   : (K, 2) array of correspondences, each row is [src_idx, tgt_idx]
        weights: (K,) array of weights for each correspondence
    Output:
        R: (3,3) rotation matrix
        t: (3,) translation vector    
    """

    w = weights.reshape(-1, 1) # (K, 1)

    sum_w = np.sum(w)
    if sum_w == 0:
        raise ValueError("Sum of weights must be greater than zero")

    centroid_P = np.sum(w * P, axis=0) / sum_w
    centroid_Q = np.sum(w * Q, axis=0) / sum_w
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    H = P_centered.T @ (w * Q_centered)  # (3, 3)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_Q - R @ centroid_P

    return R, t

def gnc_gm_solver(P, Q, noise_bound: float = 0.01):
    
    """
    GNC-Geman-McClure solver
    Input:
        P: (N,3) source points
        Q: (N,3) target points
        corr: (K,2) correspondences
        noise_bound: inlier noise bound
    Output:
        R: (3,3) rotation matrix
        t: (3,) translation vector
    """

    c_bar = float(noise_bound)
    c2 = c_bar * c_bar
    
    # mu initialization
    R_init, t_init = horns_method(P, Q) # [3,3], [3,1]
    r = Q - (R_init @ P.T).T - t_init # [K, 3]
    r2 = np.sum(r**2, axis=1) # [K,]
    mu = 2.0 * float(np.max(r2)) / c2   # initial mu

    # outer loop
    while True:
        # inner loop

        # weight update
        den = r2 + mu * c2 # [K,]
        w = (mu * c2 / den) ** 2 # [K,]
        
        # variables update 
        R ,t = weighted_horns(P, Q, w) # [3,3], [3,1]
        r = Q - (R @ P.T).T - t # [K, 3]
        r2 = np.sum(r**2, axis=1) # [K,]

        mu = mu / 1.1
        if mu < 1.0 : 
            return R, t

def gnc_tls_solver(P, Q, noise_bound: float = 0.01):
    """
    GNC-Truncated Least Squares solver
    Input:
        P: (N,3) source points
        Q: (N,3) target points
        corr: (K,2) correspondences
        noise_bound: inlier noise bound
    Output:
        R: (3,3) rotation matrix
        t: (3,) translation vector
    """


    c_bar = float(noise_bound)
    c2 = c_bar * c_bar

    # mu initalization
    R_init, t_init = horns_method(P, Q) # [3,3], [3,1]
    r = Q - (R_init @ P.T).T - t_init # [K, 3]
    r2 = np.sum(r**2, axis=1) # [K,]
    r2max = float(np.max(r2)) 
    mu = c2 / (2.0 * r2max - c2) if (2.0 * r2max - c2) != 0 else 1.0   # initial mu

    R_old, t_old = np.eye(3), np.zeros(3) # [3,3], [3,]

    # outer loop
    while True:
    # inner loop
        # weight update
        th1 = (mu / (mu + 1.0)) * c2
        th2 = ((mu + 1.0) / mu) * c2
        mid = (c_bar / (np.sqrt(r2) + 1e-9)) * np.sqrt(mu * (mu + 1.0)) - mu
        w = np.where(r2 < th1, 1.0, np.where(r2 > th2, 0.0, mid))
        # variables update
        R ,t = weighted_horns(P, Q, w) # [3,3], [3,1]
        r = Q - (R @ P.T).T - t # [K, 3]
        r2 = np.sum(r**2, axis=1) # [K,]
        mu = mu * 1.1
        
        if np.linalg.norm(R - R_old) < 1e-6 and np.linalg.norm(t - t_old) < 1e-6:
            return R, t

        R_old, t_old = R, t
        

def main():
    
    RMSE_trans_horn = []
    RMSE_trans_ransac_1k = []
    RMSE_trans_ransac_10k = []
    RMSE_trans_gnc_gm = []
    RMSE_trans_gnc_tls = []

    RMSE_rot_horn = []
    RMSE_rot_ransac_1k = []
    RMSE_rot_ransac_10k = []
    RMSE_rot_gnc_gm = []
    RMSE_rot_gnc_tls = []


    # parse args
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=str, required=True, help="Path to bunny ply")
    ap.add_argument("--noise_std", type=float, default=0.01, help="Gaussian noise stddev")
    ap.add_argument("--outlier_ratio", type=float, default=0.99, help="Outlier ratio")
    ap.add_argument("--voxel", type=float, default=0.03, help="Voxel size for downsampling")
    ap.add_argument("--visualize", action="store_true", help="Visualize point clouds")
    args = ap.parse_args()
    
    # load bunny and downsample
    bunny = load_bunny_ply(args.ply)
    center = bunny.get_center()
    bunny.translate(-center)
    points = np.asarray(bunny.points) # [N, 3]
    max_dist = np.max(np.linalg.norm(points, axis=1)) # [1,]
    bunny.scale(1.0 / max_dist, center=(0,0,0)) # [N, 3]
    bunny = bunny.voxel_down_sample(args.voxel)

    # to numpy 
    pts = np.asarray(bunny.points, dtype=np.float64)
    P = pts.copy() # [N, 3]

    # define transformation range
    ROTATION_RANGE_DEG = [-180, 180]
    ROTATION_RANGE_RAD = [r * np.pi / 180.0 for r in ROTATION_RANGE_DEG]
    TRANSLATION_RANGE = [-3, 3]

    # bound for inlier noise
    beta = 5.54 * float(args.noise_std)

    for mc_run in tqdm(range(30)):

        rng = np.random.default_rng(mc_run)
        Q = pts.copy() # [N, 3]

        # Random transform

        random_euler_angles = rng.uniform(low=ROTATION_RANGE_RAD[0], high=ROTATION_RANGE_RAD[1], size=3)
        R_gt = o3d.geometry.get_rotation_matrix_from_xyz(random_euler_angles) # [3,3]
        t_gt = rng.uniform(low=TRANSLATION_RANGE[0], high=TRANSLATION_RANGE[1], size=3)

        Q = (R_gt @ Q.T).T + t_gt # [N, 3]

        # to Open3D PointCloud

        pcd_src = to_pcd(P)
        pcd_tgt = to_pcd(Q)

        # Generate correspondence from the ground truth correspondence
        
        src_corr , tgt_corr , _ = gen_corr(P, Q, noise_std=args.noise_std, outlier_ratio=args.outlier_ratio, rng=rng)

        if args.visualize:
            visualize_correspondences(pcd_src, pcd_tgt, src_corr, tgt_corr, max_lines=100, R_gt=R_gt, t_gt=t_gt)

        # Horn's method
        R, t = horns_method(src_corr, tgt_corr) # [3,3], [3,1]

        t_err = np.linalg.norm(t - t_gt)
        R_err = np.arccos((np.trace(R_gt.T @ R) - 1) / 2.0)
        R_err = np.degrees(R_err)
        if t_err < 1e-6 :
            t_err = 1e-6
        if R_err < 1e-6 :
            R_err = 1e-6
        RMSE_trans_horn.append(t_err)
        RMSE_rot_horn.append(R_err)

        if args.visualize:
            vis_reg(pcd_src, pcd_tgt, R, t)

        # minimal solver: RANSAC 1K
        R, t = ransac_horns(src_corr, tgt_corr, max_iterations=1000, distance_threshold=beta, rng=rng)

        t_err = np.linalg.norm(t - t_gt)
        R_err = np.arccos((np.trace(R_gt.T @ R) - 1) / 2.0)
        R_err = np.degrees(R_err)
        if t_err < 1e-6 :
            t_err = 1e-6
        if R_err < 1e-6 :
            R_err = 1e-6

        RMSE_trans_ransac_1k.append(t_err)
        RMSE_rot_ransac_1k.append(R_err)

        if args.visualize:
            vis_reg(pcd_src, pcd_tgt, R, t)

        # minimal solver: RANSAC 10K
        R, t = ransac_horns(src_corr, tgt_corr, max_iterations=10000, distance_threshold=beta, rng=rng)

        t_err = np.linalg.norm(t - t_gt)
        R_err = np.arccos((np.trace(R_gt.T @ R) - 1) / 2.0)
        R_err = np.degrees(R_err)
        if t_err < 1e-6 :
            t_err = 1e-6
        if R_err < 1e-6 :
            R_err = 1e-6

        RMSE_trans_ransac_10k.append(t_err)
        RMSE_rot_ransac_10k.append(R_err)

        if args.visualize:
            vis_reg(pcd_src, pcd_tgt, R, t)

        # non minimal solver: GNC-GM
        R, t = gnc_gm_solver(src_corr, tgt_corr, noise_bound=beta)

        t_err = np.linalg.norm(t - t_gt)
        R_err = np.arccos((np.trace(R_gt.T @ R) - 1) / 2.0)
        R_err = np.degrees(R_err)
        if t_err < 1e-6 :
            t_err = 1e-6
        if R_err < 1e-6 :
            R_err = 1e-6
        RMSE_trans_gnc_gm.append(t_err)
        RMSE_rot_gnc_gm.append(R_err)

        if args.visualize:
            vis_reg(pcd_src, pcd_tgt, R, t)

        # non minimal solver: GNC-TLS
        R, t = gnc_tls_solver(src_corr, tgt_corr, noise_bound=beta)

        t_err = np.linalg.norm(t - t_gt)
        R_err = np.arccos((np.trace(R_gt.T @ R) - 1) / 2.0)
        R_err = np.degrees(R_err)
        if t_err < 1e-6 :
            t_err = 1e-6
        if R_err < 1e-6 :
            R_err = 1e-6

        RMSE_trans_gnc_tls.append(t_err)
        RMSE_rot_gnc_tls.append(R_err)

        if args.visualize:
            vis_reg(pcd_src, pcd_tgt, R, t)


    RMSE_trans_dict = {
        'LS': RMSE_trans_horn,
        'RANSAC1K': RMSE_trans_ransac_1k,
        'RANSAC10K': RMSE_trans_ransac_10k,
        'GNC-GM': RMSE_trans_gnc_gm,
        'GNC-TLS': RMSE_trans_gnc_tls
    }
    plot_iqr_boxes(RMSE_trans_dict, 
                   title="Translation Error Distribution", 
                   x_label="RMSE (meters)",
                   save_path="translation_rmse.png")

    RMSE_rot_dict = {
        'LS': np.rad2deg(RMSE_rot_horn),
        'RANSAC1K': np.rad2deg(RMSE_rot_ransac_1k),
        'RANSAC10K': np.rad2deg(RMSE_rot_ransac_10k),
        'GNC-GM': np.rad2deg(RMSE_rot_gnc_gm),
        'GNC-TLS': np.rad2deg(RMSE_rot_gnc_tls)
    }
    plot_iqr_boxes(RMSE_rot_dict, 
                   title="Rotation Error Distribution", 
                   x_label="RMSE (degrees)",
                   save_path="rotation_rmse.png")


if __name__ == "__main__":
    main()
