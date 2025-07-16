#!/usr/bin/env python3
import subprocess
import json
import random
import numpy as np
import pandas as pd
from scipy.stats import multivariate_t
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os


def get_app_locations():
    try:
        applescript = '''
        tell application "Finder"
            get {name, desktop position} of every item of desktop
        end tell
        '''
        result = subprocess.run(['osascript', '-e', applescript], capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        
        if not output:
            return []

        parts = output.split(', ')
        
        names = []
        coords_flat = []
        
        for part in parts:
            try:
                coords_flat.append(int(part))
            except ValueError:
                names.append(part)
        
        locations = []
        for i in range(len(coords_flat) // 2):
            locations.append((coords_flat[i*2], coords_flat[i*2+1]))
        return locations
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Could not get app locations: {e}")
        return []



def choose_apps(df, n_neuron_cluster):
    # select random 10 nodes
    all_indices = list(range(len(df)))
    selected_indices = random.sample(all_indices, 10)
    known_points = [df[i] for i in selected_indices]
    
    # 2) 残りから (start, goal) ペアを作る
    available = list(set(all_indices) - set(selected_indices))
    start_coord = []
    goal_coord  = []

    while len(start_coord) < n_neuron_cluster and len(available) >= 2:
        # -- start を選ぶ ---------------------------------------------------
        start = random.choice(available)
        available.remove(start)
        s_coord = np.asarray(df[start])

        # -- 距離をまとめて計算 (ベクトル化) --------------------------------
        rem_arr  = np.asarray([df[i] for i in available])         # shape (M, D)
        dists    = np.linalg.norm(rem_arr - s_coord, axis=1)      # shape (M,)
        median_v = np.median(dists)                               # 距離値の中央値
        

        # 中央値以上のインデックスを抽出
        indices_ge_median = [i for i, d in enumerate(dists) if d >= median_v]

        if not indices_ge_median:
            # 該当がない場合はループスキップ
            continue

        # 中央値以上の候補からランダムに選ぶ
        chosen_idx = random.choice(indices_ge_median)
        goal = available[chosen_idx]
        g_coord = df[goal]

        start_coord.append(tuple(s_coord) if isinstance(df[start], (list, np.ndarray)) else df[start])
        goal_coord.append(g_coord)
        available.append(start)

    return known_points, start_coord, goal_coord



def generate_fields(width, height, known_points, goal_coords,
                    rep_cov_range=(8000, 10000),
                    att_cov_range=(15000, 20000),
                    per_known_samples=7,
                    per_goal_samples=7):
    """
    Parameters
    ----------
    width, height : int
        グリッドのサイズ（ピクセル）
    known_points : list[tuple]
        斥力を発生させる元点
    goal_coords : list[tuple]
        吸引を発生させるゴール点（長さ n_neuron_cluster）
    rep_cov_range : tuple(float, float)
        斥力度計算用共分散のスケール範囲
    att_cov_range : tuple(float, float)
        吸引度計算用共分散のスケール範囲
    per_known_samples : int
        各 known point からサンプルする周辺点数
    per_goal_samples : int
        各 goal からサンプルする周辺点数
    seed : int
        乱数シード（再現用）

    Returns
    -------
    repulsion  : np.ndarray shape=(height, width)
    attractions: list[np.ndarray]  長さ = len(goal_coords)
        attractions[i] は goal_coords[i] に対応
    """
    
    
    # ─────────────────────────────────────
    # 1) グリッドを 1 回だけ作る
    x = np.arange(1, width + 1)
    y = np.arange(1, height + 1)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack([xx.ravel(), yy.ravel()])  # shape (H*W, 2)

    # 2) repulsion マップ
    rep_z = np.zeros(grid.shape[0])

    for pt in known_points:
        for _ in range(per_known_samples):
            jitter = np.random.normal(scale=100, size=2)
            cx, cy = np.asarray(pt) + jitter
            cx = np.clip(cx, 0, width)
            cy = np.clip(cy, 0, height)

            cov_scale = np.random.uniform(*rep_cov_range)
            s1 = cov_scale * np.random.uniform(0.8, 1.2)
            s2 = cov_scale * np.random.uniform(0.8, 1.2)
            cov = np.array([[s1, 0], [0, s2]])

            p = multivariate_t.pdf(grid,
                                   loc=[cx, height - cy],
                                   shape=cov,
                                   df=1) * np.random.uniform(0.25, 1.0)
            rep_z += p

    repulsion = rep_z.reshape((height, width))

    # 3) attraction マップ
    attractions = []

    for gpt in goal_coords:
        att_z = np.zeros(grid.shape[0])

        for _ in range(per_goal_samples):
            jitter = np.random.normal(scale=25, size=2)
            cx, cy = np.asarray(gpt) + jitter
            cx = np.clip(cx, 0, width)
            cy = np.clip(cy, 0, height)

            cov_scale = np.random.uniform(*att_cov_range)
            s1 = cov_scale * np.random.uniform(0.8, 1.2)
            s2 = cov_scale * np.random.uniform(0.8, 1.2)
            cov = np.array([[s1, 0], [0, s2]])

            p = multivariate_t.pdf(grid,
                                   loc=[cx, height - cy],
                                   shape=cov,
                                   df=1) * np.random.uniform(0.25, 1.0)
            att_z += p

        attractions.append(att_z.reshape((height, width)))

    return repulsion, attractions


def simulate_agent_paths_pairs(repulsion,
                               attractions,        # list[array]  len == n_pairs
                               start_coords,       # list[tuple] len == n_pairs
                               goal_coords,        # list[tuple] len == n_pairs
                               steps=800,
                               hit_repeat=5):
    """
    各 (start, goal) ペアでエージェント群を動かし、パスを連結して返す。

    Returns
    -------
    paths : pd.DataFrame
        columns = ["x", "y", "ID", "pair_id"]
    """
    height, width = repulsion.shape
    all_paths = []

    for pair_idx, (start_coord, goal_coord, attraction) in enumerate(
            zip(start_coords, goal_coords, attractions)):
        goal_px = np.round(goal_coord).astype(int)

        temp=np.random.uniform(0.01, 0.25)
        agent_ids=range(20, 41)
        n = random.randint(1, 3)                        # 2〜5 個をランダムに決める
        agent_ids = random.sample(range(50, 71), n)

        for agent_id in agent_ids:
            # 初期位置 = start_coord ±5 px 以内
            pos = (np.asarray(start_coord, dtype=int) +
                   np.random.randint(-5, 5, size=2))
            pos = np.clip(pos, [0, 0], [width - 1, height - 1])
            path = [pos.copy()]
            touch_cnt = 0

            for step in range(steps):
                if np.linalg.norm(pos - goal_px) <= 25:
                    touch_cnt += 1
                    if touch_cnt >= hit_repeat:
                        break

                x0, y0 = pos
                x_rng = np.arange(max(1, x0 - 3), min(width,  x0 + 4))
                y_rng = np.arange(max(1, y0 - 3), min(height, y0 + 4))
                neigh = [(x, y) for x in x_rng for y in y_rng
                         if not (x == x0 and y == y0)]
                if not neigh:
                    break

                rep_vals = np.array([repulsion[y, x]   for (x, y) in neigh])
                att_vals = np.array([attraction[y, x] for (x, y) in neigh])

                prob1 = rep_vals / (rep_vals.mean() if rep_vals.mean() else 1)
                prob2 = att_vals / (att_vals.mean() if att_vals.mean() else 1)

                if step < 50:
                    prob = np.exp((1 - prob1) * agent_id) + \
                           np.exp((prob2 - 1) * 25)
                else:
                    prob = np.exp((1 - prob1) * agent_id) ** (1 - temp) + \
                           np.exp((prob2 - 1) * 150)     ** (1 - temp)
                prob /= prob.sum()

                pos = np.array(neigh[np.random.choice(len(neigh), p=prob)])
                path.append(pos.copy())
            
            df = pd.DataFrame(path, columns=["x", "y"])
            df["step_id"] = range(len(df))
            df["ID"]      = agent_id
            df["pair_id"] = pair_idx
            all_paths.append(df)

    return pd.concat(all_paths, ignore_index=True)


def save_trajectory_frames(paths, repulsion, max_steps, out_dir="images"):
    """
    step_id < 1 〜 max_steps までのフレームを壁紙スタイルで保存する関数。

    Parameters
    ----------
    paths : pd.DataFrame
        カラムに ["x", "y", "step_id", "ID", "pair_id"] を持つ軌跡データ。
    repulsion : np.ndarray
        背景ヒートマップとして描画する 2D グリッド。
    max_steps : int
        ステップ数（画像の最大数）。
    out_dir : str
        画像の保存先ディレクトリ。存在しない場合は作成される。
    """

    height, width = repulsion.shape
    colors = ["black", "blue"]
    cmap = LinearSegmentedColormap.from_list("blue_to_black", colors)

    for i in range(1, max_steps + 1):
        plt.figure(figsize=(10, 6))

        # 背景（ヒートマップ）
        plt.imshow(repulsion, extent=[1, width, 1, height], origin='lower', cmap=cmap)

        # エージェントの軌跡（step_id < i のみ）
        filtered = paths[paths["step_id"] < i]
        for (pair_id, agent_id), group in filtered.groupby(["pair_id", "ID"]):
            plt.plot(group["x"], group["y"], linewidth=0.5, color="darkred", alpha=0.5)

        # 描画要素の完全非表示化（壁紙仕様）
        plt.axis('off')

        # 保存（余白・ラベルなし、背景黒）
        plt.savefig(
            os.path.join(out_dir, f"{i}.png"),
            bbox_inches='tight',
            pad_inches=0,
            facecolor='black',
            dpi=300
        )
        plt.close()