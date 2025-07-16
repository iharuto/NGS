import axon_sim_fun as ASF
import datetime
import random

# 今日の日付を yymmdd 形式で取得
today_str = datetime.datetime.today().strftime("%y%m%d")
seed = int(today_str)

# 各種シード設定
random.seed(seed)

width = 1920
height = 1080
n_neuron_cluster = 20
max_steps = 800

df = ASF.get_app_locations()
known_points, start_coord, goal_coord = ASF.choose_apps(df, n_neuron_cluster)

repulsion, attraction = ASF.generate_fields(width, height, known_points, goal_coord)
goal_coords = [(x, height - y) for (x, y) in goal_coord]

paths = ASF.simulate_agent_paths_pairs(repulsion, attraction, start_coord, goal_coords, steps=max_steps, hit_repeat=1)

ASF.save_trajectory_frames(paths=paths, repulsion=repulsion, max_steps=max_steps)