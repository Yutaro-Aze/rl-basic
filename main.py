# pip install gymnasium[classic_control]
from collections import deque

import gymnasium as gym
import numpy as np

# ===== 1) 環境 =====
env = gym.make("CartPole-v1")

n_actions = env.action_space.n  # 2

# ===== 2) 離散化の設定 =====
# 観測: [cart_pos, cart_vel, pole_angle, pole_tip_vel]
# 安定学習のために妥当な範囲を切る（厳密ではなく「よく出る」領域をカバー）
bins = {
    # (最小, 最大, ビン数)
    0: (-4.8, 4.8, 8),         # 台車位置
    1: (-3.0, 3.0, 8),         # 台車速度
    2: (-0.418, 0.418, 16),    # 竿角度（約±24°）
    3: (-3.5, 3.5, 8),         # 竿先速度
}

def create_bin_edges(low, high, n_bins):
    # np.digitize 用に「境界」を作る（内部境界 n_bins-1 個）
    return np.linspace(low, high, n_bins - 1)

bin_edges = [create_bin_edges(*bins[i]) for i in range(4)]
n_bins_per_dim = [bins[i][2] for i in range(4)]

def discretize(obs):
    """連続観測 obs -> (i,j,k,l) の離散インデックス"""
    idxs = []
    for x, edges in zip(obs, bin_edges, strict=False):
        # 範囲外も 0..n_bins-1 にクリップされる
        idx = np.digitize(x, edges)  # 返り値は [0..len(edges)]
        idxs.append(idx)
    return tuple(idxs)  # 長さ4のタプル

# ===== 3) Qテーブル =====
Q = np.zeros((*n_bins_per_dim, n_actions), dtype=np.float32)

# ===== 4) ハイパーパラメータ =====
alpha = 0.1         # 学習率
gamma = 0.99        # 割引率
epsilon_start = 1.0 # 初期 ε
epsilon_end = 0.05  # 最終 ε
epsilon_decay_episodes = 800  # これくらいで線形に減衰
episodes = 2000
max_steps = 1000     # 念のための上限

def epsilon_by_episode(ep):
    if ep >= epsilon_decay_episodes:
        return epsilon_end
    return epsilon_start - (epsilon_start - epsilon_end) * (ep / epsilon_decay_episodes)

# ===== 5) 学習ループ =====
scores = []
moving = deque(maxlen=100)

for ep in range(episodes):
    obs, _ = env.reset()
    s = discretize(obs)
    epsilon = epsilon_by_episode(ep)
    total_r = 0

    for _t in range(max_steps):
        # ε-greedy
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            a = int(np.argmax(Q[s]))

        next_obs, r, terminated, truncated, _ = env.step(a)
        s_next = discretize(next_obs)
        total_r += r

        # 終端なら次状態価値は0、トランケート（時間切れ）は継続扱いでもよい
        if terminated:
            target = r
        else:
            target = r + gamma * np.max(Q[s_next])

        # Q更新
        Q[s + (a,)] += alpha * (target - Q[s + (a,)])

        s = s_next
        if terminated or truncated:
            break

    scores.append(total_r)
    moving.append(total_r)

    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1:4d} | avg100 = {np.mean(moving):.1f} | epsilon = {epsilon:.3f}")

# ===== 6) 評価（greedy） =====
def evaluate(n_eval_episodes=10, render=False):
    rs = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        s = discretize(obs)
        total = 0
        while True:
            a = int(np.argmax(Q[s]))
            obs, r, terminated, truncated, _ = env.step(a)
            s = discretize(obs)
            total += r
            if render:
                env.render()
            if terminated or truncated:
                rs.append(total)
                break
    return np.mean(rs), np.std(rs)

mean_r, std_r = evaluate()
print(f"Eval: mean return over 10 episodes = {mean_r:.1f} ± {std_r:.1f}")
env.close()
