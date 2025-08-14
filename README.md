# 表形式Q学習によるCartPoleの解法

このプロジェクトは、OpenAI Gym / Gymnasium の古典制御問題 [CartPole-v1](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) を、**表形式（タブラー）Q学習** で解く実装です。

CartPole の状態空間は **連続値** なので、そのままでは Qテーブルを使えません。本実装では、状態をビン分割して離散化し、表形式のQ学習を適用します。

---

## 📜 概要

- **環境**: CartPole-v1
- **アルゴリズム**: Q学習（オフポリシーTD制御）
- **状態表現**: 観測値の離散化（ビン分割）
- **方策**: ε-greedy
- **目標**: 100エピソード平均報酬が 475 以上（`v1` の解法条件）

---

## 🧮 手法

1. **離散化**  
   4次元の連続状態 `(台車位置, 台車速度, 棒の角度, 棒先の速度)` を、等間隔のビンに分割し、離散インデックスに変換します。

2. **Qテーブル**  
   形状は `[bins_cart_pos, bins_cart_vel, bins_angle, bins_angle_vel, n_actions]`。

3. **学習則**  
   ```math
   Q(s,a) ← Q(s,a) + α [ r + γ max_{a'} Q(s', a') − Q(s,a) ]
