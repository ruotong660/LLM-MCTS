from pathlib import Path
import copy
import math
import random
import time

import numpy as np
import pandas as pd

from citylearn.citylearn import CityLearnEnv


SCHEMA_PATH = Path("data/datasets/citylearn_challenge_2022_phase_1/schema.json")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

# =========================
# 先短测试，不要直接全年跑
# 跑通后再把 MAX_REAL_STEPS 改成 None
# =========================
MAX_REAL_STEPS = None

# MCTS 参数
MCTS_ITERATIONS = 8
ROLLING_HORIZON = 4
EXPLORATION_C = 1.4

# CityLearn 当前实验约定：
# action > 0：充电
# action < 0：放电
# action = 0：不动作
ACTION_VALUES = [-0.2, -0.1, 0.0, 0.1, 0.2]

random.seed(42)
np.random.seed(42)


def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def convert_done(done):
    if isinstance(done, (list, tuple, np.ndarray)):
        return bool(np.all(done))
    return bool(done)


def step_env(env, actions):
    result = env.step(actions)

    if len(result) == 5:
        observations, rewards, terminated, truncated, info = result
        done = convert_done(terminated) or convert_done(truncated)
    else:
        observations, rewards, done, info = result
        done = convert_done(done)

    return observations, rewards, done, info


def reward_to_float(rewards):
    """
    CityLearn 多建筑环境可能返回 list / ndarray。
    这里把所有建筑 reward 合成一个标量，MCTS 目标是最大化累计 reward。
    """
    arr = np.asarray(rewards, dtype=float)
    return float(np.sum(arr))


def constant_action(action_space, value):
    """
    给所有 building 设置同一个动作值。
    """
    if isinstance(action_space, list):
        actions = []
        for space in action_space:
            action = np.full(space.shape, value, dtype=np.float32)
            action = np.clip(action, space.low, space.high)
            actions.append(action)
        return actions

    action = np.full(action_space.shape, value, dtype=np.float32)
    action = np.clip(action, action_space.low, action_space.high)
    return action


class MCTSNode:
    def __init__(
        self,
        env_state,
        parent=None,
        action_from_parent=None,
        depth=0,
        path_reward=0.0,
        done=False,
    ):
        self.env_state = env_state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.depth = depth
        self.path_reward = path_reward
        self.done = done

        self.children = []
        self.untried_actions = ACTION_VALUES.copy()

        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def average_value(self):
        if self.visits == 0:
            return -float("inf")
        return self.value / self.visits


def uct_select_child(node):
    """
    UCT 选择。
    """
    best_score = -float("inf")
    best_child = None

    for child in node.children:
        if child.visits == 0:
            score = float("inf")
        else:
            exploitation = child.value / child.visits
            exploration = EXPLORATION_C * math.sqrt(
                math.log(node.visits + 1) / child.visits
            )
            score = exploitation + exploration

        if score > best_score:
            best_score = score
            best_child = child

    return best_child


def expand_node(node):
    """
    从未尝试动作中取一个动作，扩展一个子节点。
    """
    action_value = node.untried_actions.pop(0)

    env_copy = copy.deepcopy(node.env_state)
    actions = constant_action(env_copy.action_space, action_value)

    _, rewards, done, _ = step_env(env_copy, actions)

    immediate_reward = reward_to_float(rewards)

    child = MCTSNode(
        env_state=env_copy,
        parent=node,
        action_from_parent=action_value,
        depth=node.depth + 1,
        path_reward=node.path_reward + immediate_reward,
        done=done,
    )

    node.children.append(child)
    return child


def rollout_from_node(node, horizon):
    """
    从当前节点开始，随机模拟到滚动窗口结束。
    """
    sim_env = copy.deepcopy(node.env_state)

    total_reward = node.path_reward
    current_depth = node.depth
    done = node.done

    while current_depth < horizon and not done:
        action_value = random.choice(ACTION_VALUES)
        actions = constant_action(sim_env.action_space, action_value)

        _, rewards, done, _ = step_env(sim_env, actions)

        total_reward += reward_to_float(rewards)
        current_depth += 1

    return total_reward


def backup(node, total_return):
    """
    反向传播。
    """
    current = node
    while current is not None:
        current.visits += 1
        current.value += total_return
        current = current.parent


def mcts_plan_action(env, horizon=ROLLING_HORIZON, iterations=MCTS_ITERATIONS):
    """
    在当前真实环境状态下，使用 MCTS 选择当前一步动作。
    """
    root_env = copy.deepcopy(env)

    root = MCTSNode(
        env_state=root_env,
        parent=None,
        action_from_parent=None,
        depth=0,
        path_reward=0.0,
        done=False,
    )

    for _ in range(iterations):
        node = root

        # 1. Selection
        while (
            node.depth < horizon
            and not node.done
            and node.is_fully_expanded()
            and len(node.children) > 0
        ):
            node = uct_select_child(node)

        # 2. Expansion
        if node.depth < horizon and not node.done and len(node.untried_actions) > 0:
            node = expand_node(node)

        # 3. Simulation
        total_return = rollout_from_node(node, horizon)

        # 4. Backup
        backup(node, total_return)

    if len(root.children) == 0:
        return 0.0

    # 选择平均价值最高的第一步动作
    best_child = max(root.children, key=lambda child: child.average_value())
    return best_child.action_from_parent


def run_standard_mcts():
    print("schema 文件是否存在：", SCHEMA_PATH.exists())
    print("schema 路径：", SCHEMA_PATH)

    env = CityLearnEnv(str(SCHEMA_PATH))
    reset_env(env)

    done = False
    time_step = 0
    records = []

    start_time = time.time()

    print("\n========== 开始运行 standard_mcts ==========")
    print("MAX_REAL_STEPS =", MAX_REAL_STEPS)
    print("MCTS_ITERATIONS =", MCTS_ITERATIONS)
    print("ROLLING_HORIZON =", ROLLING_HORIZON)
    print("ACTION_VALUES =", ACTION_VALUES)

    while not done:
        if MAX_REAL_STEPS is not None and time_step >= MAX_REAL_STEPS:
            print(f"\n短测试达到 {MAX_REAL_STEPS} 步，提前停止。")
            break

        plan_start = time.time()

        action_value = mcts_plan_action(
            env,
            horizon=ROLLING_HORIZON,
            iterations=MCTS_ITERATIONS,
        )

        planning_time = time.time() - plan_start

        actions = constant_action(env.action_space, action_value)
        observations, rewards, done, info = step_env(env, actions)

        reward_value = reward_to_float(rewards)

        records.append(
            {
                "time_step": time_step,
                "action_value": action_value,
                "reward": reward_value,
                "planning_time_seconds": planning_time,
            }
        )

        time_step += 1

        if time_step % 500 == 0:
            print(
                f"standard_mcts 已运行到第 {time_step} 步，"
                f"当前动作={action_value}, "
                f"单步规划耗时={planning_time:.3f}s"
            )

    total_time = time.time() - start_time

    action_df = pd.DataFrame(records)
    action_path = RESULT_DIR / "standard_mcts_actions_debug.csv"
    action_df.to_csv(action_path, index=False)

    print("\n========== standard_mcts 运行结束 ==========")
    print("实际运行步数：", time_step)
    print("总耗时：", total_time, "秒")
    print("动作记录保存到：", action_path)

    if done and MAX_REAL_STEPS is None:
        kpis = env.evaluate()
        kpis["method"] = "standard_mcts"

        kpi_path = RESULT_DIR / "standard_mcts_kpis.csv"
        kpis.to_csv(kpi_path, index=False)

        print("全年仿真完成，KPI 保存到：", kpi_path)
        print(kpis)
    else:
        print("\n注意：当前是短测试，不生成正式 KPI。")
        print("确认无报错后，把 MAX_REAL_STEPS 改成 None，再跑全年。")


if __name__ == "__main__":
    run_standard_mcts()