from pathlib import Path
import copy
import math
import random
import time

import numpy as np
import pandas as pd

from citylearn.citylearn import CityLearnEnv


SCHEMA_PATH = Path("data/datasets/citylearn_challenge_2022_phase_1/schema.json")
DATA_DIR = Path("data/datasets/citylearn_challenge_2022_phase_1")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

# =========================
# 先短测试，确认无报错
# 跑通后再改成 None 跑全年
# =========================
MAX_REAL_STEPS = None

# MCTS 参数
MCTS_ITERATIONS = 8
ROLLING_HORIZON = 4
EXPLORATION_C = 1.2

# 缩小动作幅度，减少峰值风险
# CityLearn 当前实验约定：
# action > 0：充电
# action < 0：放电
# action = 0：不动作
ACTION_VALUES = [-0.1, 0.0, 0.1]

# tuned carbon-aware RBC 最优参数
PRICE_WEIGHT = 0.5
CARBON_WEIGHT = 0.5
LOW_Q = 0.2
HIGH_Q = 0.8
RBC_ACTION_MAGNITUDE = 0.1

# 自定义能碳目标权重
ALIGNMENT_WEIGHT = 8.0
POLICY_PRIOR_WEIGHT = 2.0
SMOOTH_WEIGHT = 0.3
PEAK_CHARGE_PENALTY_WEIGHT = 1.0

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
    arr = np.asarray(rewards, dtype=float)
    return float(np.sum(arr))


def constant_action(action_space, value):
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


def load_first_numeric_column(csv_path):
    df = pd.read_csv(csv_path)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_columns) == 0:
        raise ValueError(f"{csv_path} 中没有找到数值列")

    return df[numeric_columns[0]].astype(float).to_numpy()


def normalize_array(x):
    x = np.asarray(x, dtype=float)
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)

    if abs(x_max - x_min) < 1e-9:
        return np.zeros_like(x)

    return (x - x_min) / (x_max - x_min)


pricing = load_first_numeric_column(DATA_DIR / "pricing.csv")
carbon_intensity = load_first_numeric_column(DATA_DIR / "carbon_intensity.csv")

pricing_norm = normalize_array(pricing)
carbon_norm = normalize_array(carbon_intensity)

cost_carbon_score = PRICE_WEIGHT * pricing_norm + CARBON_WEIGHT * carbon_norm

low_threshold = np.quantile(cost_carbon_score, LOW_Q)
high_threshold = np.quantile(cost_carbon_score, HIGH_Q)


def get_score(time_step):
    idx = time_step % len(cost_carbon_score)
    return cost_carbon_score[idx]


def is_peak_hour(time_step):
    hour = time_step % 24
    return 17 <= hour <= 21


def carbon_aware_prior_action(time_step):
    """
    tuned carbon-aware RBC 先验：
    低价低碳时段：充电
    高价或高碳时段：放电
    其他时段：不动作
    """
    score = get_score(time_step)

    if score <= low_threshold:
        return RBC_ACTION_MAGNITUDE
    elif score >= high_threshold:
        return -RBC_ACTION_MAGNITUDE
    else:
        return 0.0


def sorted_actions_by_prior(time_step):
    """
    按照 tuned carbon-aware RBC 先验给动作排序，
    让 MCTS 优先扩展更可能合理的动作。
    """
    prior = carbon_aware_prior_action(time_step)

    return sorted(
        ACTION_VALUES,
        key=lambda a: abs(a - prior)
    )


def custom_cc_reward(env_rewards, action_value, previous_action, time_step):
    """
    Cost-Carbon-Aware MCTS 的自定义即时评价。

    目标：
    1. 低能碳得分时倾向充电；
    2. 高能碳得分时倾向放电；
    3. 避免频繁切换；
    4. 晚高峰避免充电；
    5. 少量保留 CityLearn 默认 reward 信息。
    """
    score = get_score(time_step)
    prior_action = carbon_aware_prior_action(time_step)

    # 1. 能碳方向匹配项
    # score 低，action 正值充电更好；
    # score 高，action 负值放电更好。
    alignment_reward = -ALIGNMENT_WEIGHT * (score - 0.5) * action_value

    # 2. 与 tuned carbon-aware RBC 先验接近
    prior_reward = -POLICY_PRIOR_WEIGHT * abs(action_value - prior_action)

    # 3. 动作平滑，避免频繁跳变
    smooth_penalty = -SMOOTH_WEIGHT * abs(action_value - previous_action)

    # 4. 晚高峰不要充电
    peak_penalty = 0.0
    if is_peak_hour(time_step) and action_value > 0:
        peak_penalty = -PEAK_CHARGE_PENALTY_WEIGHT * action_value

    # 5. 保留少量环境 reward，避免完全脱离 CityLearn
    env_reward = reward_to_float(env_rewards)
    env_reward_component = 0.02 * np.tanh(env_reward / 10.0)

    total = (
        alignment_reward
        + prior_reward
        + smooth_penalty
        + peak_penalty
        + env_reward_component
    )

    return float(total)


class MCTSNode:
    def __init__(
        self,
        env_state,
        env_time,
        previous_action,
        parent=None,
        action_from_parent=None,
        depth=0,
        path_reward=0.0,
        done=False,
    ):
        self.env_state = env_state
        self.env_time = env_time
        self.previous_action = previous_action

        self.parent = parent
        self.action_from_parent = action_from_parent
        self.depth = depth
        self.path_reward = path_reward
        self.done = done

        self.children = []
        self.untried_actions = sorted_actions_by_prior(env_time)

        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def average_value(self):
        if self.visits == 0:
            return -float("inf")
        return self.value / self.visits


def uct_select_child(node):
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
    action_value = node.untried_actions.pop(0)

    env_copy = copy.deepcopy(node.env_state)
    actions = constant_action(env_copy.action_space, action_value)

    _, env_rewards, done, _ = step_env(env_copy, actions)

    shaped_reward = custom_cc_reward(
        env_rewards=env_rewards,
        action_value=action_value,
        previous_action=node.previous_action,
        time_step=node.env_time,
    )

    child = MCTSNode(
        env_state=env_copy,
        env_time=node.env_time + 1,
        previous_action=action_value,
        parent=node,
        action_from_parent=action_value,
        depth=node.depth + 1,
        path_reward=node.path_reward + shaped_reward,
        done=done,
    )

    node.children.append(child)
    return child


def rollout_from_node(node, horizon):
    sim_env = copy.deepcopy(node.env_state)

    total_reward = node.path_reward
    current_depth = node.depth
    current_time = node.env_time
    previous_action = node.previous_action
    done = node.done

    while current_depth < horizon and not done:
        # rollout 不再完全随机，而是多数情况下采用 carbon-aware 先验
        if random.random() < 0.8:
            action_value = carbon_aware_prior_action(current_time)
        else:
            action_value = random.choice(ACTION_VALUES)

        actions = constant_action(sim_env.action_space, action_value)

        _, env_rewards, done, _ = step_env(sim_env, actions)

        shaped_reward = custom_cc_reward(
            env_rewards=env_rewards,
            action_value=action_value,
            previous_action=previous_action,
            time_step=current_time,
        )

        total_reward += shaped_reward

        previous_action = action_value
        current_time += 1
        current_depth += 1

    return total_reward


def backup(node, total_return):
    current = node

    while current is not None:
        current.visits += 1
        current.value += total_return
        current = current.parent


def cc_mcts_plan_action(
    env,
    current_time_step,
    previous_action,
    horizon=ROLLING_HORIZON,
    iterations=MCTS_ITERATIONS,
):
    root_env = copy.deepcopy(env)

    root = MCTSNode(
        env_state=root_env,
        env_time=current_time_step,
        previous_action=previous_action,
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

    best_child = max(root.children, key=lambda child: child.average_value())
    return best_child.action_from_parent


def run_cc_mcts():
    print("schema 文件是否存在：", SCHEMA_PATH.exists())
    print("schema 路径：", SCHEMA_PATH)
    print("pricing 长度：", len(pricing))
    print("carbon_intensity 长度：", len(carbon_intensity))
    print("low_threshold：", low_threshold)
    print("high_threshold：", high_threshold)

    env = CityLearnEnv(str(SCHEMA_PATH))
    reset_env(env)

    done = False
    time_step = 0
    previous_action = 0.0
    records = []

    start_time = time.time()

    print("\n========== 开始运行 cc_mcts ==========")
    print("MAX_REAL_STEPS =", MAX_REAL_STEPS)
    print("MCTS_ITERATIONS =", MCTS_ITERATIONS)
    print("ROLLING_HORIZON =", ROLLING_HORIZON)
    print("ACTION_VALUES =", ACTION_VALUES)

    while not done:
        if MAX_REAL_STEPS is not None and time_step >= MAX_REAL_STEPS:
            print(f"\n短测试达到 {MAX_REAL_STEPS} 步，提前停止。")
            break

        plan_start = time.time()

        action_value = cc_mcts_plan_action(
            env=env,
            current_time_step=time_step,
            previous_action=previous_action,
            horizon=ROLLING_HORIZON,
            iterations=MCTS_ITERATIONS,
        )

        planning_time = time.time() - plan_start

        actions = constant_action(env.action_space, action_value)
        observations, rewards, done, info = step_env(env, actions)

        reward_value = reward_to_float(rewards)
        score = get_score(time_step)
        prior_action = carbon_aware_prior_action(time_step)

        records.append(
            {
                "time_step": time_step,
                "action_value": action_value,
                "prior_action": prior_action,
                "score": score,
                "env_reward": reward_value,
                "planning_time_seconds": planning_time,
            }
        )

        previous_action = action_value
        time_step += 1

        if time_step % 500 == 0:
            print(
                f"cc_mcts 已运行到第 {time_step} 步，"
                f"当前动作={action_value}, "
                f"prior={prior_action}, "
                f"score={score:.4f}, "
                f"单步规划耗时={planning_time:.3f}s"
            )

    total_time = time.time() - start_time

    action_df = pd.DataFrame(records)
    action_path = RESULT_DIR / "cc_mcts_actions_debug.csv"
    action_df.to_csv(action_path, index=False)

    print("\n========== cc_mcts 运行结束 ==========")
    print("实际运行步数：", time_step)
    print("总耗时：", total_time, "秒")
    print("动作记录保存到：", action_path)

    if done and MAX_REAL_STEPS is None:
        kpis = env.evaluate()
        kpis["method"] = "cc_mcts"

        kpi_path = RESULT_DIR / "cc_mcts_kpis.csv"
        kpis.to_csv(kpi_path, index=False)

        print("全年仿真完成，KPI 保存到：", kpi_path)
        print(kpis)
    else:
        print("\n注意：当前是短测试，不生成正式 KPI。")
        print("确认无报错后，把 MAX_REAL_STEPS 改成 None，再跑全年。")


if __name__ == "__main__":
    run_cc_mcts()