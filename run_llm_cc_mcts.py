from pathlib import Path
import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd

from citylearn.citylearn import CityLearnEnv
from experiment_config import print_dataset_registry, resolve_dataset
from llm_providers import (
    available_providers,
    get_default_model,
    print_available_providers,
    semantic_param_path as provider_semantic_param_path,
    slugify_model_name,
)

try:
    import torch
except ImportError:
    torch = None


DEFAULT_SCHEMA_PATH = Path("data/datasets/citylearn_challenge_2022_phase_1/schema.json")
DEFAULT_DATA_DIR = DEFAULT_SCHEMA_PATH.parent
DEFAULT_RESULT_DIR = Path("results")

SCHEMA_PATH = DEFAULT_SCHEMA_PATH
DATA_DIR = DEFAULT_DATA_DIR
RESULT_DIR = DEFAULT_RESULT_DIR
RESULT_DIR.mkdir(exist_ok=True)

# ============================================================
# 默认先短测试。
# 三个语义场景短测试都成功后，再改成 None 跑全年。
# ============================================================
MAX_REAL_STEPS = None

# MCTS 参数
MCTS_ITERATIONS = 8
ROLLING_HORIZON = 4
EXPLORATION_C = 1.2

# CityLearn 当前实验约定：
# action > 0：充电
# action < 0：放电
# action = 0：不动作
ACTION_VALUES = [-0.1, 0.0, 0.1]

# ============================================================
# 内置的三个语义解析场景。
#
# 每次启动只运行一个场景，避免一次运行中混杂多套语义参数。
# 示例：
# python run_llm_cc_mcts.py --semantic-scenario cost_priority
# python run_llm_cc_mcts.py --semantic-scenario low_carbon_priority
# python run_llm_cc_mcts.py --semantic-scenario peak_aware
#
# 仍兼容旧写法：
# SEMANTIC_PARAM_PATH=semantic_params/cost_priority.json METHOD_NAME=llm_cc_mcts_cost_priority python run_llm_cc_mcts.py
# ============================================================
SEMANTIC_SCENARIOS = {
    "cost_priority": {
        "param_path": Path("semantic_params/cost_priority.json"),
        "method_name": "llm_cc_mcts_cost_priority",
    },
    "low_carbon_priority": {
        "param_path": Path("semantic_params/low_carbon_priority.json"),
        "method_name": "llm_cc_mcts_low_carbon_priority",
    },
    "peak_aware": {
        "param_path": Path("semantic_params/peak_aware.json"),
        "method_name": "llm_cc_mcts_peak_aware",
    },
}

REQUIRED_SEMANTIC_KEYS = [
    "price_weight",
    "carbon_weight",
    "low_q",
    "high_q",
    "rbc_action_magnitude",
    "alignment_weight",
    "policy_prior_weight",
    "smooth_weight",
    "peak_charge_penalty_weight",
]

SEMANTIC_PARAM_PATH = None
METHOD_NAME = None
semantic_params = {}

PRICE_WEIGHT = 0.0
CARBON_WEIGHT = 0.0
LOW_Q = 0.0
HIGH_Q = 0.0
RBC_ACTION_MAGNITUDE = 0.0

ALIGNMENT_WEIGHT = 0.0
POLICY_PRIOR_WEIGHT = 0.0
SMOOTH_WEIGHT = 0.0
PEAK_CHARGE_PENALTY_WEIGHT = 0.0

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
    if torch is None:
        result = env.step(actions)
    else:
        # CityLearn 2023 datasets may use PyTorch internally. MCTS repeatedly
        # deep-copies simulated environments, so avoid keeping autograd graphs
        # inside env state after step().
        with torch.no_grad():
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
    """
    给所有 building 设置相同动作。
    当前为了降低搜索复杂度，先采用 district-level shared action。
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


pricing = np.array([], dtype=float)
carbon_intensity = np.array([], dtype=float)
pricing_norm = np.array([], dtype=float)
carbon_norm = np.array([], dtype=float)

# 能碳综合得分：
# 分数越低，越适合充电；
# 分数越高，越适合放电。
cost_carbon_score = np.array([], dtype=float)
low_threshold = 0.0
high_threshold = 0.0


def configure_result_dir(result_dir):
    global RESULT_DIR

    RESULT_DIR = Path(result_dir)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def configure_dataset(schema_path=DEFAULT_SCHEMA_PATH, data_dir=None):
    global SCHEMA_PATH
    global DATA_DIR
    global pricing
    global carbon_intensity
    global pricing_norm
    global carbon_norm

    SCHEMA_PATH = Path(schema_path)
    DATA_DIR = Path(data_dir) if data_dir is not None else SCHEMA_PATH.parent

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"找不到 schema 文件：{SCHEMA_PATH}")

    pricing_path = DATA_DIR / "pricing.csv"
    carbon_path = DATA_DIR / "carbon_intensity.csv"

    if not pricing_path.exists():
        raise FileNotFoundError(f"找不到 pricing 文件：{pricing_path}")

    if not carbon_path.exists():
        raise FileNotFoundError(f"找不到 carbon_intensity 文件：{carbon_path}")

    pricing = load_first_numeric_column(pricing_path)
    carbon_intensity = load_first_numeric_column(carbon_path)
    pricing_norm = normalize_array(pricing)
    carbon_norm = normalize_array(carbon_intensity)


def read_semantic_params(semantic_param_path):
    semantic_param_path = Path(semantic_param_path)

    if not semantic_param_path.exists():
        raise FileNotFoundError(
            f"找不到语义参数文件：{semantic_param_path}\n"
            "请先生成 semantic_params.json，或者通过 "
            "--semantic-param-path / --semantic-scenario 指定参数文件。"
        )

    with open(semantic_param_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    missing_keys = [key for key in REQUIRED_SEMANTIC_KEYS if key not in params]
    if missing_keys:
        raise KeyError(
            f"{semantic_param_path} 缺少必要字段：{', '.join(missing_keys)}"
        )

    return params


def configure_semantic_run(semantic_param_path, method_name):
    global SEMANTIC_PARAM_PATH
    global METHOD_NAME
    global semantic_params
    global PRICE_WEIGHT
    global CARBON_WEIGHT
    global LOW_Q
    global HIGH_Q
    global RBC_ACTION_MAGNITUDE
    global ALIGNMENT_WEIGHT
    global POLICY_PRIOR_WEIGHT
    global SMOOTH_WEIGHT
    global PEAK_CHARGE_PENALTY_WEIGHT
    global cost_carbon_score
    global low_threshold
    global high_threshold

    if len(pricing_norm) == 0 or len(carbon_norm) == 0:
        raise RuntimeError("请先调用 configure_dataset() 读取数据集。")

    SEMANTIC_PARAM_PATH = Path(semantic_param_path)
    METHOD_NAME = method_name
    semantic_params = read_semantic_params(SEMANTIC_PARAM_PATH)

    PRICE_WEIGHT = float(semantic_params["price_weight"])
    CARBON_WEIGHT = float(semantic_params["carbon_weight"])
    LOW_Q = float(semantic_params["low_q"])
    HIGH_Q = float(semantic_params["high_q"])
    RBC_ACTION_MAGNITUDE = float(semantic_params["rbc_action_magnitude"])

    ALIGNMENT_WEIGHT = float(semantic_params["alignment_weight"])
    POLICY_PRIOR_WEIGHT = float(semantic_params["policy_prior_weight"])
    SMOOTH_WEIGHT = float(semantic_params["smooth_weight"])
    PEAK_CHARGE_PENALTY_WEIGHT = float(
        semantic_params["peak_charge_penalty_weight"]
    )

    cost_carbon_score = PRICE_WEIGHT * pricing_norm + CARBON_WEIGHT * carbon_norm
    low_threshold = float(np.quantile(cost_carbon_score, LOW_Q))
    high_threshold = float(np.quantile(cost_carbon_score, HIGH_Q))


def get_score(time_step):
    if len(cost_carbon_score) == 0:
        raise RuntimeError("请先配置语义参数，再调用 get_score()。")

    idx = time_step % len(cost_carbon_score)
    return float(cost_carbon_score[idx])


def is_peak_hour(time_step):
    hour = time_step % 24
    return 17 <= hour <= 21


def carbon_aware_prior_action(time_step):
    """
    由 LLM 语义参数引导的动作先验。

    score 较低：低价/低碳时段，倾向充电；
    score 较高：高价/高碳时段，倾向放电；
    中间时段：不动作。
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
    用语义解析得到的动作先验，对 MCTS expansion 阶段的候选动作排序。
    这体现 LLM 对 MCTS 候选动作生成/筛选的引导作用。
    """
    prior = carbon_aware_prior_action(time_step)
    return sorted(ACTION_VALUES, key=lambda a: abs(a - prior))


def custom_llm_reward(env_rewards, action_value, previous_action, time_step):
    """
    LLM-guided MCTS 的自定义即时评价函数。

    LLM 不直接输出最终动作，而是通过 semantic_params.json 更新：
    - 成本权重 PRICE_WEIGHT
    - 碳排权重 CARBON_WEIGHT
    - 方向匹配权重 ALIGNMENT_WEIGHT
    - 动作先验权重 POLICY_PRIOR_WEIGHT
    - 平滑惩罚 SMOOTH_WEIGHT
    - 晚高峰充电惩罚 PEAK_CHARGE_PENALTY_WEIGHT

    MCTS 根据这些参数进行搜索。
    """
    score = get_score(time_step)
    prior_action = carbon_aware_prior_action(time_step)

    # 1. 能碳方向匹配项
    # score 低时，action > 0 充电更好；
    # score 高时，action < 0 放电更好。
    alignment_reward = -ALIGNMENT_WEIGHT * (score - 0.5) * action_value

    # 2. 接近语义动作先验
    prior_reward = -POLICY_PRIOR_WEIGHT * abs(action_value - prior_action)

    # 3. 动作平滑，抑制频繁切换
    smooth_penalty = -SMOOTH_WEIGHT * abs(action_value - previous_action)

    # 4. 晚高峰避免充电
    peak_penalty = 0.0
    if is_peak_hour(time_step) and action_value > 0:
        peak_penalty = -PEAK_CHARGE_PENALTY_WEIGHT * action_value

    # 5. 保留少量 CityLearn 环境 reward 信息，避免完全脱离环境反馈
    env_reward = reward_to_float(env_rewards)
    env_reward_component = 0.02 * np.tanh(env_reward / 10.0)

    total_reward = (
        alignment_reward
        + prior_reward
        + smooth_penalty
        + peak_penalty
        + env_reward_component
    )

    return float(total_reward)


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
    """
    Expansion:
    按照语义动作先验排序后的 untried_actions 扩展节点。
    """
    action_value = node.untried_actions.pop(0)

    env_copy = copy.deepcopy(node.env_state)
    actions = constant_action(env_copy.action_space, action_value)

    _, env_rewards, done, _ = step_env(env_copy, actions)

    shaped_reward = custom_llm_reward(
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
    """
    Simulation / Rollout:
    不完全随机，而是大概率采用 LLM 语义参数引导的 carbon_aware_prior_action。
    """
    sim_env = copy.deepcopy(node.env_state)

    total_reward = node.path_reward
    current_depth = node.depth
    current_time = node.env_time
    previous_action = node.previous_action
    done = node.done

    while current_depth < horizon and not done:
        if random.random() < 0.8:
            action_value = carbon_aware_prior_action(current_time)
        else:
            action_value = random.choice(ACTION_VALUES)

        actions = constant_action(sim_env.action_space, action_value)

        _, env_rewards, done, _ = step_env(sim_env, actions)

        shaped_reward = custom_llm_reward(
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


def llm_mcts_plan_action(
    env,
    current_time_step,
    previous_action,
    horizon=ROLLING_HORIZON,
    iterations=MCTS_ITERATIONS,
):
    """
    在当前真实环境状态下，使用 LLM-guided MCTS 选择当前一步动作。
    """
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


def run_llm_cc_mcts(
    max_real_steps=MAX_REAL_STEPS,
    mcts_iterations=MCTS_ITERATIONS,
    rolling_horizon=ROLLING_HORIZON,
):
    if METHOD_NAME is None or SEMANTIC_PARAM_PATH is None:
        raise RuntimeError("请先配置语义场景，再调用 run_llm_cc_mcts()。")

    print("schema 文件是否存在：", SCHEMA_PATH.exists())
    print("schema 路径：", SCHEMA_PATH)
    print("data 目录：", DATA_DIR)
    print("pricing 长度：", len(pricing))
    print("carbon_intensity 长度：", len(carbon_intensity))
    print("low_threshold：", low_threshold)
    print("high_threshold：", high_threshold)

    print(f"\n========== 开始运行 {METHOD_NAME} ==========")
    print("语义参数文件：", SEMANTIC_PARAM_PATH)
    print("语义场景：", semantic_params.get("scenario_name"))
    print("语义解释：", semantic_params.get("explanation"))
    print("LLM provider：", semantic_params.get("llm_provider"))
    print("LLM model：", semantic_params.get("model_name"))
    print("PRICE_WEIGHT =", PRICE_WEIGHT)
    print("CARBON_WEIGHT =", CARBON_WEIGHT)
    print("LOW_Q =", LOW_Q)
    print("HIGH_Q =", HIGH_Q)
    print("RBC_ACTION_MAGNITUDE =", RBC_ACTION_MAGNITUDE)
    print("ALIGNMENT_WEIGHT =", ALIGNMENT_WEIGHT)
    print("POLICY_PRIOR_WEIGHT =", POLICY_PRIOR_WEIGHT)
    print("SMOOTH_WEIGHT =", SMOOTH_WEIGHT)
    print("PEAK_CHARGE_PENALTY_WEIGHT =", PEAK_CHARGE_PENALTY_WEIGHT)
    print("MAX_REAL_STEPS =", max_real_steps)
    print("MCTS_ITERATIONS =", mcts_iterations)
    print("ROLLING_HORIZON =", rolling_horizon)
    print("EXPLORATION_C =", EXPLORATION_C)
    print("ACTION_VALUES =", ACTION_VALUES)

    env = CityLearnEnv(str(SCHEMA_PATH))
    reset_env(env)

    done = False
    time_step = 0
    previous_action = 0.0
    records = []

    start_time = time.time()

    while not done:
        if max_real_steps is not None and time_step >= max_real_steps:
            print(f"\n短测试达到 {max_real_steps} 步，提前停止。")
            break

        plan_start = time.time()

        action_value = llm_mcts_plan_action(
            env=env,
            current_time_step=time_step,
            previous_action=previous_action,
            horizon=rolling_horizon,
            iterations=mcts_iterations,
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
                "method": METHOD_NAME,
                "semantic_param_path": str(SEMANTIC_PARAM_PATH),
                "scenario_name": semantic_params.get("scenario_name"),
                "price_weight": PRICE_WEIGHT,
                "carbon_weight": CARBON_WEIGHT,
                "smooth_weight": SMOOTH_WEIGHT,
                "peak_charge_penalty_weight": PEAK_CHARGE_PENALTY_WEIGHT,
            }
        )

        previous_action = action_value
        time_step += 1

        if time_step % 500 == 0:
            print(
                f"{METHOD_NAME} 已运行到第 {time_step} 步，"
                f"当前动作={action_value}, "
                f"prior={prior_action}, "
                f"score={score:.4f}, "
                f"单步规划耗时={planning_time:.3f}s"
            )

    total_time = time.time() - start_time

    action_df = pd.DataFrame(records)
    action_path = RESULT_DIR / f"{METHOD_NAME}_actions_debug.csv"
    action_df.to_csv(action_path, index=False)

    print(f"\n========== {METHOD_NAME} 运行结束 ==========")
    print("实际运行步数：", time_step)
    print("总耗时：", total_time, "秒")
    print("动作记录保存到：", action_path)

    if done and max_real_steps is None:
        kpis = env.evaluate()
        kpis["method"] = METHOD_NAME
        kpis["semantic_param_path"] = str(SEMANTIC_PARAM_PATH)
        kpis["scenario_name"] = semantic_params.get("scenario_name")
        kpis["price_weight"] = PRICE_WEIGHT
        kpis["carbon_weight"] = CARBON_WEIGHT
        kpis["smooth_weight"] = SMOOTH_WEIGHT
        kpis["peak_charge_penalty_weight"] = PEAK_CHARGE_PENALTY_WEIGHT

        kpi_path = RESULT_DIR / f"{METHOD_NAME}_kpis.csv"
        kpis.to_csv(kpi_path, index=False)

        print("全年仿真完成，KPI 保存到：", kpi_path)
        print(kpis)
    else:
        print("\n注意：当前是短测试，不生成正式 KPI。")
        print("确认无报错后，用 --max-real-steps None 再跑全年。")


def parse_optional_int(value):
    if value is None:
        return None

    value = str(value).strip()
    if value.lower() in {"none", "null", "full", "all"}:
        return None

    return int(value)


def print_available_scenarios():
    print("可用语义场景：")
    for scenario_name, config in SEMANTIC_SCENARIOS.items():
        print(
            f"- {scenario_name}: "
            f"{config['param_path']} -> {config['method_name']}"
        )


def build_arg_parser():
    examples = """
示例：
  python run_llm_cc_mcts.py --semantic-scenario cost_priority --max-real-steps 200
  python run_llm_cc_mcts.py --semantic-scenario low_carbon_priority --max-real-steps 200
  python run_llm_cc_mcts.py --semantic-scenario peak_aware --max-real-steps 200

全年仿真：
  python run_llm_cc_mcts.py --semantic-scenario cost_priority --max-real-steps None

自定义语义参数文件：
  python run_llm_cc_mcts.py --semantic-param-path semantic_params/my_case.json --method-name llm_cc_mcts_my_case
"""

    schema_path = Path(os.getenv("SCHEMA_PATH", DEFAULT_SCHEMA_PATH))
    data_dir = os.getenv("DATA_DIR")
    result_dir = Path(os.getenv("RESULT_DIR", DEFAULT_RESULT_DIR))
    max_real_steps = parse_optional_int(os.getenv("MAX_REAL_STEPS", MAX_REAL_STEPS))
    mcts_iterations = int(os.getenv("MCTS_ITERATIONS", MCTS_ITERATIONS))
    rolling_horizon = int(os.getenv("ROLLING_HORIZON", ROLLING_HORIZON))
    exploration_c = float(os.getenv("EXPLORATION_C", EXPLORATION_C))

    parser = argparse.ArgumentParser(
        description="分开运行单个 LLM 语义解析场景的 CityLearn MCTS 实验。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="列出内置的 3 个语义场景后退出。",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="列出已注册的数据集后退出。",
    )
    parser.add_argument(
        "--list-llm-providers",
        action="store_true",
        help="列出已支持的 LLM provider 后退出。",
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("DATASET_ID"),
        help="选择 configs/datasets.json 中注册的数据集 ID。",
    )
    parser.add_argument(
        "--semantic-scenario",
        "--scenario",
        choices=sorted(SEMANTIC_SCENARIOS),
        dest="scenario",
        help="选择一个内置语义场景；每次启动只运行一个场景。",
    )
    parser.add_argument(
        "--semantic-param-path",
        type=Path,
        default=None,
        help="自定义语义参数 JSON 文件路径；不能和 --semantic-scenario 同时使用。",
    )
    parser.add_argument(
        "--llm-provider",
        choices=available_providers(),
        default=None,
        help=(
            "选择语义参数所属的 LLM provider；设置后默认读取 "
            "semantic_params/{provider}/{model}/{scenario}.json。"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="语义参数所属模型名；与 --llm-provider 一起使用。",
    )
    parser.add_argument(
        "--semantic-param-root",
        type=Path,
        default=Path("semantic_params"),
        help="按 provider/model 组织的语义参数根目录。",
    )
    parser.add_argument(
        "--method-name",
        default=None,
        help="输出文件使用的方法名；内置场景默认使用各自独立 method_name。",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=schema_path,
        help="CityLearn schema.json 路径。",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(data_dir) if data_dir else None,
        help="包含 pricing.csv 和 carbon_intensity.csv 的目录；默认使用 schema 所在目录。",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=result_dir,
        help="结果输出目录。",
    )
    parser.add_argument(
        "--max-real-steps",
        type=parse_optional_int,
        default=max_real_steps,
        help="真实环境运行步数；传 None/full/all 表示跑完整 episode。",
    )
    parser.add_argument(
        "--mcts-iterations",
        type=int,
        default=mcts_iterations,
        help="每步 MCTS 迭代次数。",
    )
    parser.add_argument(
        "--rolling-horizon",
        type=int,
        default=rolling_horizon,
        help="MCTS 滚动规划 horizon。",
    )
    parser.add_argument(
        "--exploration-c",
        type=float,
        default=exploration_c,
        help="UCT exploration 系数。",
    )
    return parser


def resolve_semantic_run(args, parser):
    if args.scenario is not None and args.semantic_param_path is not None:
        parser.error("--semantic-scenario 和 --semantic-param-path 只能选择一个。")

    if args.scenario is not None:
        config = SEMANTIC_SCENARIOS[args.scenario]
        if args.llm_provider is not None:
            model_name = args.model or get_default_model(args.llm_provider)
            semantic_param_path = provider_semantic_param_path(
                provider_name=args.llm_provider,
                model_name=model_name,
                scenario_name=args.scenario,
                output_root=args.semantic_param_root,
            )
            method_name = args.method_name or (
                f"{config['method_name']}_{args.llm_provider}_"
                f"{slugify_model_name(model_name)}"
            )
        else:
            semantic_param_path = config["param_path"]
            method_name = args.method_name or config["method_name"]
        return semantic_param_path, method_name

    semantic_param_path = args.semantic_param_path
    if semantic_param_path is None:
        semantic_param_path = Path(os.getenv("SEMANTIC_PARAM_PATH", "semantic_params.json"))

    if args.method_name is not None:
        method_name = args.method_name
    elif os.getenv("METHOD_NAME") is not None:
        method_name = os.getenv("METHOD_NAME")
    elif args.semantic_param_path is not None:
        method_name = f"llm_cc_mcts_{semantic_param_path.stem}"
    else:
        method_name = "llm_cc_mcts"

    return semantic_param_path, method_name


def main():
    global EXPLORATION_C

    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_scenarios:
        print_available_scenarios()
        return

    if args.list_datasets:
        print_dataset_registry()
        return

    if args.list_llm_providers:
        print_available_providers()
        return

    semantic_param_path, method_name = resolve_semantic_run(args, parser)
    schema_path = args.schema_path
    data_dir = args.data_dir

    if args.dataset is not None:
        dataset_config = resolve_dataset(args.dataset)
        schema_path = dataset_config.schema_path
        data_dir = dataset_config.data_dir

        if args.method_name is None:
            method_name = f"{method_name}_{args.dataset}"

    EXPLORATION_C = args.exploration_c
    configure_result_dir(args.result_dir)
    configure_dataset(schema_path=schema_path, data_dir=data_dir)
    configure_semantic_run(
        semantic_param_path=semantic_param_path,
        method_name=method_name,
    )
    run_llm_cc_mcts(
        max_real_steps=args.max_real_steps,
        mcts_iterations=args.mcts_iterations,
        rolling_horizon=args.rolling_horizon,
    )


if __name__ == "__main__":
    main()
