from pathlib import Path
import argparse
import copy
import itertools
import time

import numpy as np
import pandas as pd

from citylearn.citylearn import CityLearnEnv
from experiment_config import print_dataset_registry, resolve_dataset


RESULT_DIR = Path("results")
DEFAULT_DATASET = "cl_2022_p1"

DEFAULT_ACTION_VALUES = [-0.1, 0.0, 0.1]
DEFAULT_HORIZON = 4
DEFAULT_BEAM_WIDTH = 5

PRICE_WEIGHT = 0.5
CARBON_WEIGHT = 0.5
LOW_Q = 0.2
HIGH_Q = 0.8
ALIGNMENT_WEIGHT = 8.0
SMOOTH_WEIGHT = 0.3
PEAK_CHARGE_PENALTY_WEIGHT = 1.0


def parse_optional_int(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "null", "all", "full", "-1"}:
        return None
    return int(text)


def parse_action_values(value):
    if isinstance(value, list):
        return [float(item) for item in value]

    values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise ValueError("action-values 不能为空。")
    return values


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
    try:
        import torch
    except ImportError:
        torch = None

    if torch is None:
        result = env.step(actions)
    else:
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
        raise ValueError(f"{csv_path} 中没有找到数值列。")
    return df[numeric_columns[0]].astype(float).to_numpy()


def normalize_array(values):
    values = np.asarray(values, dtype=float)
    min_value = np.nanmin(values)
    max_value = np.nanmax(values)
    if abs(max_value - min_value) < 1e-9:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def is_peak_hour(time_step):
    hour = time_step % 24
    return 17 <= hour <= 21


class RollingOptimizationPolicy:
    def __init__(
        self,
        data_dir,
        action_values,
        horizon,
        beam_width,
        exact,
    ):
        pricing = load_first_numeric_column(data_dir / "pricing.csv")
        carbon_intensity = load_first_numeric_column(data_dir / "carbon_intensity.csv")

        pricing_norm = normalize_array(pricing)
        carbon_norm = normalize_array(carbon_intensity)

        self.score_series = PRICE_WEIGHT * pricing_norm + CARBON_WEIGHT * carbon_norm
        self.low_threshold = float(np.quantile(self.score_series, LOW_Q))
        self.high_threshold = float(np.quantile(self.score_series, HIGH_Q))
        self.action_values = action_values
        self.horizon = horizon
        self.beam_width = beam_width
        self.exact = exact

    def get_score(self, time_step):
        idx = time_step % len(self.score_series)
        return float(self.score_series[idx])

    def prior_action(self, time_step):
        score = self.get_score(time_step)
        if score <= self.low_threshold:
            return max(self.action_values)
        if score >= self.high_threshold:
            return min(self.action_values)
        return 0.0

    def shaped_reward(self, env_rewards, action_value, previous_action, time_step):
        score = self.get_score(time_step)

        alignment_reward = -ALIGNMENT_WEIGHT * (score - 0.5) * action_value
        smooth_penalty = -SMOOTH_WEIGHT * abs(action_value - previous_action)

        peak_penalty = 0.0
        if is_peak_hour(time_step) and action_value > 0:
            peak_penalty = -PEAK_CHARGE_PENALTY_WEIGHT * action_value

        env_reward = reward_to_float(env_rewards)
        env_reward_component = 0.02 * np.tanh(env_reward / 10.0)

        return float(
            alignment_reward
            + smooth_penalty
            + peak_penalty
            + env_reward_component
        )

    def ordered_actions(self, time_step):
        prior = self.prior_action(time_step)
        return sorted(self.action_values, key=lambda action: abs(action - prior))

    def plan_action(self, env, current_time_step, previous_action):
        if self.exact:
            return self._exact_plan_action(env, current_time_step, previous_action)
        return self._beam_plan_action(env, current_time_step, previous_action)

    def _exact_plan_action(self, env, current_time_step, previous_action):
        best_value = -float("inf")
        best_first_action = 0.0

        action_grid = [self.ordered_actions(current_time_step + i) for i in range(self.horizon)]
        for sequence in itertools.product(*action_grid):
            sim_env = copy.deepcopy(env)
            value = 0.0
            done = False
            prev = previous_action

            for depth, action_value in enumerate(sequence):
                if done:
                    break
                actions = constant_action(sim_env.action_space, action_value)
                _, rewards, done, _ = step_env(sim_env, actions)
                value += self.shaped_reward(
                    env_rewards=rewards,
                    action_value=action_value,
                    previous_action=prev,
                    time_step=current_time_step + depth,
                )
                prev = action_value

            if value > best_value:
                best_value = value
                best_first_action = sequence[0]

        return best_first_action, best_value, len(self.action_values) ** self.horizon

    def _beam_plan_action(self, env, current_time_step, previous_action):
        root_env = copy.deepcopy(env)
        beam = [
            {
                "env": root_env,
                "value": 0.0,
                "sequence": [],
                "previous_action": previous_action,
                "done": False,
            }
        ]
        evaluated_nodes = 0

        for depth in range(self.horizon):
            candidates = []
            time_step = current_time_step + depth

            for node in beam:
                if node["done"]:
                    candidates.append(node)
                    continue

                for action_value in self.ordered_actions(time_step):
                    sim_env = copy.deepcopy(node["env"])
                    actions = constant_action(sim_env.action_space, action_value)
                    _, rewards, done, _ = step_env(sim_env, actions)
                    evaluated_nodes += 1

                    reward = self.shaped_reward(
                        env_rewards=rewards,
                        action_value=action_value,
                        previous_action=node["previous_action"],
                        time_step=time_step,
                    )
                    candidates.append(
                        {
                            "env": sim_env,
                            "value": node["value"] + reward,
                            "sequence": node["sequence"] + [action_value],
                            "previous_action": action_value,
                            "done": done,
                        }
                    )

            candidates.sort(key=lambda item: item["value"], reverse=True)
            beam = candidates[: self.beam_width]

        best = max(beam, key=lambda item: item["value"])
        first_action = best["sequence"][0] if best["sequence"] else 0.0
        return first_action, float(best["value"]), evaluated_nodes


def run_mpc_baseline(
    dataset_id,
    max_real_steps,
    result_dir,
    method_name,
    action_values,
    horizon,
    beam_width,
    exact,
):
    dataset = resolve_dataset(dataset_id)
    result_dir.mkdir(parents=True, exist_ok=True)

    policy = RollingOptimizationPolicy(
        data_dir=dataset.data_dir,
        action_values=action_values,
        horizon=horizon,
        beam_width=beam_width,
        exact=exact,
    )

    env = CityLearnEnv(str(dataset.schema_path))
    reset_env(env)

    print(f"\n========== 开始运行 {method_name} ==========")
    print("dataset =", dataset_id)
    print("schema 路径：", dataset.schema_path)
    print("MAX_REAL_STEPS =", max_real_steps)
    print("ACTION_VALUES =", action_values)
    print("HORIZON =", horizon)
    print("BEAM_WIDTH =", beam_width)
    print("EXACT =", exact)
    print("low_threshold =", policy.low_threshold)
    print("high_threshold =", policy.high_threshold)

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
        action_value, planned_value, evaluated_nodes = policy.plan_action(
            env=env,
            current_time_step=time_step,
            previous_action=previous_action,
        )
        planning_time = time.time() - plan_start

        actions = constant_action(env.action_space, action_value)
        observations, rewards, done, info = step_env(env, actions)

        records.append(
            {
                "time_step": time_step,
                "action_value": action_value,
                "prior_action": policy.prior_action(time_step),
                "score": policy.get_score(time_step),
                "planned_value": planned_value,
                "evaluated_nodes": evaluated_nodes,
                "env_reward": reward_to_float(rewards),
                "planning_time_seconds": planning_time,
            }
        )

        previous_action = action_value
        time_step += 1

        if time_step % 500 == 0:
            print(
                f"{method_name} 已运行到第 {time_step} 步，"
                f"当前动作={action_value}, "
                f"规划值={planned_value:.4f}, "
                f"候选节点={evaluated_nodes}, "
                f"单步规划耗时={planning_time:.3f}s"
            )

    total_time = time.time() - start_time
    action_path = result_dir / f"{method_name}_actions_debug.csv"
    pd.DataFrame(records).to_csv(action_path, index=False)

    print("\n========== rolling optimization 运行结束 ==========")
    print("实际运行步数：", time_step)
    print("总耗时：", total_time, "秒")
    print("动作记录保存到：", action_path)

    if done and max_real_steps is None:
        kpis = env.evaluate()
        kpis["method"] = method_name
        kpis["dataset_id"] = dataset_id
        kpis["schema_path"] = str(dataset.schema_path)
        kpis["horizon"] = horizon
        kpis["beam_width"] = beam_width
        kpis["exact"] = exact

        kpi_path = result_dir / f"{method_name}_kpis.csv"
        kpis.to_csv(kpi_path, index=False)

        print("全年仿真完成，KPI 保存到：", kpi_path)
    else:
        print("\n注意：当前是短测试，不生成正式 KPI。")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="运行滚动优化/MPC baseline，用确定性滚动搜索对比 MCTS。"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="列出可用数据集后退出。",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="数据集 ID。",
    )
    parser.add_argument(
        "--max-real-steps",
        type=parse_optional_int,
        default=100,
        help="真实环境运行步数；None/all/full 表示跑完整 episode。",
    )
    parser.add_argument(
        "--method-name",
        default=None,
        help="输出结果中的方法名；不填则自动生成。",
    )
    parser.add_argument(
        "--action-values",
        type=parse_action_values,
        default=DEFAULT_ACTION_VALUES,
        help="逗号分隔动作集合，例如 -0.1,0,0.1。",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="滚动优化预测窗口。",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=DEFAULT_BEAM_WIDTH,
        help="beam search 保留的候选序列数。",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="枚举全部动作序列；短测试可用，全年可能很慢。",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=RESULT_DIR,
        help="结果输出目录。",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_datasets:
        print_dataset_registry()
        return

    method_name = args.method_name
    if method_name is None:
        if args.dataset == DEFAULT_DATASET:
            method_name = "rolling_mpc"
        else:
            method_name = f"rolling_mpc_{args.dataset}"

    run_mpc_baseline(
        dataset_id=args.dataset,
        max_real_steps=args.max_real_steps,
        result_dir=args.result_dir,
        method_name=method_name,
        action_values=args.action_values,
        horizon=args.horizon,
        beam_width=args.beam_width,
        exact=args.exact,
    )


if __name__ == "__main__":
    main()
