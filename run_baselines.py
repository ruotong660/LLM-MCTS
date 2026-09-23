from pathlib import Path
import numpy as np
import pandas as pd

from citylearn.citylearn import CityLearnEnv


SCHEMA_PATH = Path("data/datasets/citylearn_challenge_2022_phase_1/schema.json")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)


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


def constant_action(action_space, value):
    """
    给所有建筑生成相同动作。
    action > 0 通常表示充电
    action < 0 通常表示放电
    action = 0 表示不动作
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


def no_control_policy(env, time_step):
    """
    No Control:
    不主动控制储能，所有动作为 0。
    """
    return constant_action(env.action_space, 0.0)


def random_policy(env, time_step):
    """
    Random Agent:
    随机动作。
    """
    if isinstance(env.action_space, list):
        return [space.sample() for space in env.action_space]
    return env.action_space.sample()


def simple_rbc_policy(env, time_step):
    """
    Simple RBC:
    一个简单规则控制策略。

    0-6 点：低谷时段，充电
    10-15 点：白天可能有光伏，适当充电
    17-21 点：晚高峰，放电
    其他时间：不动作
    """
    hour = time_step % 24

    if 0 <= hour <= 6:
        action_value = 0.15
    elif 10 <= hour <= 15:
        action_value = 0.25
    elif 17 <= hour <= 21:
        action_value = -0.25
    else:
        action_value = 0.0

    return constant_action(env.action_space, action_value)


def run_experiment(method_name, policy_fn):
    print(f"\n========== 开始运行 {method_name} ==========")

    env = CityLearnEnv(str(SCHEMA_PATH))
    observations = reset_env(env)

    done = False
    time_step = 0

    while not done:
        actions = policy_fn(env, time_step)
        observations, rewards, done, info = step_env(env, actions)

        time_step += 1

        if time_step % 1000 == 0:
            print(f"{method_name} 已经运行到第 {time_step} 个时间步")

    print(f"{method_name} 仿真结束，共运行 {time_step} 个时间步")

    kpis = env.evaluate()
    kpis["method"] = method_name

    output_path = RESULT_DIR / f"{method_name}_kpis.csv"
    kpis.to_csv(output_path, index=False)

    print(f"{method_name} 结果已保存到：{output_path}")

    return kpis


if __name__ == "__main__":
    print("schema 文件是否存在：", SCHEMA_PATH.exists())
    print("schema 路径：", SCHEMA_PATH)

    all_results = []

    all_results.append(run_experiment("no_control", no_control_policy))
    all_results.append(run_experiment("random_agent", random_policy))
    all_results.append(run_experiment("simple_rbc", simple_rbc_policy))

    all_kpis = pd.concat(all_results, ignore_index=True)
    all_kpis.to_csv(RESULT_DIR / "all_baseline_kpis.csv", index=False)

    district_summary = all_kpis[
        (all_kpis["level"] == "district")
        & (
            all_kpis["cost_function"].isin(
                [
                    "cost_total",
                    "carbon_emissions_total",
                    "all_time_peak_average",
                    "daily_one_minus_load_factor_average",
                    "annual_normalized_unserved_energy_total",
                ]
            )
        )
    ]

    district_summary = district_summary[
        ["method", "cost_function", "value", "name", "level"]
    ]

    district_summary.to_csv(RESULT_DIR / "district_summary.csv", index=False)

    print("\n========== District 层面对比结果 ==========")
    print(district_summary)

    print("\n所有 baseline 实验完成！")
    print("完整结果保存到：results/all_baseline_kpis.csv")
    print("核心汇总保存到：results/district_summary.csv")