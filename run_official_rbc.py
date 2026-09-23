from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd

from citylearn.agents.rbc import BasicBatteryRBC, BasicRBC, OptimizedRBC
from citylearn.citylearn import CityLearnEnv
from experiment_config import print_dataset_registry, resolve_dataset


RESULT_DIR = Path("results")
DEFAULT_DATASET = "cl_2022_p1"

CONTROLLERS = {
    "basic": BasicRBC,
    "optimized": OptimizedRBC,
    "battery": BasicBatteryRBC,
}


def parse_optional_int(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "null", "all", "full", "-1"}:
        return None
    return int(text)


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


def summarize_action(actions):
    flat = np.asarray(actions, dtype=float).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(np.mean(flat))


def run_official_rbc(
    dataset_id,
    controller_name,
    max_real_steps,
    result_dir,
    method_name,
):
    dataset = resolve_dataset(dataset_id)
    result_dir.mkdir(parents=True, exist_ok=True)

    controller_cls = CONTROLLERS[controller_name]
    env = CityLearnEnv(str(dataset.schema_path))
    observations = reset_env(env)
    agent = controller_cls(env)

    print(f"\n========== 开始运行 {method_name} ==========")
    print("dataset =", dataset_id)
    print("schema 路径：", dataset.schema_path)
    print("controller =", controller_cls.__name__)
    print("MAX_REAL_STEPS =", max_real_steps)

    done = False
    time_step = 0
    records = []
    start_time = time.time()

    while not done:
        if max_real_steps is not None and time_step >= max_real_steps:
            print(f"\n短测试达到 {max_real_steps} 步，提前停止。")
            break

        actions = agent.predict(observations, deterministic=True)
        observations, rewards, done, info = step_env(env, actions)

        records.append(
            {
                "time_step": time_step,
                "action_value": summarize_action(actions),
            }
        )

        time_step += 1
        if time_step % 1000 == 0:
            print(f"{method_name} 已运行到第 {time_step} 步")

    total_time = time.time() - start_time
    actions_path = result_dir / f"{method_name}_actions_debug.csv"
    pd.DataFrame(records).to_csv(actions_path, index=False)

    print("\n========== official RBC 运行结束 ==========")
    print("实际运行步数：", time_step)
    print("总耗时：", total_time, "秒")
    print("动作记录保存到：", actions_path)

    if done and max_real_steps is None:
        kpis = env.evaluate()
        kpis["method"] = method_name
        kpis["dataset_id"] = dataset_id
        kpis["schema_path"] = str(dataset.schema_path)
        kpis["controller"] = controller_cls.__name__

        kpi_path = result_dir / f"{method_name}_kpis.csv"
        kpis.to_csv(kpi_path, index=False)
        print("全年仿真完成，KPI 保存到：", kpi_path)
    else:
        print("\n注意：当前是短测试，不生成正式 KPI。")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="运行 CityLearn 官方 RBC baseline。"
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
        "--controller",
        choices=sorted(CONTROLLERS),
        default="optimized",
        help="官方 RBC 控制器类型。",
    )
    parser.add_argument(
        "--max-real-steps",
        type=parse_optional_int,
        default=300,
        help="真实环境运行步数；None/all/full 表示跑完整 episode。",
    )
    parser.add_argument(
        "--method-name",
        default=None,
        help="输出结果中的方法名；不填则自动生成。",
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
        if args.dataset == DEFAULT_DATASET and args.controller == "optimized":
            method_name = "official_rbc"
        else:
            method_name = f"official_rbc_{args.controller}_{args.dataset}"

    run_official_rbc(
        dataset_id=args.dataset,
        controller_name=args.controller,
        max_real_steps=args.max_real_steps,
        result_dir=args.result_dir,
        method_name=method_name,
    )


if __name__ == "__main__":
    main()
