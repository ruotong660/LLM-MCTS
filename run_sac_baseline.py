from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd

from citylearn.agents.rbc import OptimizedRBC
from citylearn.agents.sac import SAC, SACRBC
from citylearn.citylearn import CityLearnEnv
from experiment_config import print_dataset_registry, resolve_dataset


RESULT_DIR = Path("results")
DEFAULT_DATASET = "cl_2022_p1"


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
        # CityLearn 2023 local dynamics may use torch models. Keeping env.step
        # under no_grad avoids retaining computation graphs during rollouts.
        with torch.no_grad():
            result = env.step(actions)

    if len(result) == 5:
        observations, rewards, terminated, truncated, info = result
        done = convert_done(terminated) or convert_done(truncated)
    else:
        observations, rewards, done, info = result
        done = convert_done(done)
        terminated = done
        truncated = False

    return observations, rewards, bool(terminated), bool(truncated), done, info


def reward_to_float(rewards):
    arr = np.asarray(rewards, dtype=float)
    return float(np.sum(arr))


def summarize_action(actions):
    flat = np.asarray(actions, dtype=float).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(np.mean(flat))


def build_sac_agent(env, args):
    common_kwargs = {
        "hidden_dimension": [args.hidden_dim, args.hidden_dim],
        "discount": args.discount,
        "tau": args.tau,
        "alpha": args.alpha,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "replay_buffer_capacity": args.replay_buffer_capacity,
        "standardize_start_time_step": args.standardize_start,
        "end_exploration_time_step": args.end_exploration,
        "action_scaling_coefficienct": args.action_scaling,
        "reward_scaling": args.reward_scaling,
        "update_per_time_step": args.update_per_time_step,
    }

    if args.agent == "sac_rbc":
        return SACRBC(env, rbc=OptimizedRBC, **common_kwargs)

    return SAC(env, **common_kwargs)


def train_agent(env, agent, args):
    total_steps = 0
    train_records = []
    start_time = time.time()

    print("\n========== 开始训练 SAC baseline ==========")
    print("agent =", args.agent)
    print("train_episodes =", args.train_episodes)
    print("max_train_steps =", args.max_train_steps)
    print("batch_size =", args.batch_size)
    print("standardize_start =", args.standardize_start)
    print("end_exploration =", args.end_exploration)
    print("update_per_time_step =", args.update_per_time_step)

    for episode in range(args.train_episodes):
        observations = reset_env(env)
        terminated = False
        truncated = False
        done = False
        episode_step = 0
        episode_reward = 0.0

        while not done:
            if args.max_train_steps is not None and total_steps >= args.max_train_steps:
                break

            actions = agent.predict(observations, deterministic=False)
            next_observations, rewards, terminated, truncated, done, info = step_env(
                env, actions
            )
            agent.update(
                observations,
                actions,
                rewards,
                next_observations,
                terminated=terminated,
                truncated=truncated,
            )

            reward_value = reward_to_float(rewards)
            episode_reward += reward_value
            train_records.append(
                {
                    "phase": "train",
                    "episode": episode,
                    "episode_step": episode_step,
                    "global_step": total_steps,
                    "action_value": summarize_action(actions),
                    "reward": reward_value,
                }
            )

            observations = next_observations
            total_steps += 1
            episode_step += 1

            if total_steps % args.log_interval == 0:
                print(
                    f"训练步数={total_steps}, episode={episode + 1}, "
                    f"当前 reward={reward_value:.4f}"
                )

        print(
            f"episode {episode + 1}/{args.train_episodes} 训练结束，"
            f"步数={episode_step}, 累计 reward={episode_reward:.4f}"
        )

        if args.max_train_steps is not None and total_steps >= args.max_train_steps:
            break

    total_time = time.time() - start_time
    print("SAC 训练总步数：", total_steps)
    print("SAC 训练总耗时：", total_time, "秒")
    return train_records


def evaluate_agent(env, agent, args, method_name, result_dir, dataset):
    observations = reset_env(env)
    agent.reset()

    done = False
    time_step = 0
    records = []
    start_time = time.time()

    deterministic = args.eval_deterministic
    if deterministic and not all(agent.normalized):
        deterministic = False
        print(
            "注意：SAC 尚未完成 observation/reward 标准化，"
            "本次评测退回 stochastic/exploration 动作。"
        )

    print("\n========== 开始评测 SAC baseline ==========")
    print("method =", method_name)
    print("eval_deterministic =", deterministic)
    print("max_eval_steps =", args.max_eval_steps)

    while not done:
        if args.max_eval_steps is not None and time_step >= args.max_eval_steps:
            print(f"\n评测短测试达到 {args.max_eval_steps} 步，提前停止。")
            break

        plan_start = time.time()
        actions = agent.predict(observations, deterministic=deterministic)
        planning_time = time.time() - plan_start

        observations, rewards, terminated, truncated, done, info = step_env(env, actions)
        reward_value = reward_to_float(rewards)

        records.append(
            {
                "phase": "eval",
                "time_step": time_step,
                "action_value": summarize_action(actions),
                "reward": reward_value,
                "planning_time_seconds": planning_time,
            }
        )

        time_step += 1
        if time_step % args.log_interval == 0:
            print(
                f"评测步数={time_step}, "
                f"当前动作={records[-1]['action_value']:.4f}, "
                f"reward={reward_value:.4f}"
            )

    total_time = time.time() - start_time
    action_path = result_dir / f"{method_name}_actions_debug.csv"
    pd.DataFrame(records).to_csv(action_path, index=False)

    print("\n========== SAC baseline 评测结束 ==========")
    print("实际评测步数：", time_step)
    print("评测总耗时：", total_time, "秒")
    print("动作记录保存到：", action_path)

    if done and args.max_eval_steps is None:
        kpis = env.evaluate()
        kpis["method"] = method_name
        kpis["dataset_id"] = dataset.dataset_id
        kpis["schema_path"] = str(dataset.schema_path)
        kpis["agent"] = args.agent
        kpis["train_episodes"] = args.train_episodes
        kpis["max_train_steps"] = args.max_train_steps

        kpi_path = result_dir / f"{method_name}_kpis.csv"
        kpis.to_csv(kpi_path, index=False)
        print("全年评测完成，KPI 保存到：", kpi_path)
    else:
        print("\n注意：当前是短评测，不生成正式 KPI。")


def run_sac_baseline(args):
    dataset = resolve_dataset(args.dataset)
    result_dir = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("SAC baseline 需要 torch。") from exc

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    method_name = args.method_name
    if method_name is None:
        if args.dataset == DEFAULT_DATASET and args.agent == "sac_rbc":
            method_name = "sac_baseline"
        else:
            method_name = f"{args.agent}_{args.dataset}"

    print("dataset =", args.dataset)
    print("schema 路径：", dataset.schema_path)
    print("method =", method_name)

    env = CityLearnEnv(str(dataset.schema_path))
    agent = build_sac_agent(env, args)

    train_records = train_agent(env, agent, args)
    train_path = result_dir / f"{method_name}_train_debug.csv"
    pd.DataFrame(train_records).to_csv(train_path, index=False)
    print("训练记录保存到：", train_path)

    evaluate_agent(
        env=env,
        agent=agent,
        args=args,
        method_name=method_name,
        result_dir=result_dir,
        dataset=dataset,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="运行 CityLearn SAC/SACRBC 学习型 baseline。"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="列出可用数据集后退出。",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="数据集 ID。")
    parser.add_argument(
        "--agent",
        choices=["sac", "sac_rbc"],
        default="sac_rbc",
        help="SAC 变体；sac_rbc 使用 OptimizedRBC 作为探索策略。",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=1,
        help="训练 episode 数。",
    )
    parser.add_argument(
        "--max-train-steps",
        type=parse_optional_int,
        default=300,
        help="最大训练步数；None/all/full 表示按 episode 跑完。",
    )
    parser.add_argument(
        "--max-eval-steps",
        type=parse_optional_int,
        default=200,
        help="最大评测步数；None/all/full 表示跑完整 episode 并生成 KPI。",
    )
    parser.add_argument(
        "--eval-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="评测时是否使用确定性策略。",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-buffer-capacity", type=int, default=50000)
    parser.add_argument("--standardize-start", type=int, default=128)
    parser.add_argument("--end-exploration", type=int, default=2000)
    parser.add_argument("--action-scaling", type=float, default=0.3)
    parser.add_argument("--reward-scaling", type=float, default=5.0)
    parser.add_argument("--update-per-time-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_datasets:
        print_dataset_registry()
        return

    run_sac_baseline(args)


if __name__ == "__main__":
    main()
