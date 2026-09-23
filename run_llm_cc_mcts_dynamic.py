from pathlib import Path
import argparse
import time

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
import run_llm_cc_mcts as base


DEFAULT_METHOD_NAME = "llm_cc_mcts_dynamic_switching"
DEFAULT_SCHEDULE = "0:cost_priority,3000:low_carbon_priority,6000:peak_aware"

LLM_PROVIDER = None
MODEL_NAME = None
SEMANTIC_PARAM_ROOT = Path("semantic_params")


def parse_optional_int(value):
    if value is None:
        return None

    value = str(value).strip()
    if value.lower() in {"none", "null", "full", "all"}:
        return None

    return int(value)


def parse_schedule(schedule_text):
    schedule = []

    for item in schedule_text.split(","):
        item = item.strip()
        if not item:
            continue

        try:
            step_text, scenario_name = item.split(":", maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                "schedule 格式应为 step:scenario，例如 "
                "0:cost_priority,3000:low_carbon_priority"
            ) from exc

        step = int(step_text)
        scenario_name = scenario_name.strip()

        if scenario_name not in base.SEMANTIC_SCENARIOS:
            raise ValueError(
                f"未知语义场景：{scenario_name}，可选值："
                f"{', '.join(sorted(base.SEMANTIC_SCENARIOS))}"
            )

        schedule.append((step, scenario_name))

    if not schedule:
        raise ValueError("schedule 不能为空。")

    schedule = sorted(schedule, key=lambda x: x[0])

    if schedule[0][0] != 0:
        raise ValueError("schedule 必须从 0 步开始，例如 0:cost_priority。")

    for i in range(1, len(schedule)):
        if schedule[i][0] <= schedule[i - 1][0]:
            raise ValueError("schedule 中的切换步数必须严格递增。")

    return schedule


def format_schedule(schedule):
    return ",".join(f"{step}:{scenario}" for step, scenario in schedule)


def configure_scenario(scenario_key, method_name):
    config = base.SEMANTIC_SCENARIOS[scenario_key]
    if LLM_PROVIDER is not None:
        semantic_param_path = provider_semantic_param_path(
            provider_name=LLM_PROVIDER,
            model_name=MODEL_NAME,
            scenario_name=scenario_key,
            output_root=SEMANTIC_PARAM_ROOT,
        )
    else:
        semantic_param_path = config["param_path"]

    base.configure_semantic_run(
        semantic_param_path=semantic_param_path,
        method_name=method_name,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="运行同一 episode 内动态切换语义目标的 LLM-guided MCTS 实验。"
    )
    parser.add_argument(
        "--schedule",
        default=DEFAULT_SCHEDULE,
        help=(
            "语义切换计划，格式 step:scenario,step:scenario。"
            f"默认：{DEFAULT_SCHEDULE}"
        ),
    )
    parser.add_argument(
        "--method-name",
        default=DEFAULT_METHOD_NAME,
        help="输出文件使用的方法名。",
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
        default=None,
        help="选择 configs/datasets.json 中注册的数据集 ID。",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=base.DEFAULT_SCHEMA_PATH,
        help="CityLearn schema.json 路径。",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="包含 pricing.csv 和 carbon_intensity.csv 的目录；默认使用 schema 所在目录。",
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
        "--result-dir",
        type=Path,
        default=base.DEFAULT_RESULT_DIR,
        help="结果输出目录。",
    )
    parser.add_argument(
        "--max-real-steps",
        type=parse_optional_int,
        default=None,
        help="真实环境运行步数；None/full/all 表示跑完整 episode。",
    )
    parser.add_argument(
        "--mcts-iterations",
        type=int,
        default=base.MCTS_ITERATIONS,
        help="每步 MCTS 迭代次数。",
    )
    parser.add_argument(
        "--rolling-horizon",
        type=int,
        default=base.ROLLING_HORIZON,
        help="MCTS 滚动规划 horizon。",
    )
    parser.add_argument(
        "--exploration-c",
        type=float,
        default=base.EXPLORATION_C,
        help="UCT exploration 系数。",
    )
    return parser


def run_dynamic_experiment(
    schedule,
    method_name,
    max_real_steps=None,
    mcts_iterations=base.MCTS_ITERATIONS,
    rolling_horizon=base.ROLLING_HORIZON,
    seed=None,
):
    if seed is not None:
        base.random.seed(seed)
        base.np.random.seed(seed)
        if base.torch is not None:
            base.torch.manual_seed(seed)
    schedule_text = format_schedule(schedule)
    current_schedule_index = 0
    current_scenario = schedule[current_schedule_index][1]
    configure_scenario(current_scenario, method_name)

    print(f"\n========== 开始运行 {method_name} ==========")
    print("动态语义切换计划：", schedule_text)
    print("schema 路径：", base.SCHEMA_PATH)
    print("data 目录：", base.DATA_DIR)
    print("MAX_REAL_STEPS =", max_real_steps)
    print("MCTS_ITERATIONS =", mcts_iterations)
    print("ROLLING_HORIZON =", rolling_horizon)
    print("EXPLORATION_C =", base.EXPLORATION_C)

    available_steps = len(base.pricing)
    skipped_switches = [
        (step, scenario) for step, scenario in schedule if step >= available_steps
    ]
    if skipped_switches:
        print(
            "\n注意：当前数据集 pricing/carbon 序列长度为 "
            f"{available_steps}，以下切换点不会触发："
        )
        for step, scenario in skipped_switches:
            print(f"- step={step}, scenario={scenario}")
        print(
            "如果要在该数据集上测试动态切换，请用 --schedule 指定更短的切换计划。"
        )

    env = CityLearnEnv(str(base.SCHEMA_PATH), random_seed=seed)
    base.reset_env(env)

    done = False
    time_step = 0
    previous_action = 0.0
    records = []
    switch_records = []
    start_time = time.time()

    print(
        f"\n[语义切换] step=0, scenario={current_scenario}, "
        f"semantic_param_path={base.SEMANTIC_PARAM_PATH}"
    )
    switch_records.append(
        {
            "time_step": 0,
            "semantic_scenario_key": current_scenario,
            "semantic_param_path": str(base.SEMANTIC_PARAM_PATH),
            "scenario_name": base.semantic_params.get("scenario_name"),
            "price_weight": base.PRICE_WEIGHT,
            "carbon_weight": base.CARBON_WEIGHT,
            "peak_charge_penalty_weight": base.PEAK_CHARGE_PENALTY_WEIGHT,
        }
    )

    while not done:
        if max_real_steps is not None and time_step >= max_real_steps:
            print(f"\n短测试达到 {max_real_steps} 步，提前停止。")
            break

        next_schedule_index = current_schedule_index + 1
        if (
            next_schedule_index < len(schedule)
            and time_step >= schedule[next_schedule_index][0]
        ):
            current_schedule_index = next_schedule_index
            current_scenario = schedule[current_schedule_index][1]
            configure_scenario(current_scenario, method_name)
            print(
                f"\n[语义切换] step={time_step}, "
                f"scenario={current_scenario}, "
                f"semantic_param_path={base.SEMANTIC_PARAM_PATH}"
            )
            switch_records.append(
                {
                    "time_step": time_step,
                    "semantic_scenario_key": current_scenario,
                    "semantic_param_path": str(base.SEMANTIC_PARAM_PATH),
                    "scenario_name": base.semantic_params.get("scenario_name"),
                    "price_weight": base.PRICE_WEIGHT,
                    "carbon_weight": base.CARBON_WEIGHT,
                    "peak_charge_penalty_weight": (
                        base.PEAK_CHARGE_PENALTY_WEIGHT
                    ),
                }
            )

        plan_start = time.time()
        action_value = base.llm_mcts_plan_action(
            env=env,
            current_time_step=time_step,
            previous_action=previous_action,
            horizon=rolling_horizon,
            iterations=mcts_iterations,
        )
        planning_time = time.time() - plan_start

        actions = base.constant_action(env.action_space, action_value)
        _, rewards, done, _ = base.step_env(env, actions)

        reward_value = base.reward_to_float(rewards)
        score = base.get_score(time_step)
        prior_action = base.carbon_aware_prior_action(time_step)

        records.append(
            {
                "time_step": time_step,
                "semantic_scenario_key": current_scenario,
                "semantic_param_path": str(base.SEMANTIC_PARAM_PATH),
                "scenario_name": base.semantic_params.get("scenario_name"),
                "action_value": action_value,
                "prior_action": prior_action,
                "score": score,
                "env_reward": reward_value,
                "planning_time_seconds": planning_time,
                "method": method_name,
                "price_weight": base.PRICE_WEIGHT,
                "carbon_weight": base.CARBON_WEIGHT,
                "low_q": base.LOW_Q,
                "high_q": base.HIGH_Q,
                "smooth_weight": base.SMOOTH_WEIGHT,
                "peak_charge_penalty_weight": base.PEAK_CHARGE_PENALTY_WEIGHT,
            }
        )

        previous_action = action_value
        time_step += 1

        if time_step % 500 == 0:
            print(
                f"{method_name} 已运行到第 {time_step} 步，"
                f"当前语义={current_scenario}, "
                f"动作={action_value}, "
                f"prior={prior_action}, "
                f"score={score:.4f}, "
                f"单步规划耗时={planning_time:.3f}s"
            )

    total_time = time.time() - start_time

    action_df = pd.DataFrame(records)
    action_path = base.RESULT_DIR / f"{method_name}_actions_debug.csv"
    action_df.to_csv(action_path, index=False)

    switch_df = pd.DataFrame(switch_records)
    switch_path = base.RESULT_DIR / f"{method_name}_switch_log.csv"
    switch_df.to_csv(switch_path, index=False)

    print(f"\n========== {method_name} 运行结束 ==========")
    print("实际运行步数：", time_step)
    print("总耗时：", total_time, "秒")
    print("动作记录保存到：", action_path)
    print("切换记录保存到：", switch_path)

    if done and max_real_steps is None:
        kpis = env.evaluate()
        kpis["method"] = method_name
        kpis["dynamic_schedule"] = schedule_text
        kpis["semantic_param_path"] = "dynamic_switching"
        kpis["scenario_name"] = "dynamic_switching"

        kpi_path = base.RESULT_DIR / f"{method_name}_kpis.csv"
        kpis.to_csv(kpi_path, index=False)
        print("全年仿真完成，KPI 保存到：", kpi_path)
        print(kpis)
    else:
        print("\n注意：当前是短测试，不生成正式 KPI。")
        print("确认无报错后，用 --max-real-steps None 再跑全年。")

    return env


def main():
    global LLM_PROVIDER
    global MODEL_NAME
    global SEMANTIC_PARAM_ROOT

    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_datasets:
        print_dataset_registry()
        return

    if args.list_llm_providers:
        print_available_providers()
        return

    schedule = parse_schedule(args.schedule)
    schema_path = args.schema_path
    data_dir = args.data_dir
    method_name = args.method_name

    LLM_PROVIDER = args.llm_provider
    MODEL_NAME = (
        args.model
        if args.model is not None
        else get_default_model(args.llm_provider)
        if args.llm_provider is not None
        else None
    )
    SEMANTIC_PARAM_ROOT = args.semantic_param_root

    if args.llm_provider is not None and args.method_name == DEFAULT_METHOD_NAME:
        method_name = (
            f"{method_name}_{args.llm_provider}_{slugify_model_name(MODEL_NAME)}"
        )

    if args.dataset is not None:
        dataset_config = resolve_dataset(args.dataset)
        schema_path = dataset_config.schema_path
        data_dir = dataset_config.data_dir

        if args.method_name == DEFAULT_METHOD_NAME:
            method_name = f"{method_name}_{args.dataset}"

    base.EXPLORATION_C = args.exploration_c
    base.configure_result_dir(args.result_dir)
    base.configure_dataset(schema_path=schema_path, data_dir=data_dir)

    run_dynamic_experiment(
        schedule=schedule,
        method_name=method_name,
        max_real_steps=args.max_real_steps,
        mcts_iterations=args.mcts_iterations,
        rolling_horizon=args.rolling_horizon,
    )


if __name__ == "__main__":
    main()
