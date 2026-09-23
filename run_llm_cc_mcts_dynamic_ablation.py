import argparse
import copy
import random

from experiment_config import resolve_dataset
from llm_providers import (
    get_default_model,
    semantic_param_path as provider_semantic_param_path,
    slugify_model_name,
)
import run_llm_cc_mcts as base
import run_llm_cc_mcts_dynamic as dynamic


ABLATION_CHOICES = [
    "no_prior",
    "no_action_sort",
    "no_peak_penalty",
    "no_smooth",
]

ABLATION_DESCRIPTIONS = {
    "no_prior": (
        "去掉语义动作先验：关闭 policy prior reward，"
        "rollout 不再优先采用 prior action，候选动作也不按 prior 排序。"
    ),
    "no_action_sort": "去掉候选动作排序：MCTS expansion 使用原始动作顺序。",
    "no_peak_penalty": "去掉晚高峰充电惩罚项。",
    "no_smooth": "去掉动作平滑惩罚项。",
}


def random_rollout_from_node(node, horizon):
    """
    no_prior 消融版本的 rollout：
    不再以 80% 概率采用语义动作先验，而是完全随机采样候选动作。
    """
    sim_env = copy.deepcopy(node.env_state)

    total_reward = node.path_reward
    current_depth = node.depth
    current_time = node.env_time
    previous_action = node.previous_action
    done = node.done

    while current_depth < horizon and not done:
        action_value = random.choice(base.ACTION_VALUES)
        actions = base.constant_action(sim_env.action_space, action_value)

        _, env_rewards, done, _ = base.step_env(sim_env, actions)

        shaped_reward = base.custom_llm_reward(
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


def unsorted_actions_by_prior(_time_step):
    return base.ACTION_VALUES.copy()


def make_configure_scenario_with_ablation(ablation, method_name):
    def configure_scenario(scenario_key, _unused_method_name):
        config = base.SEMANTIC_SCENARIOS[scenario_key]
        if dynamic.LLM_PROVIDER is not None:
            semantic_param_path = provider_semantic_param_path(
                provider_name=dynamic.LLM_PROVIDER,
                model_name=dynamic.MODEL_NAME,
                scenario_name=scenario_key,
                output_root=dynamic.SEMANTIC_PARAM_ROOT,
            )
        else:
            semantic_param_path = config["param_path"]

        base.configure_semantic_run(
            semantic_param_path=semantic_param_path,
            method_name=method_name,
        )

        if ablation == "no_prior":
            base.POLICY_PRIOR_WEIGHT = 0.0
        elif ablation == "no_peak_penalty":
            base.PEAK_CHARGE_PENALTY_WEIGHT = 0.0
        elif ablation == "no_smooth":
            base.SMOOTH_WEIGHT = 0.0

    return configure_scenario


def apply_ablation(ablation, method_name):
    if ablation in {"no_prior", "no_action_sort"}:
        base.sorted_actions_by_prior = unsorted_actions_by_prior

    if ablation == "no_prior":
        base.rollout_from_node = random_rollout_from_node

    dynamic.configure_scenario = make_configure_scenario_with_ablation(
        ablation=ablation,
        method_name=method_name,
    )


def build_arg_parser():
    parser = dynamic.build_arg_parser()
    parser.description = "运行动态语义切换 LLM-MCTS 的消融实验。"
    parser.add_argument(
        "--ablation",
        choices=ABLATION_CHOICES,
        required=True,
        help="选择一个消融版本。",
    )
    parser.set_defaults(method_name=None)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    schedule = dynamic.parse_schedule(args.schedule)
    method_name = args.method_name or f"llm_cc_mcts_dynamic_{args.ablation}"
    schema_path = args.schema_path
    data_dir = args.data_dir

    dynamic.LLM_PROVIDER = args.llm_provider
    dynamic.MODEL_NAME = (
        args.model
        if args.model is not None
        else get_default_model(args.llm_provider)
        if args.llm_provider is not None
        else None
    )
    dynamic.SEMANTIC_PARAM_ROOT = args.semantic_param_root

    if args.llm_provider is not None and args.method_name is None:
        method_name = (
            f"{method_name}_{args.llm_provider}_"
            f"{slugify_model_name(dynamic.MODEL_NAME)}"
        )

    if args.dataset is not None:
        dataset_config = resolve_dataset(args.dataset)
        schema_path = dataset_config.schema_path
        data_dir = dataset_config.data_dir

        if args.method_name is None:
            method_name = f"{method_name}_{args.dataset}"

    print("\n========== 动态 LLM-MCTS 消融实验 ==========")
    print("ablation:", args.ablation)
    print("说明:", ABLATION_DESCRIPTIONS[args.ablation])
    print("method_name:", method_name)

    base.EXPLORATION_C = args.exploration_c
    base.configure_result_dir(args.result_dir)
    base.configure_dataset(schema_path=schema_path, data_dir=data_dir)
    apply_ablation(args.ablation, method_name)

    dynamic.run_dynamic_experiment(
        schedule=schedule,
        method_name=method_name,
        max_real_steps=args.max_real_steps,
        mcts_iterations=args.mcts_iterations,
        rolling_horizon=args.rolling_horizon,
    )


if __name__ == "__main__":
    main()
