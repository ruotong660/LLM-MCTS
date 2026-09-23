from pathlib import Path
import argparse
import json
import re

from llm_providers import (
    available_providers,
    complete_chat,
    get_base_url,
    get_default_model,
    print_available_providers,
    semantic_output_dir,
)


SCENARIO_DIR = Path("semantic_scenarios")
OUTPUT_ROOT = Path("semantic_params")


SYSTEM_PROMPT = """
你是城市微电网储能调度的语义解析器。
你的任务是把外部运行要求文本转换成结构化调度参数。

只能输出 JSON，不要输出解释文字。

JSON 字段必须包含：
{
  "scenario_name": string,
  "price_weight": number,
  "carbon_weight": number,
  "low_q": number,
  "high_q": number,
  "rbc_action_magnitude": number,
  "alignment_weight": number,
  "policy_prior_weight": number,
  "smooth_weight": number,
  "peak_charge_penalty_weight": number,
  "explanation": string
}

取值规则：
1. price_weight + carbon_weight 必须等于 1.0。
2. 如果文本强调成本，则 price_weight 应大于 carbon_weight。
3. 如果文本强调低碳，则 carbon_weight 应大于 price_weight。
4. 如果文本强调晚高峰或削峰，则 peak_charge_penalty_weight 应较高。
5. low_q 是能碳综合得分的低分位阈值，通常在 0.15 到 0.30 之间。
6. high_q 是能碳综合得分的高分位阈值，通常在 0.70 到 0.85 之间。
7. rbc_action_magnitude 通常为 0.1，避免动作过激。
8. alignment_weight 越大，越强调能碳方向匹配，通常 8 到 12。
9. policy_prior_weight 通常 1 到 3。
10. smooth_weight 越大，越抑制频繁充放电，通常 0.2 到 0.6。
11. peak_charge_penalty_weight 越大，越避免晚高峰充电，通常 1 到 2。
"""


def extract_json(text):
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise ValueError("LLM 输出中没有找到 JSON。原始输出：\n" + text)

    return json.loads(match.group(0))


def validate_params(params):
    required_keys = [
        "scenario_name",
        "price_weight",
        "carbon_weight",
        "low_q",
        "high_q",
        "rbc_action_magnitude",
        "alignment_weight",
        "policy_prior_weight",
        "smooth_weight",
        "peak_charge_penalty_weight",
        "explanation",
    ]

    for key in required_keys:
        if key not in params:
            raise ValueError(f"缺少字段：{key}")

    numeric_keys = [
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

    for key in numeric_keys:
        params[key] = float(params[key])

    total = params["price_weight"] + params["carbon_weight"]

    if abs(total - 1.0) > 1e-6:
        params["price_weight"] = params["price_weight"] / total
        params["carbon_weight"] = params["carbon_weight"] / total

    if not (0.0 < params["low_q"] < params["high_q"] < 1.0):
        raise ValueError("需要满足 0 < low_q < high_q < 1")

    return params


def parse_one_file(provider_name, model_name, input_path, output_dir):
    semantic_text = input_path.read_text(encoding="utf-8")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": semantic_text},
    ]

    raw_output = complete_chat(
        provider_name=provider_name,
        model_name=model_name,
        messages=messages,
        temperature=0.0,
    )

    print(f"\n========== {input_path.name} {provider_name}/{model_name} 原始输出 ==========")
    print(raw_output)

    params = extract_json(raw_output)
    params = validate_params(params)

    params["source_text"] = semantic_text
    params["model_name"] = model_name
    params["llm_provider"] = provider_name
    params["scenario_file"] = input_path.name

    output_path = output_dir / f"{input_path.stem}.json"

    output_path.write_text(
        json.dumps(params, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n已保存：{output_path}")
    print(json.dumps(params, ensure_ascii=False, indent=2))

    return output_path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="批量调用 LLM，将 semantic_scenarios/*.txt 解析为结构化调度参数。"
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="列出可用 LLM provider 后退出。",
    )
    parser.add_argument(
        "--llm-provider",
        choices=available_providers(),
        default="deepseek",
        help="选择 LLM provider。",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="模型名称；不填则使用 provider 默认模型。",
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=SCENARIO_DIR,
        help="语义场景 txt 文件目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="语义参数输出根目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="显式指定输出目录；不填则输出到 semantic_params/{provider}/{model}/。",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_providers:
        print_available_providers()
        return

    provider_name = args.llm_provider
    model_name = args.model or get_default_model(provider_name)
    output_dir = args.output_dir or semantic_output_dir(
        provider_name=provider_name,
        model_name=model_name,
        output_root=args.output_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_files = sorted(args.scenario_dir.glob("*.txt"))

    if len(scenario_files) == 0:
        raise FileNotFoundError(f"{args.scenario_dir} 文件夹里没有 txt 文件")

    print("LLM provider：", provider_name)
    print("LLM model：", model_name)
    print("Base URL：", get_base_url(provider_name))
    print("输出目录：", output_dir)
    print("将解析以下语义场景：")
    for path in scenario_files:
        print("-", path)

    output_paths = []

    for path in scenario_files:
        output_paths.append(
            parse_one_file(
                provider_name=provider_name,
                model_name=model_name,
                input_path=path,
                output_dir=output_dir,
            )
        )

    print("\n========== 所有语义场景解析完成 ==========")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
