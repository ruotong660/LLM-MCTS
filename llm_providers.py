from pathlib import Path
import os
import re

from openai import OpenAI


PROVIDERS = {
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com",
        "default_model": "deepseek-v4-flash",
    },
    "qwen": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
    },
}


def available_providers():
    return sorted(PROVIDERS)


def print_available_providers():
    print("可用 LLM provider：")
    for provider_name in available_providers():
        config = PROVIDERS[provider_name]
        print(
            f"- {provider_name}: default_model={config['default_model']}, "
            f"api_key_env={config['api_key_env']}, "
            f"base_url={get_base_url(provider_name)}"
        )


def get_provider_config(provider_name):
    if provider_name not in PROVIDERS:
        available = ", ".join(available_providers())
        raise KeyError(f"未知 LLM provider：{provider_name}。可选：{available}")

    return PROVIDERS[provider_name]


def get_default_model(provider_name):
    return get_provider_config(provider_name)["default_model"]


def get_base_url(provider_name):
    config = get_provider_config(provider_name)
    return os.getenv(config["base_url_env"], config["default_base_url"])


def get_api_key(provider_name):
    config = get_provider_config(provider_name)
    api_key = os.getenv(config["api_key_env"])

    if not api_key:
        raise RuntimeError(
            f"没有检测到 {config['api_key_env']}。\n"
            f"请先设置环境变量，例如：\n"
            f'export {config["api_key_env"]}="你的 API Key"'
        )

    return api_key


def build_chat_client(provider_name):
    return OpenAI(
        api_key=get_api_key(provider_name),
        base_url=get_base_url(provider_name),
    )


def complete_chat(provider_name, model_name, messages, temperature=0.0):
    client = build_chat_client(provider_name)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def slugify_model_name(model_name):
    model_name = str(model_name).strip()
    model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    return model_name.strip("_")


def semantic_output_dir(provider_name, model_name, output_root=Path("semantic_params")):
    return Path(output_root) / provider_name / slugify_model_name(model_name)


def semantic_param_path(
    provider_name,
    model_name,
    scenario_name,
    output_root=Path("semantic_params"),
):
    return semantic_output_dir(provider_name, model_name, output_root) / f"{scenario_name}.json"
