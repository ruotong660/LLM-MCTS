from dataclasses import dataclass
from pathlib import Path
import json


DATASET_REGISTRY_PATH = Path("configs/datasets.json")


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    schema_path: Path
    data_dir: Path
    description: str = ""


def load_dataset_registry(registry_path=DATASET_REGISTRY_PATH):
    registry_path = Path(registry_path)

    if not registry_path.exists():
        raise FileNotFoundError(f"找不到数据集配置文件：{registry_path}")

    with open(registry_path, "r", encoding="utf-8") as f:
        raw_registry = json.load(f)

    registry = {}
    for dataset_id, config in raw_registry.items():
        registry[dataset_id] = DatasetConfig(
            dataset_id=dataset_id,
            schema_path=Path(config["schema_path"]),
            data_dir=Path(config.get("data_dir", Path(config["schema_path"]).parent)),
            description=config.get("description", ""),
        )

    return registry


def resolve_dataset(dataset_id, registry_path=DATASET_REGISTRY_PATH):
    registry = load_dataset_registry(registry_path)

    if dataset_id not in registry:
        available = ", ".join(sorted(registry))
        raise KeyError(f"未知数据集：{dataset_id}。可选数据集：{available}")

    config = registry[dataset_id]

    if not config.schema_path.exists():
        raise FileNotFoundError(f"数据集 {dataset_id} 缺少 schema：{config.schema_path}")

    if not config.data_dir.exists():
        raise FileNotFoundError(f"数据集 {dataset_id} 缺少 data_dir：{config.data_dir}")

    pricing_path = config.data_dir / "pricing.csv"
    carbon_path = config.data_dir / "carbon_intensity.csv"

    if not pricing_path.exists():
        raise FileNotFoundError(f"数据集 {dataset_id} 缺少 pricing.csv：{pricing_path}")

    if not carbon_path.exists():
        raise FileNotFoundError(
            f"数据集 {dataset_id} 缺少 carbon_intensity.csv：{carbon_path}"
        )

    return config


def print_dataset_registry(registry_path=DATASET_REGISTRY_PATH):
    registry = load_dataset_registry(registry_path)

    print("可用数据集：")
    for dataset_id, config in sorted(registry.items()):
        print(
            f"- {dataset_id}: {config.description} "
            f"({config.schema_path})"
        )
