import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class ReleaseTests(unittest.TestCase):
    def test_archived_result_arithmetic(self):
        module("verify_release").verify_results(ROOT)

    def test_secret_scan(self):
        verify = module("verify_release")
        self.assertTrue(verify.scan_text("sk-" + "a" * 32))
        self.assertFalse(verify.scan_text('api_key = os.getenv("DEEPSEEK_API_KEY")'))

    def test_import_validates_all_sources_before_copy(self):
        importer = module("import_dataset")
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "release"
            source = Path(tmp) / "dataset"
            source.mkdir()
            hashes = {}
            for i in range(9):
                path = source / f"file_{i}.csv"
                path.write_text(str(i))
                hashes[importer.PREFIX + path.name] = importer.digest(path)
            config = root / "artifacts/fixed_parameter_control/config.json"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"input_hashes": hashes}))
            (source / "file_8.csv").write_text("changed")
            with patch.object(importer, "ROOT", root):
                with self.assertRaises(ValueError):
                    importer.import_dataset(source)
                self.assertFalse((root / "data").exists())
                (source / "file_8.csv").write_text("8")
                importer.import_dataset(source)
                importer.import_dataset(source)
                (root / importer.PREFIX / "file_0.csv").write_text("different")
                with self.assertRaises(ValueError):
                    importer.import_dataset(source)


if __name__ == "__main__":
    unittest.main()
