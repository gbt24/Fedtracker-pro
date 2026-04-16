"""项目工程化文件测试。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


class TestProjectScaffold(unittest.TestCase):
    def _repo_root(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    def test_setup_py_exists_and_declares_package(self) -> None:
        setup_path = os.path.join(self._repo_root(), "setup.py")
        self.assertTrue(os.path.exists(setup_path))
        with open(setup_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("setup(", content)
        self.assertIn('name="fedtracker-pro"', content)
        self.assertIn('find_packages(include=["src*", "experiments*"])', content)

    def test_makefile_exists_and_has_key_targets(self) -> None:
        makefile_path = os.path.join(self._repo_root(), "Makefile")
        self.assertTrue(os.path.exists(makefile_path))
        with open(makefile_path, "r", encoding="utf-8") as f:
            content = f.read()
        for target in [
            "install:",
            "test:",
            "test-cov:",
            "run-baseline:",
            "run-ablation:",
        ]:
            self.assertIn(target, content)
        self.assertIn("python -m compileall -q src experiments", content)

    def test_readme_mentions_phase8_artifacts(self) -> None:
        readme_path = os.path.join(self._repo_root(), "README.md")
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("exp_robustness.py", content)
        self.assertIn("exp_scalability.py", content)
        self.assertIn("configs/default.yaml", content)

    def test_requirements_dev_exists_and_has_test_tools(self) -> None:
        path = os.path.join(self._repo_root(), "requirements-dev.txt")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("pytest", content)
        self.assertIn("mypy", content)

    def test_scripts_exist(self) -> None:
        for script_name in ["setup.sh", "train.sh", "evaluate.sh", "visualize.sh"]:
            script_path = os.path.join(self._repo_root(), "scripts", script_name)
            self.assertTrue(os.path.exists(script_path))

    def test_docs_exist(self) -> None:
        for doc_name in ["api.md", "tutorial.md"]:
            doc_path = os.path.join(self._repo_root(), "docs", doc_name)
            self.assertTrue(os.path.exists(doc_path))

    def test_test_package_init_files_exist(self) -> None:
        for rel_path in ["tests/__init__.py", "tests/fixtures/__init__.py"]:
            abs_path = os.path.join(self._repo_root(), rel_path)
            self.assertTrue(os.path.exists(abs_path))

    def test_integration_tests_exist(self) -> None:
        for rel_path in [
            "tests/integration/test_end_to_end.py",
            "tests/integration/test_attacks.py",
            "tests/integration/test_multi_layer.py",
        ]:
            abs_path = os.path.join(self._repo_root(), rel_path)
            self.assertTrue(os.path.exists(abs_path))


if __name__ == "__main__":
    unittest.main()
