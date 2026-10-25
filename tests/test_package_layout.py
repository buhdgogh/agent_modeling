"""Migration checks for package imports, resource paths and checkpoint models."""
import ast
import importlib
import os
from pathlib import Path
import pickle
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from agent_modeling import config as paths
from agent_modeling.storage.state import (
    LEGACY_MODEL_MODULES, MODEL_TYPES, trusted_checkpoint_types,
)
from agent_modeling.models import StratumItem

ROOT = Path(__file__).resolve().parents[1]


class PackageLayoutTests(unittest.TestCase):
    def test_all_internal_import_modules_and_symbols_exist(self):
        package = ROOT / "agent_modeling"
        sources = {}
        for path in package.rglob("*.py"):
            module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
            if module.endswith(".__init__"):
                module = module[:-9]
            sources[module] = ast.parse(path.read_text(encoding="utf-8"))
        for module, tree in sources.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("agent_modeling."):
                    with self.subTest(module=module, dependency=node.module):
                        self.assertIn(node.module, sources)
                        target_tree = sources[node.module]
                        names = {n.name for n in ast.walk(target_tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                        names.update(n.id for n in ast.walk(target_tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
                        names.update(alias.asname or alias.name for n in ast.walk(target_tree)
                                     if isinstance(n, (ast.Import, ast.ImportFrom)) for alias in n.names)
                        for alias in node.names:
                            self.assertIn(alias.name, names)

    def test_configuration_runtime_and_model_roots_survive_cwd_change(self):
        previous = Path.cwd()
        try:
            os.chdir(ROOT / "tests")
            self.assertEqual(paths.PROJECT_ROOT, ROOT)
            self.assertEqual(paths.ENV_FILE, ROOT / "api_key.env")
            self.assertEqual(paths.CHECKPOINT_ROOT, ROOT / "runtime" / "checkpoints")
            self.assertEqual(paths.PEACE_ROOT, ROOT / "PEACE")
            self.assertTrue(paths.DEFAULT_IMAGE_OUTPUT.is_relative_to(ROOT / "runtime"))
            self.assertTrue(paths.DEFAULT_BOREHOLE_OUTPUT.is_relative_to(ROOT / "runtime"))
        finally:
            os.chdir(previous)

    def test_new_models_and_legacy_modules_are_exactly_whitelisted(self):
        whitelist = trusted_checkpoint_types()
        for module, names in MODEL_TYPES.items():
            for name in names:
                self.assertIn((module, name), whitelist)
                model = getattr(importlib.import_module(module), name)
                self.assertEqual(model.__module__, module)
        for old, new in LEGACY_MODEL_MODULES.items():
            self.assertIs(importlib.import_module(old), importlib.import_module(new))
        self.assertFalse(any(name == "*" for _, name in whitelist))

    def test_old_serialized_model_can_resolve_after_package_move(self):
        trusted_checkpoint_types()
        record = StratumItem(formation="测试地层")
        for old_module in ("text_info_agent", "agent_modeling.schemas.text"):
            with self.subTest(module=old_module):
                with patch.object(StratumItem, "__module__", old_module):
                    old_payload = pickle.dumps(record)
                restored = pickle.loads(old_payload)
                self.assertIsInstance(restored, StratumItem)
                self.assertEqual(restored.formation, "测试地层")

    def test_entrypoint_executes_ui_on_every_streamlit_rerun(self):
        with patch("runpy.run_module") as execute:
            runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
            runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
        self.assertEqual(execute.call_count, 2)
        execute.assert_called_with("agent_modeling.interface", run_name="__main__")

    def test_document_and_schema_imports_do_not_load_heavy_services(self):
        code = (
            "import sys; import agent_modeling.documents.parser; "
            "import agent_modeling.models; import agent_modeling.models; "
            "import agent_modeling.models; "
            "assert not ({'langchain_core','torch','cv2','mineru','mysql.connector'} & set(sys.modules))"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
