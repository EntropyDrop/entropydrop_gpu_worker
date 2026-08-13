import ast
from pathlib import Path


def test_dense_uv_runtime_uses_shared_production_defaults():
    source = Path("dense_uv_runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    integer_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
    }

    assert "production_preprocessing_defaults" in source
    assert "production_splat_defaults" in source
    assert "**self.splat_kwargs" in source
    # The old runtime silently overrode run_infer.sh's value of 15 with 40.
    assert 40 not in integer_literals

