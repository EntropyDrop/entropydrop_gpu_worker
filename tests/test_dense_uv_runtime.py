from pathlib import Path


def test_dense_uv_runtime_uses_v61_production_defaults():
    source = Path("dense_uv_runtime.py").read_text(encoding="utf-8")

    assert "SkingToolkit v61 pipeline" in source
    assert "outer_uv_min_source_pixels=33" in source
    assert 'color_aggregation="grid_mode"' in source
    assert "outer_uv_occupancy=False" in source
