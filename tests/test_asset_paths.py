import pytest

from asset_paths import dense_uv_asset_paths


def test_dense_uv_paths_join_worker_roots_and_task_parameters():
    checkpoint, mappings = dense_uv_asset_paths(
        "/models/Sking",
        "/renderer",
        "SKING_DDJ_v66.pt",
        "mappings_256x512",
    )

    assert checkpoint.as_posix() == "/models/Sking/SKING_DDJ_v66.pt"
    assert mappings.as_posix() == "/renderer/mappings_256x512"


@pytest.mark.parametrize("unsafe", ["", "/absolute.pt", "../escape.pt"])
def test_dense_uv_paths_reject_unsafe_task_paths(unsafe):
    with pytest.raises(ValueError, match="safe relative path"):
        dense_uv_asset_paths("/models", "/renderer", unsafe, "mappings")
