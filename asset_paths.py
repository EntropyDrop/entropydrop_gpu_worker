from pathlib import Path


def rooted_asset_path(root: str, relative_path: str, label: str) -> Path:
    relative = Path(relative_path)
    if (
        not relative_path
        or relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
    ):
        raise ValueError(
            f"{label} must be a safe relative path: {relative_path!r}"
        )
    return Path(root) / relative


def dense_uv_asset_paths(
    sking_root_dir: str,
    dmr_root_dir: str,
    dense_uv_checkpoint_file: str,
    DMR_mappings_dir: str,
    model_version: str | None = None,
) -> tuple[Path, Path]:
    if model_version in {"SKING_DDJ_v130", "SKING_DDJ_v131"}:
        rooted_asset_path(".", dense_uv_checkpoint_file, "dense_uv_checkpoint_file")
        rooted_asset_path(".", DMR_mappings_dir, "DMR_mappings_dir")
        if (dense_uv_checkpoint_file != f"{model_version}/parser.pt"
                or DMR_mappings_dir != f"{model_version}/mappings"):
            raise ValueError(f"{model_version} requires its bundled parser.pt and mappings")
        root = Path(__file__).resolve().parent / "releases" / model_version
        return root / "parser.pt", root / "mappings"
    return (
        rooted_asset_path(
            sking_root_dir,
            dense_uv_checkpoint_file,
            "dense_uv_checkpoint_file",
        ),
        rooted_asset_path(
            dmr_root_dir,
            DMR_mappings_dir,
            "DMR_mappings_dir",
        ),
    )
