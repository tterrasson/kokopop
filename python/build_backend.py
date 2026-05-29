import os
import tarfile
from pathlib import Path

from scikit_build_core import build as _scikit_build


PROJECT_NAME = "kokopop"
VERSION = "0.1.0"


def _iter_files(root: Path):
    excluded_dirs = {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "build",
        "dist",
        ".venv",
    }
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix in {".pyc", ".pyo"}:
                continue
            yield path


def _add_tree(tar: tarfile.TarFile, src: Path, dst_root: str) -> None:
    for path in _iter_files(src):
        tar.add(path, arcname=f"{dst_root}/{path.relative_to(src)}", recursive=False)


def build_sdist(sdist_directory: str, config_settings=None) -> str:
    here = Path(__file__).resolve().parent
    repo = here.parent
    root_name = f"{PROJECT_NAME}-{VERSION}"
    output = Path(sdist_directory) / f"{root_name}.tar.gz"

    with tarfile.open(output, "w:gz", format=tarfile.PAX_FORMAT) as tar:
        _add_tree(tar, here, root_name)
        for rel in ("CMakeLists.txt", "include", "src", "patches"):
            _add_tree(tar, repo / rel, f"{root_name}/kokopop_core/{rel}") if (
                repo / rel
            ).is_dir() else tar.add(
                repo / rel, arcname=f"{root_name}/kokopop_core/{rel}"
            )
        tar.add(
            repo / "tools" / "apply_patch_if_needed.cmake",
            arcname=f"{root_name}/kokopop_core/tools/apply_patch_if_needed.cmake",
        )
        if (repo / "LICENSE").exists():
            tar.add(repo / "LICENSE", arcname=f"{root_name}/LICENSE")

    return output.name


def build_wheel(
    wheel_directory: str, config_settings=None, metadata_directory=None
) -> str:
    return _scikit_build.build_wheel(
        wheel_directory, config_settings, metadata_directory
    )


def build_editable(
    wheel_directory: str, config_settings=None, metadata_directory=None
) -> str:
    return _scikit_build.build_editable(
        wheel_directory, config_settings, metadata_directory
    )


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings=None
) -> str:
    return _scikit_build.prepare_metadata_for_build_wheel(
        metadata_directory, config_settings
    )


def get_requires_for_build_wheel(config_settings=None):
    return _scikit_build.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return _scikit_build.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return _scikit_build.get_requires_for_build_editable(config_settings)
