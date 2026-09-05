from __future__ import annotations

import shutil
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import certifi

from config import ModelSpec


def ensure_models(models_dir: Path, specs: tuple[ModelSpec, ...]) -> dict[str, Path]:
    """Download official MediaPipe task bundles once; inference remains local."""
    models_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for spec in specs:
        destination = models_dir / spec.filename
        paths[spec.filename] = destination
        if destination.exists() and destination.stat().st_size > 100_000:
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            print(f"Downloading MediaPipe model: {spec.filename}")
            # Some framework Python installations on macOS do not expose the system CA store.
            context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(spec.url, timeout=30, context=context) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(destination)
        except (OSError, urllib.error.URLError) as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not obtain {spec.filename}. Connect once to download the local model, "
                f"then retry. Details: {error}"
            ) from error
    return paths
