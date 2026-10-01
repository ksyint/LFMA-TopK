"""Checkpoint file inventories, parameter storage, and integrity checks."""

import hashlib
import json
from pathlib import Path


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_inventory(directory, hashes=True):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "inventory.json":
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Artifact path escapes its checkpoint directory: {path}")
        row = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size}
        if hashes:
            row["sha256"] = digest_file(path)
        rows.append(row)
    return {
        "format": "lfma-artifact-inventory-v1",
        "files": rows,
        "total_bytes": sum(row["bytes"] for row in rows),
        "file_count": len(rows),
    }


def write_inventory(directory):
    report = directory_inventory(directory)
    path = Path(directory) / "inventory.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)
    return report


def verify_inventory(directory):
    root = Path(directory)
    expected = json.loads((root / "inventory.json").read_text())
    if expected.get("format") != "lfma-artifact-inventory-v1":
        raise ValueError("Unsupported checkpoint inventory")
    actual = directory_inventory(root)
    before = {row["path"]: row for row in expected["files"]}
    after = {row["path"]: row for row in actual["files"]}
    missing = sorted(before.keys() - after.keys())
    extra = sorted(after.keys() - before.keys())
    changed = [
        name for name in before.keys() & after.keys() if before[name] != after[name]
    ]
    result = {"missing": missing, "extra": extra, "changed": sorted(changed)}
    result["valid"] = not any(result.values())
    return result


def tensor_inventory(directory):
    from safetensors import safe_open

    path = Path(directory) / "adapter_model.safetensors"
    rows = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in sorted(handle.keys()):
            view = handle.get_slice(name)
            shape = view.get_shape()
            count = 1
            for dimension in shape:
                count *= dimension
            rows.append(
                {
                    "name": name,
                    "shape": shape,
                    "elements": count,
                    "dtype": view.get_dtype(),
                }
            )
    return rows


def checkpoint_inventory(directory, verify=False):
    root = Path(directory)
    from lfma.artifacts.storage import read_metadata
    from lfma.artifacts.resume import execution_metadata

    metadata = read_metadata(root)
    files = directory_inventory(root, hashes=False)
    tensors = tensor_inventory(root)
    storage = {"coefficients": 0, "support_indices": 0, "classifier": 0}
    for row in tensors:
        if row["name"].endswith(".c"):
            storage["coefficients"] += row["elements"]
        elif row["name"].endswith(".indices"):
            storage["support_indices"] += row["elements"]
        elif row["name"].startswith("classifier."):
            storage["classifier"] += row["elements"]
        else:
            raise ValueError(f"Unknown adapter tensor: {row['name']}")
    result = {
        "checkpoint": str(root.resolve()),
        "backbone": metadata["config"]["model"]["backbone"],
        "task": metadata["config"]["data"]["task"],
        "epoch": metadata["epoch"],
        "best_score": metadata["best_score"],
        "parameters": storage,
        "files": files,
        "tensors": tensors,
        "execution": execution_metadata(root),
    }
    if verify:
        result["integrity"] = verify_inventory(root)
    return result
