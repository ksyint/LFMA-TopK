"""Exact sparse support budgets for encoder projection matrices."""

import math
from collections import defaultdict

from lfma.core import BACKBONES, IMAGE_TASKS


def projection_layout(backbone):
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone: {backbone}")
    spec = BACKBONES[backbone]
    roles = ("query",) if spec["family"] == "vit" else ("query", "value")
    return [
        {
            "layer": layer,
            "role": role,
            "width": spec["width"],
            "elements": spec["width"] ** 2,
        }
        for layer in range(spec["layers"])
        for role in roles
    ]


def support_budget(backbone, ratio, classes=None):
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("Support ratio must be finite and lie in (0,1]")
    layout = projection_layout(backbone)
    for row in layout:
        row["coefficients"] = int(row["elements"] * ratio)
        if row["coefficients"] < 1:
            raise ValueError("Requested ratio yields an empty projection support")
        row["real_parameters"] = 2 * row["coefficients"]
        row["coefficient_bytes_fp32"] = row["real_parameters"] * 4
        row["index_bytes_int64"] = row["coefficients"] * 8
        row["dense_update_bytes_fp32"] = row["elements"] * 4
    total = sum(row["real_parameters"] for row in layout)
    result = {
        "backbone": backbone,
        "ratio": ratio,
        "projections": len(layout),
        "complex_coefficients": total // 2,
        "real_parameters": total,
        "coefficient_bytes_fp32": total * 4,
        "index_bytes_int64": total // 2 * 8,
        "adam_moment_bytes_fp32": total * 8,
        "layers": layout,
    }
    if classes is not None:
        if classes < 1:
            raise ValueError("Head classes must be positive")
        width = BACKBONES[backbone]["width"]
        head = width * classes + classes
        if BACKBONES[backbone]["family"] == "roberta":
            head += width * width + width
        result["head_parameters"] = head
        result["total_trainable_parameters"] = total + head
    return result


def ratio_for_budget(backbone, real_parameters):
    layout = projection_layout(backbone)
    if real_parameters < 2 * len(layout):
        raise ValueError(
            "Budget must allocate at least one complex coefficient per projection"
        )
    per_projection = min(layout[0]["elements"], real_parameters // (2 * len(layout)))
    ratio = per_projection / layout[0]["elements"]
    # Move inward to retain the intended integer support despite float rounding.
    if int(layout[0]["elements"] * ratio) < per_projection:
        ratio = math.nextafter(ratio, 1.0)
    report = support_budget(backbone, ratio)
    report["requested_real_parameters"] = int(real_parameters)
    report["unallocated_parameters"] = int(real_parameters) - report["real_parameters"]
    return report


def budget_grid(backbones, ratios, tasks=None):
    rows = []
    tasks = tasks or [None]
    for backbone in backbones:
        for ratio in ratios:
            for task in tasks:
                family = BACKBONES[backbone]["family"]
                if task is not None and (task in IMAGE_TASKS) != (family == "vit"):
                    continue
                classes = (
                    None
                    if task is None
                    else IMAGE_TASKS.get(task, 1 if task == "stsb" else 2)
                )
                row = support_budget(backbone, ratio, classes)
                row.pop("layers")
                row["task"] = task
                rows.append(row)
    return rows


def actual_budget(model):
    from lfma.adaptation.layers import FourierLinear

    rows = []
    adapter_ids = set()
    for name, layer in model.named_modules():
        if not isinstance(layer, FourierLinear):
            continue
        adapter_ids.add(id(layer.c))
        rows.append(
            {
                "name": name,
                "shape": list(layer.base_layer.weight.shape),
                "coefficients": layer.k,
                "real_parameters": layer.c.numel(),
                "alpha": layer.alpha,
                "ratio": layer.k / layer.base_layer.weight.numel(),
                "coefficient_bytes": layer.c.numel() * layer.c.element_size(),
                "index_bytes": layer.indices.numel() * layer.indices.element_size(),
            }
        )
    if not rows:
        raise ValueError("Model has no Fourier adapters")
    other = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in adapter_ids
    )
    return {
        "projections": rows,
        "adapter_real_parameters": sum(row["real_parameters"] for row in rows),
        "other_trainable_parameters": other,
        "frozen_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
    }


def budget_by_layer(report):
    groups = defaultdict(lambda: {"coefficients": 0, "real_parameters": 0, "roles": []})
    for row in report["layers"]:
        group = groups[row["layer"]]
        group["coefficients"] += row["coefficients"]
        group["real_parameters"] += row["real_parameters"]
        group["roles"].append(row["role"])
    return [dict(layer=layer, **row) for layer, row in sorted(groups.items())]
