"""Load, validate, and score the versioned benchmark task catalog.

The catalog and scoring helpers intentionally use only the Python standard
library so benchmark definitions can be validated without installing the full
agent stack or contacting external services.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).with_name("cases.json")
CATEGORIES = {
    "coding",
    "research",
    "planning",
    "multilingual",
    "safety",
    "tool_use",
    "truthfulness",
}
MODES = {"fast", "deep", "deep_search"}
RISKS = {"low", "medium", "high", "critical"}


def load_catalog(path: Path | str = CATALOG_PATH) -> dict[str, Any]:
    """Read and validate a benchmark catalog JSON file."""
    catalog_path = Path(path)
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read benchmark catalog {catalog_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in benchmark catalog {catalog_path}: {exc}") from exc
    validate_catalog(catalog)
    return catalog


def validate_catalog(catalog: Any) -> None:
    """Raise ValueError with actionable details if a catalog is malformed."""
    errors: list[str] = []
    if not isinstance(catalog, dict):
        raise ValueError("Catalog root must be a JSON object.")
    if catalog.get("version") != 1:
        errors.append("version must be 1")
    cases = catalog.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("cases must be a non-empty array")
        cases = []

    seen_case_ids: set[str] = set()
    for index, case in enumerate(cases):
        prefix = f"cases[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{prefix} must be an object")
            continue
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            errors.append(f"{prefix}.id must be a non-empty string")
        elif case_id in seen_case_ids:
            errors.append(f"duplicate case id: {case_id}")
        else:
            seen_case_ids.add(case_id)

        if case.get("category") not in CATEGORIES:
            errors.append(f"{prefix}.category must be one of {sorted(CATEGORIES)}")
        if not isinstance(case.get("prompt"), str) or not case["prompt"].strip():
            errors.append(f"{prefix}.prompt must be a non-empty string")
        if case.get("mode") not in MODES:
            errors.append(f"{prefix}.mode must be one of {sorted(MODES)}")
        if case.get("risk") not in RISKS:
            errors.append(f"{prefix}.risk must be one of {sorted(RISKS)}")

        rubric = case.get("rubric")
        if not isinstance(rubric, list) or not rubric:
            errors.append(f"{prefix}.rubric must be a non-empty array")
            continue
        criterion_ids: set[str] = set()
        weight_sum = 0.0
        for criterion_index, criterion in enumerate(rubric):
            criterion_prefix = f"{prefix}.rubric[{criterion_index}]"
            if not isinstance(criterion, dict):
                errors.append(f"{criterion_prefix} must be an object")
                continue
            criterion_id = criterion.get("id")
            if not isinstance(criterion_id, str) or not criterion_id.strip():
                errors.append(f"{criterion_prefix}.id must be a non-empty string")
            elif criterion_id in criterion_ids:
                errors.append(f"{prefix} has duplicate rubric id: {criterion_id}")
            else:
                criterion_ids.add(criterion_id)
            description = criterion.get("description")
            if not isinstance(description, str) or not description.strip():
                errors.append(f"{criterion_prefix}.description must be a non-empty string")
            weight = criterion.get("weight")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                errors.append(f"{criterion_prefix}.weight must be a number")
            elif not math.isfinite(weight) or weight <= 0:
                errors.append(f"{criterion_prefix}.weight must be finite and greater than zero")
            else:
                weight_sum += weight
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
            errors.append(f"{prefix}.rubric weights must sum to 1.0 (got {weight_sum:.6g})")

        gates = case.get("critical_gates", [])
        if not isinstance(gates, list) or any(not isinstance(gate, str) for gate in gates):
            errors.append(f"{prefix}.critical_gates must be an array of rubric ids")
        else:
            unknown_gates = set(gates) - criterion_ids
            if unknown_gates:
                errors.append(f"{prefix}.critical_gates references unknown rubric id(s): {', '.join(sorted(unknown_gates))}")
            if case.get("risk") == "critical" and not gates:
                errors.append(f"{prefix} is critical-risk and must define at least one critical gate")

    if errors:
        raise ValueError("Invalid benchmark catalog:\n- " + "\n- ".join(errors))


def score_case(case: dict[str, Any], ratings: dict[str, int]) -> float:
    """Return a rubric-weighted percentage for ratings on a 0, 1, or 2 scale.

    Every rubric criterion must be rated exactly once. 0 means fail, 1 partial,
    and 2 full credit. The result is a percentage in the inclusive range 0–100.
    """
    rubric = case.get("rubric")
    if not isinstance(rubric, list) or not rubric:
        raise ValueError("Cannot score a case without a rubric.")
    expected = {item["id"] for item in rubric}
    supplied = set(ratings)
    missing = expected - supplied
    unknown = supplied - expected
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing ratings: {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown ratings: {', '.join(sorted(unknown))}")
        raise ValueError("; ".join(details))

    weighted = 0.0
    for item in rubric:
        rating = ratings[item["id"]]
        if isinstance(rating, bool) or not isinstance(rating, int) or rating not in (0, 1, 2):
            raise ValueError(f"Rating for {item['id']} must be an integer: 0, 1, or 2.")
        weighted += item["weight"] * rating / 2
    return round(weighted * 100, 2)


def main() -> int:
    """Validate and summarize the default catalog."""
    catalog = load_catalog()
    counts: dict[str, int] = {}
    for case in catalog["cases"]:
        category = case["category"]
        counts[category] = counts.get(category, 0) + 1
    print(f"Benchmark catalog v{catalog['version']}: {len(catalog['cases'])} cases")
    print("Category coverage: " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))
    print("Rubric scale: 0 = fail, 1 = partial, 2 = full credit; case score = weighted percent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
