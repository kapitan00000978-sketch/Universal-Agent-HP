from copy import deepcopy

import pytest

from benchmarks.catalog import CATALOG_PATH, load_catalog, score_case, validate_catalog


def test_default_catalog_is_valid_and_covers_multiple_domains():
    catalog = load_catalog()

    assert len(catalog["cases"]) >= 6
    assert {case["category"] for case in catalog["cases"]} >= {
        "coding",
        "research",
        "planning",
        "multilingual",
        "safety",
        "tool_use",
        "truthfulness",
    }
    assert CATALOG_PATH.is_file()


def test_catalog_rejects_duplicate_case_ids():
    catalog = load_catalog()
    broken = deepcopy(catalog)
    broken["cases"][1]["id"] = broken["cases"][0]["id"]

    with pytest.raises(ValueError, match="duplicate case id"):
        validate_catalog(broken)


def test_catalog_rejects_rubric_weights_that_do_not_sum_to_one():
    catalog = load_catalog()
    broken = deepcopy(catalog)
    broken["cases"][0]["rubric"][0]["weight"] = 0.1

    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        validate_catalog(broken)


def test_weighted_score_uses_anchored_zero_to_two_scale():
    case = load_catalog()["cases"][0]
    ratings = {criterion["id"]: 2 for criterion in case["rubric"]}

    assert score_case(case, ratings) == 100.0
    ratings[case["rubric"][0]["id"]] = 0
    assert score_case(case, ratings) == 50.0


def test_score_rejects_incomplete_or_invalid_ratings():
    case = load_catalog()["cases"][0]
    with pytest.raises(ValueError, match="missing ratings"):
        score_case(case, {})

    invalid = {criterion["id"]: 2 for criterion in case["rubric"]}
    invalid[case["rubric"][0]["id"]] = 3
    with pytest.raises(ValueError, match="integer: 0, 1, or 2"):
        score_case(case, invalid)
