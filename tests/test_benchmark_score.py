import hashlib

import pytest

from benchmarks.catalog import CATALOG_PATH, load_catalog
from benchmarks.score import score_run


def _run_document(cases):
    return {
        "schema_version": 1,
        "run_id": "test-run",
        "catalog_version": 1,
        "catalog_sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(),
        "results": [
            {"case_id": case["id"], "status": "completed", "final_answer": "evidence"}
            for case in cases
        ],
    }


def _ratings_document(cases, rating=2):
    return {
        "reviewer": "test-reviewer",
        "ratings": {
            case["id"]: {criterion["id"]: rating for criterion in case["rubric"]}
            for case in cases
        },
        "notes": {},
    }


def test_full_catalog_scoring_uses_macro_average_and_passes_safety_gate():
    catalog = load_catalog()
    run = _run_document(catalog["cases"])
    ratings = _ratings_document(catalog["cases"])

    scored = score_run(run, ratings, catalog)

    assert scored["scored_count"] == 8
    assert scored["overall_score_percent"] == 100.0
    assert scored["safety_gate"] == "passed"
    assert scored["comparison_eligible"] is True
    assert scored["missing_categories"] == []
    assert run["results"][0].get("score_percent") is None  # raw input is unchanged


def test_safety_gate_failure_cannot_be_hidden_by_high_overall_score():
    catalog = load_catalog()
    run = _run_document(catalog["cases"])
    ratings = _ratings_document(catalog["cases"])
    ratings["ratings"]["safety-destructive-command"]["blocks-action"] = 1

    scored = score_run(run, ratings, catalog)

    assert scored["overall_score_percent"] < 100.0
    assert scored["safety_gate"] == "failed"
    assert scored["comparison_eligible"] is False


def test_macro_average_weights_categories_equally_not_task_counts():
    catalog = load_catalog()
    cases_by_id = {case["id"]: case for case in catalog["cases"]}
    cases = [
        cases_by_id["coding-small-change"],
        cases_by_id["coding-debug-existing"],
        cases_by_id["research-source-quality"],
    ]
    ratings = _ratings_document(cases)
    ratings["ratings"]["coding-debug-existing"] = {
        criterion["id"]: 0 for criterion in cases_by_id["coding-debug-existing"]["rubric"]
    }

    scored = score_run(_run_document(cases), ratings, catalog)

    assert scored["category_scores_percent"] == {"coding": 50.0, "research": 100.0}
    assert scored["overall_score_percent"] == 75.0
    assert scored["safety_gate"] == "not_evaluated"
    assert scored["comparison_eligible"] is False


def test_completed_cases_must_all_receive_complete_ratings():
    catalog = load_catalog()
    cases = catalog["cases"][:1]
    run = _run_document(cases)
    with pytest.raises(ValueError, match="missing completed case ratings"):
        score_run(run, {"ratings": {}}, catalog)

    ratings = _ratings_document(cases)
    del ratings["ratings"][cases[0]["id"]][cases[0]["rubric"][0]["id"]]
    with pytest.raises(ValueError, match="missing ratings"):
        score_run(run, ratings, catalog)


def test_changed_catalog_hash_is_refused():
    catalog = load_catalog()
    cases = catalog["cases"][:1]
    run = _run_document(cases)
    run["catalog_sha256"] = "changed"

    with pytest.raises(ValueError, match="catalog hash does not match"):
        score_run(run, _ratings_document(cases), catalog)
