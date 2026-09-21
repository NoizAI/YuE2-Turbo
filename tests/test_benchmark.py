import json

import pytest

from yue2.benchmark import load_requests, summarize


def test_summary_reports_incomplete_outputs_without_treating_them_as_fast_successes():
    result = summarize([{"status": "succeeded", "generation_seconds": 80, "rtf": .4},
                        {"status": "succeeded", "generation_seconds": 60, "rtf": .3},
                        {"status": "truncated", "generation_seconds": 2, "rtf": .1},
                        {"status": "failed"}])
    assert result["attempted"] == 4 and result["succeeded"] == 2
    assert result["failed"] == 1 and result["truncated"] == 1
    assert result["p50_generation_seconds"] == 70 and result["p95_generation_seconds"] == 80
    assert "p50_generation_seconds" not in summarize([{"status": "failed"}])


def test_request_loader_validates_native_requests(tmp_path):
    source = tmp_path / "requests.jsonl"
    source.write_text(json.dumps({"style": "piano", "lyrics": "words", "seed": 42}) + "\n")
    assert load_requests(source)[0]["seed"] == 42
    source.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        load_requests(source)
