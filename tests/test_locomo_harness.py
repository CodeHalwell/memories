"""The LoCoMo harness runs end to end on a tiny fixture in the dataset's format."""

from __future__ import annotations

import json

from agent_memory.eval.locomo import (
    load_locomo,
    parse_locomo_datetime,
    run_policy,
    run_retrieval,
)

FIXTURE = [
    {
        "sample_id": f"conv-{n}",
        "conversation": {
            "speaker_a": "Ana", "speaker_b": "Ben",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Ana", "dia_id": "D1:1", "text": "Hey Ben, how are you?"},
                {"speaker": "Ben", "dia_id": "D1:2", "text": "Good! I adopted a beagle called Pickle last week."},
                {"speaker": "Ana", "dia_id": "D1:3", "text": "Lovely. I started learning the cello in March."},
                {"speaker": "Ben", "dia_id": "D1:4", "text": "ok cool"},
            ],
            "session_2_date_time": "10:00 am on 1 June, 2023",
            "session_2": [
                {"speaker": "Ana", "dia_id": "D2:1", "text": "My cello teacher is called Mrs Okafor."},
                {"speaker": "Ben", "dia_id": "D2:2", "text": "Pickle chewed my running shoes!",
                 "blip_caption": "a beagle with a shoe"},
            ],
        },
        "qa": [
            {"question": "What is the name of Ben's dog?", "answer": "Pickle", "evidence": ["D1:2"], "category": 4},
            {"question": "Who teaches Ana the cello?", "answer": "Mrs Okafor", "evidence": ["D2:1"], "category": 4},
            {"question": "What did the dog chew?", "answer": "running shoes", "evidence": ["D2:2"], "category": 1},
            {"question": "adversarial", "adversarial_answer": "x", "evidence": ["D1:1"], "category": 5},
        ],
    }
    for n in range(2)
]


def test_parse_locomo_datetime():
    assert parse_locomo_datetime("1:56 pm on 8 May, 2023") == "2023-05-08T13:56:00+00:00"
    assert parse_locomo_datetime("12:05 am on 1 January, 2024") == "2024-01-01T00:05:00+00:00"


def test_loader(tmp_path):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(FIXTURE))
    convs = load_locomo(path)
    assert len(convs) == 2 and len(convs[0].turns) == 6
    assert "beagle with a shoe" in convs[0].turns[-1].text


async def test_retrieval_experiment_runs(tmp_path):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(FIXTURE))
    res = await run_retrieval(load_locomo(path), ["raw_bm25", "full"], [1, 5], "hashing")
    assert res["raw_bm25"]["n"] == 6  # adversarial questions are excluded
    assert res["raw_bm25"]["recall@5"] >= 0.8
    assert 0.0 <= res["full"]["recall@1"] <= 1.0


async def test_policy_experiment_runs(tmp_path):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(FIXTURE * 2))  # 4 conversations for 2 folds
    data = json.loads(path.read_text())
    for i, s in enumerate(data):
        s["sample_id"] = f"conv-{i}"
    path.write_text(json.dumps(data))
    res = await run_policy(load_locomo(path), "hashing", folds=2, labels=["replay@1"])
    assert res["turns"] == 24
    assert "deployed_lexical_heuristic" in res["policies"]
