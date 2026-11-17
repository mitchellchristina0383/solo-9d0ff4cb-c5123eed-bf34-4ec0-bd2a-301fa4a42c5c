"""Tests for per-collector sequenced batch ingestion.

Covers the duty-officer requirements: timeout retries must be idempotent,
overlapping batches return the original log ids, only the contiguous new tail
is appended, gaps or changed history reject the whole batch atomically, and
the result holds under concurrent submissions and across restarts.  Also
verifies that a backup snapshot pairs collector progress with the logs of the
same committed state.
"""

import sqlite3
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from joblog.main import create_app


@pytest.fixture()
def env(tmp_path: Path):
    app = create_app(data_dir=tmp_path)
    client = TestClient(app)
    with client:
        yield {
            "app": app,
            "client": client,
            "root": tmp_path,
            "db": app.state.database,
            "manager": app.state.manager,
            "restores": app.state.restore_dir,
        }


def submit(client: TestClient, collector: str, entries: list[dict]) -> object:
    return client.post(
        f"/collectors/{collector}/logs/batch", json={"entries": entries}
    )


def batch(start: int, count: int, *, job: str = "job-a", prefix: str = "m") -> list[dict]:
    return [
        {"seq": start + i, "job_id": job, "message": f"{prefix}-{start + i}"}
        for i in range(count)
    ]


def ok_submit(client: TestClient, collector: str, entries: list[dict]) -> dict:
    response = submit(client, collector, entries)
    assert response.status_code == 200, response.text
    return response.json()


def stored_mapping(env, collector: str) -> dict[int, tuple[int, str, str]]:
    """seq -> (log_id, job_id, message) straight from the database."""
    with env["db"].reader() as conn:
        rows = conn.execute(
            "SELECT seq, log_id, job_id, message FROM collector_logs"
            " WHERE collector = ? ORDER BY seq",
            (collector,),
        ).fetchall()
    return {r["seq"]: (r["log_id"], r["job_id"], r["message"]) for r in rows}


def job_log_count(env) -> int:
    with env["db"].reader() as conn:
        return conn.execute("SELECT COUNT(*) FROM job_logs").fetchone()[0]


def test_batch_append_progress_and_plain_logs(env):
    client = env["client"]
    receipt = ok_submit(client, "col-1", batch(1, 3))

    assert [r["status"] for r in receipt["results"]] == ["appended"] * 3
    assert [r["seq"] for r in receipt["results"]] == [1, 2, 3]
    ids = [r["log_id"] for r in receipt["results"]]
    assert ids == sorted(ids) and len(set(ids)) == 3
    assert receipt["appended"] == 3 and receipt["duplicated"] == 0
    assert receipt["next_seq"] == 4

    # Batch rows are ordinary committed logs, visible through GET /logs.
    page = client.get("/logs", params={"job_id": "job-a"}).json()
    assert [item["message"] for item in page["items"]] == ["m-1", "m-2", "m-3"]

    progress = client.get("/collectors/col-1/progress").json()
    assert progress["next_seq"] == 4 and progress["stored"] == 3

    listing = client.get("/collectors/col-1/logs").json()
    assert [item["seq"] for item in listing["items"]] == [1, 2, 3]
    assert [item["log_id"] for item in listing["items"]] == ids
    tail = client.get("/collectors/col-1/logs", params={"after_seq": 2}).json()
    assert [item["seq"] for item in tail["items"]] == [3]


def test_overlap_retry_returns_original_ids(env):
    """A timed-out batch that is resent must reproduce the original receipt."""
    client = env["client"]
    first = ok_submit(client, "col-1", batch(1, 3))
    first_ids = {r["seq"]: r["log_id"] for r in first["results"]}

    # Full retry of the same batch: everything is a duplicate, same ids.
    retry = ok_submit(client, "col-1", batch(1, 3))
    assert [(r["seq"], r["log_id"]) for r in retry["results"]] == [
        (r["seq"], r["log_id"]) for r in first["results"]
    ]
    assert all(r["status"] == "duplicate" for r in retry["results"])
    assert retry["appended"] == 0 and retry["duplicated"] == 3
    assert retry["next_seq"] == 4
    assert job_log_count(env) == 3

    # Overlapping batch [2..5]: 2,3 acknowledged with original ids, 4,5 new.
    overlap = ok_submit(client, "col-1", batch(2, 4))
    by_seq = {r["seq"]: r for r in overlap["results"]}
    assert by_seq[2]["status"] == "duplicate"
    assert by_seq[2]["log_id"] == first_ids[2]
    assert by_seq[3]["log_id"] == first_ids[3]
    assert by_seq[4]["status"] == "appended"
    assert by_seq[5]["status"] == "appended"
    assert overlap["next_seq"] == 6
    assert job_log_count(env) == 5

    mapping = stored_mapping(env, "col-1")
    assert sorted(mapping) == [1, 2, 3, 4, 5]
    assert mapping[2][0] == first_ids[2]


def test_changed_old_sequence_rejects_whole_batch(env):
    client = env["client"]
    ok_submit(client, "col-1", batch(1, 3))
    before = job_log_count(env)

    forged = batch(2, 4)
    forged[1]["message"] = "tampered"  # seq 3 no longer matches history
    response = submit(client, "col-1", forged)
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error"] == "content_mismatch"
    assert detail["seq"] == 3

    # Nothing changed: no new logs, progress untouched, mapping intact.
    assert job_log_count(env) == before
    progress = client.get("/collectors/col-1/progress").json()
    assert progress["next_seq"] == 4 and progress["stored"] == 3
    assert sorted(stored_mapping(env, "col-1")) == [1, 2, 3]


def test_rejection_rolls_back_partial_batch(env):
    """A mismatch late in the batch must roll back earlier accepted entries."""
    client = env["client"]
    ok_submit(client, "col-1", batch(1, 3))
    before = job_log_count(env)

    entries = [
        {"seq": 2, "job_id": "job-a", "message": "m-2"},  # duplicate, ok
        {"seq": 3, "job_id": "job-a", "message": "CHANGED"},  # mismatch -> 409
        {"seq": 4, "job_id": "job-a", "message": "m-4"},  # would be new
        {"seq": 5, "job_id": "job-a", "message": "m-5"},  # would be new
    ]
    response = submit(client, "col-1", entries)
    assert response.status_code == 409
    assert job_log_count(env) == before
    assert client.get("/collectors/col-1/progress").json()["next_seq"] == 4
    assert sorted(stored_mapping(env, "col-1")) == [1, 2, 3]


def test_sequence_gaps_rejected(env):
    client = env["client"]

    # A new collector must start at seq 1.
    response = submit(client, "col-1", batch(2, 2))
    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "sequence_gap"
    assert client.get("/collectors/col-1/progress").status_code == 404
    assert job_log_count(env) == 0

    ok_submit(client, "col-1", batch(1, 3))
    before = job_log_count(env)

    # Jump over seq 4.
    response = submit(client, "col-1", batch(5, 2))
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error"] == "sequence_gap"
    assert detail["expected_seq"] == 4 and detail["received_seq"] == 5

    # Jump inside the batch itself.
    response = submit(
        client,
        "col-1",
        [
            {"seq": 4, "job_id": "job-a", "message": "m-4"},
            {"seq": 6, "job_id": "job-a", "message": "m-6"},
        ],
    )
    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "sequence_gap"

    # Duplicated sequence inside the batch is a gap too (not ascending).
    response = submit(
        client,
        "col-1",
        [
            {"seq": 4, "job_id": "job-a", "message": "m-4"},
            {"seq": 4, "job_id": "job-a", "message": "m-4"},
        ],
    )
    assert response.status_code == 409

    assert job_log_count(env) == before
    assert client.get("/collectors/col-1/progress").json()["next_seq"] == 4
    # The stream still continues cleanly afterwards.
    receipt = ok_submit(client, "col-1", batch(4, 1))
    assert receipt["next_seq"] == 5


def test_batch_validation(env):
    client = env["client"]
    assert submit(client, "col-1", []).status_code == 422
    assert (
        submit(client, "col-1", [{"seq": 0, "job_id": "j", "message": "m"}]).status_code
        == 422
    )
    assert (
        submit(client, "col-1", [{"seq": 1, "job_id": "", "message": "m"}]).status_code
        == 422
    )
    assert job_log_count(env) == 0


def test_unknown_collector_endpoints(env):
    client = env["client"]
    assert client.get("/collectors/nope/progress").status_code == 404
    assert client.get("/collectors/nope/logs").status_code == 404
    assert client.get("/collectors/bad name/progress").status_code == 422


def test_concurrent_identical_batches_same_collector(env):
    """Timeout retries racing each other: one appends, the rest duplicate."""
    client = env["client"]
    entries = batch(1, 10)
    receipts: list[dict] = []
    errors: list[BaseException] = []

    def worker():
        try:
            response = submit(client, "col-1", entries)
            assert response.status_code == 200, response.text
            receipts.append(response.json())
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(receipts) == 8
    # Every racer gets the same seq -> log_id mapping; each seq was appended
    # by exactly one racer and reported as duplicate by the other seven.
    mappings = [{r["seq"]: r["log_id"] for r in receipt["results"]} for receipt in receipts]
    for mapping in mappings[1:]:
        assert mapping == mappings[0]
    for receipt in receipts:
        assert receipt["next_seq"] == 11
    assert sum(r["appended"] for r in receipts) == 10
    assert sum(r["duplicated"] for r in receipts) == 7 * 10
    assert job_log_count(env) == 10
    assert client.get("/collectors/col-1/progress").json()["next_seq"] == 11


def test_concurrent_interleaved_batches_converge(env):
    """Interleaved overlapping batches from one collector stay gapless.

    Batches that lose the race see a gap and are rejected; a real collector
    retries after a conflict, so workers retry until accepted.  The final
    stream must be exactly seq 1..10, one job_logs row each, and every
    accepted receipt must agree on the seq -> log_id mapping.
    """
    client = env["client"]
    batches = [batch(1, 5), batch(6, 5), batch(1, 10), batch(4, 5)]
    receipts: list[dict] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(entries: list[dict]):
        try:
            for _ in range(50):
                response = submit(client, "col-1", entries)
                if response.status_code == 200:
                    with lock:
                        receipts.append(response.json())
                    return
                assert response.status_code == 409, response.text
                assert response.json()["detail"]["error"] in (
                    "sequence_gap",
                    "content_mismatch",
                )
            raise AssertionError("batch never accepted after retries")
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(b,)) for b in batches]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(receipts) == len(batches)

    # All accepted receipts agree on one mapping.
    mapping: dict[int, int] = {}
    for receipt in receipts:
        for r in receipt["results"]:
            assert r["log_id"] == mapping.setdefault(r["seq"], r["log_id"])
    assert sorted(mapping) == list(range(1, 11))

    # Exactly one job_logs row per sequence number, contiguous stream.
    assert job_log_count(env) == 10
    stored = stored_mapping(env, "col-1")
    assert sorted(stored) == list(range(1, 11))
    assert {seq: v[0] for seq, v in stored.items()} == mapping
    progress = client.get("/collectors/col-1/progress").json()
    assert progress["next_seq"] == 11 and progress["stored"] == 10


def test_restart_preserves_progress_and_ids(env):
    client = env["client"]
    first = ok_submit(client, "col-1", batch(1, 3))
    root = env["root"]

    # Simulated service restart: a fresh app over the same data directory.
    restarted = create_app(data_dir=root)
    with TestClient(restarted) as client2:
        retry = ok_submit(client2, "col-1", batch(1, 3))
        assert [(r["seq"], r["log_id"]) for r in retry["results"]] == [
            (r["seq"], r["log_id"]) for r in first["results"]
        ]
        assert retry["duplicated"] == 3 and retry["next_seq"] == 4

        progress = client2.get("/collectors/col-1/progress").json()
        assert progress["next_seq"] == 4 and progress["stored"] == 3

        # New entries continue the stream after the restart.
        nxt = ok_submit(client2, "col-1", batch(4, 2))
        assert [r["status"] for r in nxt["results"]] == ["appended"] * 2
        assert nxt["next_seq"] == 6

    assert job_log_count(env) == 5


def test_backup_snapshot_pairs_logs_and_progress(env):
    """The backup's collector progress must match the logs in the same snapshot."""
    client = env["client"]
    ok_submit(client, "col-1", batch(1, 5))
    ok_submit(client, "col-2", batch(1, 2, job="job-b", prefix="n"))
    backup = client.post("/backups").json()
    assert client.get(f"/backups/{backup['id']}/verify").json()["verified"] is True

    # More ingestion after the backup must not leak into it.
    ok_submit(client, "col-1", batch(6, 2))

    restored = client.post(
        f"/backups/{backup['id']}/restore", json={"target_name": "snap.db"}
    )
    assert restored.status_code == 200, restored.text
    assert restored.json()["row_count"] == 7

    with sqlite3.connect(env["restores"] / "snap.db") as conn:
        states = dict(
            conn.execute("SELECT collector, next_seq FROM collector_states").fetchall()
        )
        assert states == {"col-1": 6, "col-2": 3}
        rows = conn.execute(
            "SELECT c.collector, c.seq, c.job_id, c.message, l.job_id, l.message"
            " FROM collector_logs c JOIN job_logs l ON l.id = c.log_id"
            " ORDER BY c.collector, c.seq"
        ).fetchall()
    # Every stored entry resolves to a log row inside the same snapshot with
    # identical content, and the sequences are contiguous per collector.
    assert [(r[0], r[1]) for r in rows] == [("col-1", i) for i in range(1, 6)] + [
        ("col-2", i) for i in range(1, 3)
    ]
    for _, _, job_id, message, log_job, log_message in rows:
        assert (job_id, message) == (log_job, log_message)


def test_backup_freeze_excludes_concurrent_batch(env):
    """A batch committed after the snapshot point is absent from the backup,
    together with its progress — logs and progress stay in the same state."""
    manager = env["manager"]
    db = env["db"]
    ok_submit(env["client"], "col-1", batch(1, 5))

    def ingest_after_freeze():
        # Same writes the endpoint does, on a separate writer connection.
        with db.writer() as conn:
            for seq in (6, 7):
                cur = conn.execute(
                    "INSERT INTO job_logs (job_id, message) VALUES ('job-a', ?)",
                    (f"m-{seq}",),
                )
                conn.execute(
                    "INSERT INTO collector_logs"
                    " (collector, seq, log_id, job_id, message)"
                    " VALUES ('col-1', ?, ?, 'job-a', ?)",
                    (seq, cur.lastrowid, f"m-{seq}"),
                )
            conn.execute(
                "UPDATE collector_states SET next_seq = 8 WHERE collector = 'col-1'"
            )

    result = manager.create(after_freeze=ingest_after_freeze)
    assert result.snapshot_row_count == 5

    restored = env["client"].post(
        f"/backups/{result.id}/restore", json={"target_name": "frozen.db"}
    )
    assert restored.status_code == 200
    with sqlite3.connect(env["restores"] / "frozen.db") as conn:
        next_seq = conn.execute(
            "SELECT next_seq FROM collector_states WHERE collector = 'col-1'"
        ).fetchone()[0]
        seqs = [
            r[0]
            for r in conn.execute(
                "SELECT seq FROM collector_logs WHERE collector = 'col-1' ORDER BY seq"
            )
        ]
        logs = conn.execute("SELECT COUNT(*) FROM job_logs").fetchone()[0]
    # Progress, mapping and logs describe the same frozen state: 5 entries.
    assert next_seq == 6
    assert seqs == [1, 2, 3, 4, 5]
    assert logs == 5


def test_openapi_lists_collector_paths(env):
    schema = env["client"].get("/openapi.json").json()
    for path in (
        "/collectors/{collector}/logs/batch",
        "/collectors/{collector}/logs",
        "/collectors/{collector}/progress",
    ):
        assert path in schema["paths"]
    assert env["client"].get("/docs").status_code == 200
