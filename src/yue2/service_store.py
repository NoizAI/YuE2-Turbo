"""Durable single-machine job queue. SQLite transactions bound admission atomically."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid


TERMINAL = {"succeeded", "truncated", "failed", "cancelled"}


class QueueFull(Exception):
    pass


class IdempotencyConflict(Exception):
    pass


class JobStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, created REAL NOT NULL,
                request TEXT NOT NULL, request_hash TEXT NOT NULL,
                idem_key TEXT UNIQUE, snapshot TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _write(db, job):
        job["updated_at"] = time.time()
        db.execute("UPDATE jobs SET status=?, snapshot=? WHERE id=?",
                   (job["status"], json.dumps(job), job["id"]))

    def recover(self):
        # Queued work survives; a process killed mid-generation cannot resume exact GPU state.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT snapshot FROM jobs WHERE status='running'").fetchall():
                job = json.loads(row[0])
                job.update(status="failed", stage="finished", finished_at=time.time(),
                           error={"code": "worker_interrupted", "message": "Worker stopped during generation; submit a new job."})
                self._write(db, job)

    def submit(self, request, max_pending, idem_key=None, n=1, admission_id=None):
        raw = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if type(n) is not int or n not in (1, 2):
            raise ValueError("n must be 1 or 2")
        digest = hashlib.sha256((raw if n == 1 else "pair:" + raw).encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if idem_key is not None:
                existing = db.execute("SELECT request_hash, snapshot FROM jobs WHERE idem_key=?", (idem_key,)).fetchone()
                if existing:
                    if existing[0] != digest:
                        raise IdempotencyConflict()
                    return self._snapshot(db, json.loads(existing[1])), False
            pending = db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if pending + n > max_pending:
                raise QueueFull()
            if n == 2:
                now, group_id = time.time(), uuid.uuid4().hex
                children = []
                for index in range(2):
                    child_request = {**request, "seed": (request.get("seed", 831001) + index) % (2**63)}
                    child_id = uuid.uuid4().hex
                    child = dict(id=child_id, group_id=group_id, index=index,
                                 seed=child_request["seed"], admission_id=admission_id, status="queued", stage="queued",
                                 created_at=now, updated_at=now, started_at=None, finished_at=None,
                                 cancel_requested=False, tokens={"abc": 0, "semantic": 0},
                                 result=None, error=None)
                    child_raw = json.dumps(child_request, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?)",
                               (child_id, "queued", now, child_raw,
                                hashlib.sha256(child_raw.encode()).hexdigest(), None, json.dumps(child)))
                    children.append(child_id)
                group = dict(id=group_id, kind="group", n=2, candidate_ids=children,
                             created_at=now, updated_at=now)
                # Group rows are metadata only: workers claim the two children.
                db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?)",
                           (group_id, "group", now, raw, digest, idem_key, json.dumps(group)))
                return self._snapshot(db, group), True
            now, job_id = time.time(), uuid.uuid4().hex
            job = dict(id=job_id, admission_id=admission_id, status="queued", stage="queued", created_at=now, updated_at=now,
                       started_at=None, finished_at=None, cancel_requested=False,
                       tokens={"abc": 0, "semantic": 0}, result=None, error=None)
            db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?)",
                       (job_id, "queued", now, raw, digest, idem_key, json.dumps(job)))
            return job, True

    def get(self, job_id):
        with self.connect() as db:
            db.execute("BEGIN")
            row = db.execute("SELECT snapshot FROM jobs WHERE id=?", (job_id,)).fetchone()
            return self._snapshot(db, json.loads(row[0])) if row else None

    @staticmethod
    def _snapshot(db, job):
        if job.get("kind") != "group":
            return job
        children = [json.loads(db.execute("SELECT snapshot FROM jobs WHERE id=?", (child_id,)).fetchone()[0])
                    for child_id in job["candidate_ids"]]
        states = [child["status"] for child in children]
        complete = all(state in TERMINAL for state in states)
        if not complete:
            status = "queued" if all(state == "queued" for state in states) else "running"
        elif all(state == "succeeded" for state in states):
            status = "succeeded"
        elif all(state in {"succeeded", "truncated"} for state in states):
            status = "truncated"
        elif all(state == "cancelled" for state in states):
            status = "cancelled"
        elif any(state in {"succeeded", "truncated"} for state in states):
            status = "partial_failed"
        else:
            status = "failed"
        starts = [c["started_at"] for c in children if c["started_at"] is not None]
        outputs = [{"id": c["id"], "index": c["index"], "seed": c["seed"], **c["result"]}
                   for c in children if c.get("result")]
        return {**job, "status": status, "stage": "finished" if complete else status,
                "candidates": children, "completed_count": sum(state in TERMINAL for state in states),
                "updated_at": max(c["updated_at"] for c in children),
                "started_at": min(starts) if starts else None,
                "finished_at": max(c["finished_at"] for c in children) if complete else None,
                "cancel_requested": any(c["cancel_requested"] for c in children),
                "tokens": {key: sum(c["tokens"].get(key, 0) for c in children) for key in ("abc", "semantic")},
                "result": {"outputs": outputs} if outputs else None, "error": None}

    def claim(self):
        claimed = self.claim_many(1)
        return claimed[0] if claimed else None

    def claim_many(self, limit):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("claim limit must be a positive integer")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT snapshot, request FROM jobs WHERE status='queued' ORDER BY created, id LIMIT ?",
                (limit,)).fetchall()
            claimed = []
            for row in rows:
                job = json.loads(row[0])
                job.update(status="running", stage="claimed_waiting", started_at=time.time())
                self._write(db, job)
                claimed.append((job, json.loads(row[1])))
            return claimed

    def progress(self, job_id, **fields):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT snapshot FROM jobs WHERE id=?", (job_id,)).fetchone()
            job = json.loads(row[0])
            if job["status"] == "running":
                job.update(fields)
                self._write(db, job)

    def cancel(self, job_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT snapshot FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return None
            job = json.loads(row[0])
            if job.get("kind") == "group":
                for child_id in job["candidate_ids"]:
                    child = json.loads(db.execute("SELECT snapshot FROM jobs WHERE id=?", (child_id,)).fetchone()[0])
                    if child["status"] not in TERMINAL:
                        child["cancel_requested"] = True
                        if child["status"] == "queued":
                            child.update(status="cancelled", stage="finished", finished_at=time.time())
                        self._write(db, child)
                return self._snapshot(db, job)
            if job["status"] not in TERMINAL:
                job["cancel_requested"] = True
                if job["status"] == "queued":
                    job.update(status="cancelled", stage="finished", finished_at=time.time())
                self._write(db, job)
            return job

    def finish(self, job_id, status, *, result=None, error=None):
        if status not in TERMINAL:
            raise ValueError("Expected terminal status")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = json.loads(db.execute("SELECT snapshot FROM jobs WHERE id=?", (job_id,)).fetchone()[0])
            if job["cancel_requested"]:
                status, result, error = "cancelled", None, None
            job.update(status=status, stage="finished", finished_at=time.time(), result=result, error=error)
            self._write(db, job)

    def load_snapshot(self):
        """Only candidates count as work; group rows never consume capacity."""
        sampled_at = time.time()
        with self.connect() as db:
            rows = db.execute("SELECT snapshot FROM jobs WHERE status IN ('queued','running')").fetchall()
        jobs = [json.loads(row[0]) for row in rows]
        return {"sampled_at": sampled_at, "active_requests": sum(j["status"] == "running" for j in jobs),
                "queue_depth": sum(j["status"] == "queued" for j in jobs),
                "items": [{"id": j["id"], "admission_id": j.get("admission_id"),
                           "stage": j["stage"], "units": 1} for j in jobs]}

    def active_cover_audio(self):
        with self.connect() as db:
            rows = db.execute(
                "SELECT request FROM jobs WHERE status IN ('queued','running')").fetchall()
        paths = set()
        for row in rows:
            audio = json.loads(row[0]).get("_cover_audio")
            if audio:
                paths.add(str(Path(audio).resolve()))
        return paths

    def artifact_records(self):
        """Return the minimal durable state needed by artifact retention."""
        with self.connect() as db:
            rows = db.execute("SELECT id, status, created, snapshot FROM jobs").fetchall()
        records = {}
        for row in rows:
            job = json.loads(row["snapshot"])
            records[row["id"]] = {
                "status": row["status"],
                "finished_at": job.get("finished_at"),
                "updated_at": job.get("updated_at"),
                "created_at": job.get("created_at", row["created"]),
            }
        return records
