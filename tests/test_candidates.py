from concurrent.futures import ThreadPoolExecutor
import pytest
from yue2.service_store import JobStore, QueueFull, IdempotencyConflict
from test_service import service, FakePipeline, REQUEST, wait_for


def test_atomic_admission_and_stable_retry(tmp_path):
    store = JobStore(tmp_path / 'jobs.db')
    store.submit(REQUEST, 2)
    with pytest.raises(QueueFull):
        store.submit(REQUEST, 2, 'pair', n=2)
    with store.connect() as db:
        assert db.execute('SELECT count(*) FROM jobs').fetchone()[0] == 1
    group, created = store.submit(REQUEST, 3, 'pair', n=2)
    assert created and len(group['candidates']) == 2
    assert [c['seed'] for c in group['candidates']] == [42, 43]
    assert store.submit(REQUEST, 3, 'pair', n=2) == (group, False)
    with pytest.raises(IdempotencyConflict):
        store.submit(REQUEST, 3, 'pair', n=1)


def test_concurrent_retries_create_only_two_children(tmp_path):
    store = JobStore(tmp_path / 'jobs.db')
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: store.submit(REQUEST, 2, 'one', n=2), range(8)))
    assert sum(created for _, created in responses) == 1
    assert len({group['id'] for group, _ in responses}) == 1
    assert len(store.claim_many(8)) == 2


def test_group_progress_failure_restart_and_cancel(tmp_path):
    path = tmp_path / 'jobs.db'
    store = JobStore(path)
    group, _ = store.submit(REQUEST, 2, n=2)
    first, second = group['candidate_ids']
    store.claim_many(2)
    store.finish(first, 'succeeded', result={'audio_url': '/one'})
    assert store.get(group['id'])['status'] == 'running'
    recovered = JobStore(path)
    recovered.recover()
    result = recovered.get(group['id'])
    assert result['status'] == 'partial_failed'
    assert result['candidates'][1]['error']['code'] == 'worker_interrupted'
    assert len(result['result']['outputs']) == 1
    assert recovered.cancel(group['id'])['status'] == 'partial_failed'
    new, _ = recovered.submit({**REQUEST, 'seed': 2**63-1}, 2, n=2)
    assert [c['seed'] for c in new['candidates']] == [2**63-1, 0]
    assert recovered.cancel(new['id'])['status'] == 'cancelled'
    assert recovered.claim_many(2) == []


def test_pair_http_results_and_single_compatibility(tmp_path):
    with service(tmp_path, backend='torch', ar_nar_overlap=False) as (client, fake, app):
        response = client.post('/v1/jobs', json={**REQUEST, 'n': 2})
        assert response.status_code == 202
        group = response.json()
        result = wait_for(lambda: (r if (r := client.get('/v1/jobs/' + group['id']).json())['status'] == 'succeeded' else None))
        assert len(result['result']['outputs']) == 2
        for c in result['candidates']:
            assert client.get(c['result']['audio_url']).status_code == 200
        assert fake.calls == 2
        assert client.post('/v1/jobs', json={**REQUEST, 'n': 3}).status_code == 422
        assert client.post('/v1/jobs', json={**REQUEST, 'n': True}).status_code == 422
        single = client.post('/v1/jobs', json=REQUEST).json()
        assert 'candidates' not in single


def test_group_cancels_running_and_queued_children(tmp_path):
    fake = FakePipeline(blocked=True)
    with service(tmp_path, fake, backend='torch', ar_nar_overlap=False) as (client, _, app):
        group = client.post('/v1/jobs', json={**REQUEST, 'n': 2}).json()
        assert fake.started.wait(2)
        client.post('/v1/jobs/' + group['id'] + '/cancel')
        wait_for(lambda: client.get('/v1/jobs/' + group['id']).json()['status'] == 'cancelled')
        assert fake.calls == 1
        fake.release.set()
