import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid

root = Path('/data/noiz-yue')
env = dict(line.split('=', 1) for line in (root / 'service.env').read_text().splitlines() if '=' in line)
base = 'http://127.0.0.1:8015'
headers = {'Authorization': 'Bearer ' + env['YUE2_API_KEY']}
report = {}

def request(method, path, body=None, extra=None):
    merged = dict(headers)
    merged.update(extra or {})
    data = None if body is None else json.dumps(body).encode()
    if data is not None:
        merged['Content-Type'] = 'application/json'
    req = urllib.request.Request(base + path, data=data, headers=merged, method=method)
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.status, json.load(response)

try:
    urllib.request.urlopen(base + '/v1/jobs/unknown', timeout=10)
    raise AssertionError('Missing auth unexpectedly accepted')
except urllib.error.HTTPError as error:
    assert error.code == 401
    report['unauthenticated_status'] = error.code

assert request('GET', '/health/ready')[0] == 200
payload = json.loads((root / 'requests.jsonl').read_text().splitlines()[0])
payload.pop('id', None)
idem = {'Idempotency-Key': 'gpu5-smoke-' + uuid.uuid4().hex}
started = time.monotonic()
status, job = request('POST', '/v1/jobs', payload, idem)
assert status == 202
status, duplicate = request('POST', '/v1/jobs', payload, idem)
assert status == 200 and duplicate['id'] == job['id']
report['idempotency_verified'] = True
status, queued = request('POST', '/v1/jobs', {**payload, 'seed': 987654321})
assert status == 202
request('POST', '/v1/jobs/' + queued['id'] + '/cancel')
last = None
while time.monotonic() - started < 1200:
    _, current = request('GET', '/v1/jobs/' + job['id'])
    state = (current['status'], current['stage'])
    if state != last:
        print(f'job={job["id"]} state={state} elapsed={time.monotonic() - started:.1f}s', flush=True)
        last = state
    if current['status'] in {'succeeded', 'truncated', 'failed', 'cancelled'}:
        break
    time.sleep(2)
else:
    raise TimeoutError('API task did not finish')
report['elapsed_seconds'] = time.monotonic() - started
report['job'] = current
_, cancelled = request('GET', '/v1/jobs/' + queued['id'])
report['cancelled_job'] = cancelled
assert cancelled['status'] == 'cancelled'
if current['status'] in {'succeeded', 'truncated'}:
    audio = urllib.request.urlopen(urllib.request.Request(base + current['result']['audio_url'], headers=headers), timeout=60).read()
    assert audio.startswith(b'fLaC')
    (root / 'results/api-smoke.flac').write_bytes(audio)
    report['audio_bytes'] = len(audio)
    report['audio_sha256'] = hashlib.sha256(audio).hexdigest()
    score = urllib.request.urlopen(urllib.request.Request(base + current['result']['score_url'], headers=headers), timeout=60).read()
    (root / 'results/api-smoke.abc').write_bytes(score)
(root / 'results/api-smoke.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
assert current['status'] == 'succeeded', current
print('API smoke passed: generation, auth, idempotency, cancellation, FLAC/ABC downloads.', flush=True)
