import importlib.util
import sys
from pathlib import Path
import pytest

from yue2 import service_store as store

def test_pair_metadata_not_counted_and_restart_recovers(tmp_path):
    db=store.JobStore(tmp_path/'jobs.db')
    group,new=db.submit({'seed':42},4,'key',n=2,admission_id='reservation')
    assert len(db.load_snapshot()['items'])==2
    assert db.load_snapshot()['queue_depth']==2
    same,created=db.submit({'seed':42},4,'key',n=2,admission_id='reservation')
    assert same['id']==group['id'] and not created
    db.claim_many(4)
    assert db.load_snapshot()['active_requests']==2
    assert {i['stage'] for i in db.load_snapshot()['items']}=={'claimed_waiting'}
    db.recover()
    assert not db.load_snapshot()['items']

def test_atomic_capacity_and_cancel(tmp_path):
    db=store.JobStore(tmp_path/'jobs.db')
    first,_=db.submit({'seed':42},3,'key',n=2)
    with pytest.raises(store.QueueFull): db.submit({'seed':43},3,'another',n=2)
    db.cancel(first['id'])
    assert not db.load_snapshot()['items']
    db.submit({'seed':43},3,'another',n=2)
