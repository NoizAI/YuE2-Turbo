import pytest
import torch
from yue2.batching import BatchGraphAR, generate_tokens_batch
from yue2.cuda_graph import GraphAR
from yue2.modeling_yue2 import YuE2Config,YuE2ForCausalLM
from yue2.protocol import Sampling,ABC_END,VOCAB_SIZE

@pytest.mark.parametrize('size',[2,3,4,5,8])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_independent_rows_match_separate_graphs(device,size):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    previous=torch.get_num_threads();torch.set_num_threads(1)
    graphs=[]
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(801)
            model=YuE2ForCausalLM(YuE2Config(hidden_size=128,intermediate_size=256,num_hidden_layers=2,
                num_attention_heads=4,num_key_value_heads=2,head_dim=32,vocab_size=32,
                max_position_embeddings=64,max_latent_frames=64)).eval().to(device=device,dtype=torch.bfloat16 if device=='cuda' else torch.float32)
        prefixes=[[2,3,4,5] if row%2==0 else [6+row] for row in range(size)]
        batch=BatchGraphAR(model,prefixes,4,capture=device=='cuda');graphs.append(batch)
        singles=[GraphAR(model,[p],4,capture=device=='cuda') for p in prefixes];graphs.extend(singles)
        atol,rtol=(.02,.03) if device=='cuda' else (1e-6,1e-5)
        torch.testing.assert_close(batch.prefill(),torch.cat([g.prefill() for g in singles]),atol=atol,rtol=rtol)
        for tokens in [[7+row+step for row in range(size)] for step in range(3)]:
            got=batch.step_batch(torch.tensor(tokens,device=device).reshape(size,1))
            expected=torch.cat([g.step(t) for g,t in zip(singles,tokens)])
            torch.testing.assert_close(got,expected,atol=atol,rtol=rtol)
        assert batch.positions.tolist()==[len(p)+3 for p in prefixes]
        with pytest.raises(ValueError,match='budget'):batch.step_batch(torch.tensor([[1],[2]]))
    finally:
        for g in graphs:g.close()
        torch.set_num_threads(previous)


@pytest.mark.parametrize('size',[2,3,5])
def test_eos_is_independent_and_graph_is_closed(monkeypatch,size):
    import yue2.batching as module
    instances=[]
    class FakeGraph:
        attention_backend='test'
        def __init__(self,*args,**kwargs):self.steps=0;self.closed=False;instances.append(self)
        def logits(self,ids):
            x=torch.full((size,VOCAB_SIZE),-100.)
            for row,t in enumerate(ids):x[row,t]=100
            return x
        def prefill(self):return self.logits([ABC_END]+[3]*(size-1))
        def step_batch(self,tokens):self.steps+=1;return self.logits([4]+[ABC_END]*(size-1))
        def close(self):self.closed=True
    monkeypatch.setattr(module,'BatchGraphAR',FakeGraph)
    seen=[]
    rows=generate_tokens_batch(torch.nn.Linear(1,1),[[1]]*size,Sampling(temperature=0,min_tokens=0,max_tokens=4),list(range(size)),'abc',on_token=lambda *x:seen.append(x))
    assert [r[0] for r in rows]==[[]]+[[3] for _ in range(size-1)]
    assert [r[2] for r in rows]==[False]*size
    assert [r[1]['output_tokens'] for r in rows]==[1]+[2]*(size-1)
    assert seen==[(0,'abc',ABC_END)]+[(i,'abc',3) for i in range(1,size)]+[(i,'abc',ABC_END) for i in range(1,size)]
    assert instances[0].closed and instances[0].steps==1


@pytest.mark.parametrize('size',[2,3,5])
def test_sampling_rng_is_independent_and_cancellation_closes(monkeypatch,size):
    import yue2.batching as module
    instances=[]
    class FakeGraph:
        attention_backend='test'
        def __init__(self,*a,**k):self.closed=False;instances.append(self)
        def prefill(self):
            logits=torch.full((size,VOCAB_SIZE),-torch.inf)
            logits[:,:9]=0
            return logits
        def step_batch(self,tokens):return self.prefill()
        def close(self):self.closed=True
    monkeypatch.setattr(module,'BatchGraphAR',FakeGraph)
    cfg=Sampling(temperature=1,top_k=9,top_p=1,repetition_penalty=1,min_tokens=0,max_tokens=5)
    model=torch.nn.Linear(1,1)
    seeds=[15,28,39,42,60][:size]
    a=generate_tokens_batch(model,[[1]]*size,cfg,seeds,'abc')
    b=generate_tokens_batch(model,[[1]]*size,cfg,seeds[::-1],'abc')
    assert [x[0] for x in a]==[x[0] for x in b][::-1]
    assert all(r[2] and len(r[0])==5 for r in a)
    calls=[]
    with pytest.raises(InterruptedError):
        generate_tokens_batch(model,[[1]]*size,cfg,seeds,'abc',cancelled=lambda:len(calls)>0,on_token=lambda *x:calls.append(x))
    assert instances[-1].closed


@pytest.mark.parametrize('change',[
    {'backend':'vllm'}, {'resident_models':False}, {'offload_ar':True},
    {'quantization':'int8'}, {'device':torch.device('cpu')},
])
def test_unsupported_pipeline_rejected_before_loading(change):
    from types import SimpleNamespace
    from yue2.batching import generate_batch
    settings=dict(backend='torch',resident_models=True,offload_ar=False,
                  quantization='none',device=torch.device('cuda'))
    settings.update(change)
    pipe=SimpleNamespace(**settings)
    with pytest.raises(ValueError,match='requires CUDA'):
        generate_batch(pipe,[dict(style='pop',lyrics='hello')]*2)


def test_request_count_cfg_and_early_cancellation():
    from types import SimpleNamespace
    from yue2.batching import generate_batch
    pipe=SimpleNamespace(backend='torch',resident_models=True,offload_ar=False,
                         quantization='none',device=torch.device('cuda'))
    request=dict(style='pop',lyrics='hello')
    with pytest.raises(ValueError,match='2 to 64'):
        generate_batch(pipe,[request])
    with pytest.raises(ValueError,match='cfg_scale=1'):
        generate_batch(pipe,[dict(request,cfg_scale=2),request])
    with pytest.raises(ValueError,match='cot=full/melody'):
        generate_batch(pipe,[dict(request,cot='off'),request])
    with pytest.raises(InterruptedError,match='before batch'):
        generate_batch(pipe,[request,request],cancelled=lambda:True)


def test_memory_guard_rejects_before_cuda_allocation(monkeypatch):
    from types import SimpleNamespace
    from yue2.batching import BatchAdmissionError,batch_memory_estimate
    weight=SimpleNamespace(device=torch.device('cuda'),element_size=lambda:2)
    model=SimpleNamespace(model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=weight)),
                          config=SimpleNamespace(num_hidden_layers=28,num_key_value_heads=8,head_dim=128))
    gib=2**30
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda *a:(5*gib,32*gib))
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda *a:7*gib)
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda *a:7*gib)
    monkeypatch.setattr(torch.cuda,'get_per_process_memory_fraction',lambda *a:28/32)
    estimate=batch_memory_estimate(model,[[1]*100]*4,9000)
    assert estimate['kv_bytes']==2*28*4*9100*8*128*2
    assert not estimate['allowed']
    monkeypatch.setattr(torch,'zeros',lambda *a,**k:pytest.fail('Allocated before rejecting'))
    with pytest.raises(BatchAdmissionError,match='before KV allocation'):
        BatchGraphAR(model,[[1]*100]*4,9000)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda *a:(25*gib,32*gib))
    assert batch_memory_estimate(model,[[1]*100]*4,9000)['allowed']
