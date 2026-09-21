"""GPU-isolated, memory-admitted batch sweep; complete outputs are preserved."""
import argparse,json,time,traceback,shutil
from pathlib import Path
import numpy as np
import torch
from yue2.pipeline import YuE2Pipeline
from yue2.batching import BatchGraphAR,batch_memory_estimate,BatchAdmissionError
from yue2.cuda_graph import GraphAR
from yue2.protocol import SongRequest,token_prefixes,CODEC_OFFSET
from yue2.sampling import distribution
from yue2.benchmark import environment
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
requests=[json.loads(x) for x in (root/'requests.jsonl').read_text().splitlines()]
report={'started_at':time.time(),'requests':requests,'environment':environment(),'probes':[],'runs':[],'errors':[], 'note':'RPM excludes HTTP, saving and queue waiting. All rows use the same request within a workload. Forward probes force identical teacher tokens. Guarded maximum is specific to these inputs/default token budgets, not a hardware absolute maximum.'}
def save():
 p=out/'report.tmp';p.write_text(json.dumps(report,ensure_ascii=False,indent=2));p.replace(out/'report.json')
def stage(name):print(json.dumps({'timestamp':time.time(),'stage':name}),flush=True)
pipe=None
try:
 stage('load')
 pipe=YuE2Pipeline.from_pretrained(root/'models/YuE2-3B',vae=root/'models/YuE2-Vae',device='cuda',resident_models=True,memory_budget_gib=30,progress=False,local_files_only=True)
 pipe.preload();report['weights']=pipe.weights
 config=pipe.generation_config;model=pipe._model
 # Worst legal generated ABC length, longest input, full semantic token budget.
 worst_prefix=max([token_prefixes(SongRequest(**r),pipe.tokenizer,[0]*config.abc.max_tokens) for r in requests],key=len)
 report['admission']=[batch_memory_estimate(model,[worst_prefix]*n,config.semantic.max_tokens) for n in range(2,65)]
 allowed=[x['batch_size'] for x in report['admission'] if x['allowed']]
 if not allowed:raise BatchAdmissionError('No batch>=2 fits conservative worst-prefix admission')
 safe_max=max(allowed);report['safe_max_batch']=safe_max;report['worst_prefix_length']=len(worst_prefix);report['semantic_budget']=config.semantic.max_tokens
 stage('admitted_max:'+str(safe_max));save()
 # GPU row-isolation tests have already passed before entering this script.
 stage('warmup')
 warm=pipe(**requests[0]);warm.save_artifacts(out/'warmup'/'0');del warm
 fixture=root/'results/benchmark-gpu5/reference/000-001'
 prefix=np.load(fixture/'prefix.npy').tolist();teacher=(np.load(fixture/'semantic.npy')[:256]+CODEC_OFFSET).tolist()
 for n in range(1,safe_max+1):
  for repeat in range(3):
   for kind in (['forward','forward_sampling'] if repeat%2==0 else ['forward_sampling','forward']):
    stage(f'probe:{n}:{repeat}:{kind}');graph=None
    try:
     graph=GraphAR(model,[prefix],config.semantic.max_tokens) if n==1 else BatchGraphAR(model,[prefix]*n,config.semantic.max_tokens)
     logits=graph.prefill();tokens=torch.tensor(teacher,device='cuda',dtype=torch.long);gens=[torch.Generator(device='cuda').manual_seed(831001+i) for i in range(n)];history=[]
     torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();began=time.time()
     for step in range(len(teacher)):
      if kind=='forward_sampling':
       for row in range(n):
        scores=distribution(logits[row:row+1],config.semantic,history,step,'semantic')
        int(torch.multinomial(scores.softmax(-1),1,generator=gens[row]).item())
      tok=tokens[step:step+1]
      logits=graph.step(tok) if n==1 else graph.step_batch(tok.reshape(1,1).expand(n,1))
      history.append(teacher[step])
     torch.cuda.synchronize();seconds=time.perf_counter()-t
     report['probes'].append({'batch':n,'repeat':repeat,'kind':kind,'seconds':seconds,'forward_tokens':len(teacher)*n,'tokens_per_second':len(teacher)*n/seconds,'started_at':began,'finished_at':time.time(),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30});save()
    finally:
     if graph is not None:graph.close()
     del graph
     logits=None;tokens=None;gens=None;scores=None
  # A new process is not required; clear unreferenced graph pools between sizes.
  torch.cuda.empty_cache()
 def full(n,index,repeat):
  name=['short','long'][index];stage(f'full:{name}:{n}:{repeat}')
  if shutil.disk_usage(out).free<10*2**30:raise RuntimeError('Stop: less than 10GiB disk free')
  folder=out/'full'/name/f'batch-{n:02d}'/f'repeat-{repeat}';folder.mkdir(parents=True,exist_ok=False)
  pair=[dict(requests[index]) for _ in range(n)];(folder/'inputs.json').write_text(json.dumps(pair,ensure_ascii=False,indent=2))
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.time();t=time.perf_counter()
  songs=[pipe(**pair[0],on_stage=stage)] if n==1 else pipe.generate_batch(pair,on_stage=stage)
  torch.cuda.synchronize();seconds=time.perf_counter()-t
  valid=[not any(s.truncated.values()) for s in songs];duration=[len(s.audio)/s.sample_rate for s in songs]
  row={'case':name,'batch':n,'repeat':repeat,'started_at':started,'finished_at':time.time(),'seconds':seconds,'rpm':sum(valid)*60/seconds,'successful_songs':sum(valid),'audio_seconds':duration,'normalized_audio_throughput':sum(d for d,v in zip(duration,valid) if v)/seconds,'truncated':[s.truncated for s in songs],'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'timings':[s.timing for s in songs]}
  for i,s in enumerate(songs):s.save_artifacts(folder/str(i))
  report['runs'].append(row);save();stage('result:'+json.dumps({k:row[k] for k in ['case','batch','repeat','seconds','rpm','audio_seconds','successful_songs']}));del songs
  torch.cuda.empty_cache()
 # Ascend through every integer, including 3,4,5. Alternate workload order.
 for n in range(1,safe_max+1):
  for index in ([0,1] if n%2 else [1,0]):full(n,index,0)
 # Recheck all candidates leading either raw RPM or duration-normalized throughput.
 finalists={}
 for index,name in enumerate(['short','long']):
  rows=[r for r in report['runs'] if r['case']==name and r['successful_songs']==r['batch']]
  finalists[name]=sorted({r['batch'] for metric in ['rpm','normalized_audio_throughput'] for r in sorted(rows,key=lambda r:r[metric],reverse=True)[:2]},reverse=True)
 report['finalists']=finalists;save()
 for index,name in enumerate(['short','long']):
  for n in finalists[name]:full(n,index,1)
 report['completed_at']=time.time();save();stage('COMPLETE')
except BaseException as error:
 report['errors'].append({'time':time.time(),'error':repr(error),'traceback':traceback.format_exc()});save();traceback.print_exc();raise
finally:
 if pipe is not None:pipe.close()
