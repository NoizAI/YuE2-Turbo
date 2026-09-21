"""Offline paired batch=2 experiment. See docs/batch2.md for required fixtures."""
import argparse
import json,time,traceback,hashlib
from pathlib import Path
import numpy as np
import torch
from yue2.pipeline import YuE2Pipeline
from yue2.cuda_graph import GraphAR
from yue2.batching import BatchGraphAR
from yue2.protocol import CODEC_OFFSET
from yue2.benchmark import environment
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True,help='Root with models, requests.jsonl and reference traces')
parser.add_argument('--output',type=Path,required=True,help='New experiment directory')
args=parser.parse_args()
root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
report={'environment':environment(),'runs':[],'fixed_trace':[],'started_at':time.time(),'note':'Batch AR only; NAR/VAE sequential. Full calls return both results together. Fixed-trace timings exclude sampling and prefill.'}
def save():
 p=out/'report.json.tmp';p.write_text(json.dumps(report,ensure_ascii=False,indent=2));p.replace(out/'report.json')
def stage(s):print(json.dumps({'t':time.time(),'stage':s}),flush=True)
requests=[json.loads(x) for x in (root/'requests.jsonl').read_text().splitlines()]
report['requests']=requests
started=time.perf_counter()
pipe=YuE2Pipeline.from_pretrained(root/'models/YuE2-3B',vae=root/'models/YuE2-Vae',device='cuda',resident_models=True,memory_budget_gib=30,progress=False,local_files_only=True)
pipe.preload();report['load_seconds']=time.perf_counter()-started;report['weights']=pipe.weights;save()
try:
 for mode in ['serial','batch']:
  stage('warmup:'+mode)
  songs=[pipe(**requests[0])] if mode=='serial' else pipe.generate_batch([requests[0],requests[0]],on_stage=stage)
  for i,song in enumerate(songs):song.save_artifacts(out/'warmup'/mode/str(i))
  del songs
 # Equal work, different contexts and teacher tokens: isolates shared forward performance from song length changes.
 prefixes=[];teacher=[]
 for i in [0,1]:
  d=root/'results/benchmark-gpu5/reference'/f'000-{i:03d}'
  prefixes.append(np.load(d/'prefix.npy').tolist());teacher.append(np.load(d/'semantic.npy')[:512]+CODEC_OFFSET)
 tokens=torch.as_tensor(np.array(teacher),device='cuda',dtype=torch.long)
 examples={}
 for repeat in range(3):
  for mode in (['serial','batch'] if repeat%2==0 else ['batch','serial']):
   graphs=[];held=[]
   try:
    if mode=='serial':graphs=[GraphAR(pipe._model,[p],513) for p in prefixes]
    else:graphs=[BatchGraphAR(pipe._model,prefixes,513)]
    for g in graphs:g.prefill()
    torch.cuda.synchronize();t=time.perf_counter()
    if mode=='serial':
     for row,g in enumerate(graphs):
      for step in range(512):
       logits=g.step(tokens[row,step:step+1])
       if step in [0,127,511]:held.append(logits.clone())
    else:
     for step in range(512):
      logits=graphs[0].step_batch(tokens[:,step:step+1])
      if step in [0,127,511]:held.append(logits.clone())
    torch.cuda.synchronize();seconds=time.perf_counter()-t
    report['fixed_trace'].append({'mode':mode,'repeat':repeat,'seconds':seconds,'forward_tokens':1024,'forward_tokens_per_second':1024/seconds})
    if repeat==0:
     examples[mode]=torch.stack(held).reshape(2,3,-1).cpu() if mode=='serial' else torch.stack(held).permute(1,0,2).cpu()
    stage('fixed_trace:'+mode+':'+str(round(seconds,3)))
   finally:
    for g in graphs:g.close()
    del held
   save()
 diff=(examples['serial'].float()-examples['batch'].float()).abs()
 report['fixed_trace_logits']={'sampled_contexts':6,'max_absolute_difference':diff.max().item(),'mean_absolute_difference':diff.mean().item(),'argmax_equal_fraction':(examples['serial'].argmax(-1)==examples['batch'].argmax(-1)).float().mean().item(),'note':'BF16 batch shape may change rounding. Six teacher-forced checkpoints, not a guarantee about sampled songs.'}
 save();del examples,diff
 # Homogeneous pairs and unequal-length pair; repeat the short pair with reversed order.
 cases=[('short-short',0,[requests[0],requests[0]],['serial','batch']),('long-long',0,[requests[1],requests[1]],['serial','batch']),('mixed',0,requests,['serial','batch']),('short-short',1,[requests[0],requests[0]],['batch','serial'])]
 for name,repeat,pair,modes in cases:
  for mode in modes:
   stage(name+':'+str(repeat)+':'+mode);directory=out/name/str(repeat)/mode
   directory.mkdir(parents=True,exist_ok=False);(directory/'inputs.json').write_text(json.dumps(pair,ensure_ascii=False,indent=2))
   torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.time();t=time.perf_counter();ready=[]
   if mode=='serial':
    songs=[]
    for request in pair:songs.append(pipe(**request,on_stage=stage));ready.append(time.perf_counter()-t)
   else:
    songs=pipe.generate_batch(pair,on_stage=stage)
   torch.cuda.synchronize();seconds=time.perf_counter()-t
   if mode=='batch':ready=[seconds]*2
   record={'case':name,'repeat':repeat,'mode':mode,'started_at':began,'finished_at':time.time(),'generation_seconds':seconds,'ready_seconds':ready,'pair_rpm':120/seconds,'audio_seconds':[len(s.audio)/s.sample_rate for s in songs],'audio_seconds_per_wall_second':sum(len(s.audio)/s.sample_rate for s in songs)/seconds,'truncated':[s.truncated for s in songs],'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'timings':[s.timing for s in songs]}
   for i,s in enumerate(songs):s.save_artifacts(directory/str(i))
   record['audio_sha256']=[hashlib.sha256((directory/str(i)/'audio.flac').read_bytes()).hexdigest() for i in range(2)]
   report['runs'].append(record);save();print(json.dumps({k:record[k] for k in ['case','repeat','mode','generation_seconds','audio_seconds','truncated']}),flush=True);del songs
except BaseException as error:
 report['error']=repr(error);save();traceback.print_exc();raise
finally:
 pipe.close()
report['completed_at']=time.time();save();print('COMPLETE',flush=True)
