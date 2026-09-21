import json
from pathlib import Path
import secrets

root = Path('/data/noiz-yue')
env = root / 'service.env'
if not env.exists():
    env.touch(mode=0o600, exist_ok=False)
    env.write_text('\n'.join([
        'YUE2_API_KEY=' + secrets.token_hex(32),
        'YUE2_HOST=0.0.0.0', 'YUE2_PORT=8000',
        'YUE2_DATA_DIR=/opt/noiz-yue/jobs',
        'YUE2_MODEL=/opt/noiz-yue/models/YuE2-3B', 'YUE2_VAE=/opt/noiz-yue/models/YuE2-Vae',
        'YUE2_LOCAL_FILES_ONLY=true', 'YUE2_DEVICE=cuda', 'YUE2_BACKEND=vllm',
        'YUE2_RESIDENT_MODELS=true', 'YUE2_AR_CONCURRENCY=4',
        'YUE2_NAR_BATCH_SIZE=2', 'YUE2_AR_NAR_OVERLAP=true',
        'YUE2_VLLM_MAX_NUM_SEQS=4', 'YUE2_VLLM_MAX_NUM_BATCHED_TOKENS=8192',
        'YUE2_VLLM_GPU_MEMORY_UTILIZATION=0.30',
        'YUE2_AR_BATCH_WAIT_MS=50',
        'YUE2_MEMORY_BUDGET_GIB=30',
        'YUE2_ODE_STEPS=32', 'YUE2_WARMUP=true', 'YUE2_MAX_PENDING=8',
        'YUE2_TASK_TIMEOUT_SECONDS=1200',
    ]) + '\n')
short = json.loads((root / 'source/examples/benchmark-requests.jsonl').read_text().splitlines()[0])
long = {
    'id': 'long-english-pop', 'seed': 831001, 'cot': 'full',
    'style': 'English, warm piano pop, expressive female voice, acoustic piano, bass and light drums, memorable melody, 88 BPM, complete song with an outro',
    'lyrics': '''[Verse]
We take the road beside the water
We watch the early morning rise
The distant hills are turning golden
The daylight opens up the skies
I keep a letter in my pocket
A little map of where we went
And every word becomes a promise
To treasure all the time we spent

[Chorus]
Let the day come into view
Every road begins with you
Hold a little room for light
We will sing beyond the night
Let the quiet valley know
We have somewhere new to go
Raise a song into the blue
Every road begins with you

[Verse]
The station clock is softly ticking
A paper cup beside the door
The city wakes and starts to listen
To footsteps on the wooden floor
We leave a shadow in the doorway
We carry hope into the rain
And when the clouds have passed above us
We learn to find the light again

[Chorus]
Let the day come into view
Every road begins with you
Hold a little room for light
We will sing beyond the night
Let the quiet valley know
We have somewhere new to go
Raise a song into the blue
Every road begins with you

[Bridge]
When the winter feels too long
We can hold a simple song
When the sky forgets its blue
I will walk the road with you

[Chorus]
Let the day come into view
Every road begins with you
Hold a little room for light
We will sing beyond the night
Let the quiet valley know
We have somewhere new to go
Raise a song into the blue
Every road begins with you

[Outro]
Every road begins with you
Every morning starts anew'''
}
(root / 'requests.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in [short, long]) + '\n')
print('Prepared secret environment file and two benchmark requests; no secrets printed.')
