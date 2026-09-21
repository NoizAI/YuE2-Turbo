# Shared worker load reporting

The worker writes a local JSON snapshot every second; `deploy/heartbeat.py` reads
it and reports to both controllers every five seconds. No API keys are copied
into heartbeat configs. Existing register credentials remain unchanged.

Snapshots count every live candidate/transcription, including work from either
environment or direct callers. Controllers publish them to the shared physical
GPU admission ledger. Stale/missing snapshots fail closed for new requests.

Deploy by physical GPU: stop its heartbeats in both environments, mark both
model workers draining and publish worker events, wait for HTTP slots and
actual model queues to empty, update and restart that card, verify readiness
and fresh snapshot, then resume its heartbeats. Repeat for the other card.
Existing job DBs and audio artifacts must be preserved.

YuE snapshot paths are `/data/noiz-yue/jobs/load.json` (GPU5) and
`/data/noiz-yue/jobs-gpu7/load.json` (GPU7). Their existing ubuntu-owned parent
directories need the setgid bit so root container writers create group-readable
0640 snapshots for the ubuntu heartbeat service. Use `chmod g+s` on those two
directories; do not copy API credentials or make the snapshot world-readable.

SheetSage snapshots are `/opt/noiz-sheetsage2/run/gpu5.json` and `gpu7.json`;
create that directory owned by the worker service user (currently ubuntu).
SheetSage drain markers are `run/draining-gpu5` and `run/draining-gpu7`;
YuE markers are `draining` in its corresponding jobs directory. Remove markers
to resume admission. Status, download and cancel endpoints remain accessible.
