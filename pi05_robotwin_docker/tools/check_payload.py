"""Fast destination readiness checks; restore_payload performs full hashes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
payload = ROOT / 'payload'
marker = payload / 'PAYLOAD_VERIFIED.json'
if not marker.is_file():
    raise SystemExit('Run python3 tools/restore_payload.py --archives /path/to/archives first')
index = json.loads((payload / 'data/index.json').read_text())
norm = json.loads((ROOT / 'app/assets/robotwin_clean_randomized_27500/norm_stats_provenance.json').read_text())
verified = json.loads(marker.read_text())
if not index.get('production') or len(index['episodes']) != 27500 or verified['episodes'] != 27500:
    raise SystemExit('Expected all 27,500 clean+random episodes')
if norm.get('adapt_to_pi') is not False:
    raise SystemExit('Refusing normalization from adapt_to_pi=True')
missing = [e['raw_path'] for e in index['episodes'] if not (payload / 'data' / e['raw_path']).is_file()]
if missing:
    raise SystemExit(f'Missing {len(missing)} HDF5 files, first: {missing[0]}')
(ROOT / 'outputs').mkdir(exist_ok=True)
print('PAYLOAD_READY: 27,500 episodes; adapt_to_pi=False; full checksum verification was recorded during restore')
