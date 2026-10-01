"""Record portable source hashes and checkpoint-reproduction source bundle."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'app'


def main():
    paths = []
    for part in ['configs','scripts','openpi_snapshot','assets']:
        for p in (APP/part).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
                paths.append(p)
    hashes = {str(p.relative_to(APP)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    (APP/'manifests/run_sources.json').write_text(json.dumps(dict(sha256=hashes),indent=2)+'\n')
    bundle = APP/'manifests/reproduction_bundle.tar.gz'
    with tarfile.open(bundle,'w:gz') as t:
        for p in paths:
            t.add(p,arcname=str(p.relative_to(APP)),recursive=False)
        t.add(APP/'manifests/run_sources.json',arcname='manifests/run_sources.json')
    (APP/'manifests/reproduction_bundle.json').write_text(json.dumps(dict(
        sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),bytes=bundle.stat().st_size,
        adapt_to_pi=False,includes='training sources and normalization; raw data and tokenizer are separate payloads'),indent=2)+'\n')
    print('FROZEN',len(hashes),'files')


if __name__ == '__main__':
    main()
