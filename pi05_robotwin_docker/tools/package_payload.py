"""Make independent ZIP64 shards of the exact indexed data and verified base.

Does not modify source data or upload anything. Re-running reuses completed,
checksum-verified shards. Ordinary Python can restore these on the destination.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import time
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, value):
    p = Path(path)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(p)


def build_shard(out, number, entries):
    archive = out / f'payload-{number:03d}.zip'
    meta = archive.with_suffix('.json')
    specification = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    if archive.exists() and meta.exists():
        old = json.loads(meta.read_text())
        if old['specification'] == specification and digest(archive) == old['sha256']:
            return old
        raise ValueError(f'Existing shard does not match the source: {archive}')
    temporary = archive.with_suffix('.zip.partial')
    files = []
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as z:
        for e in entries:
            source = Path(e['source'])
            before = source.stat()
            h = hashlib.sha256()
            with source.open('rb') as f, z.open(e['path'], 'w', force_zip64=True) as target:
                for b in iter(lambda: f.read(4 << 20), b''):
                    h.update(b)
                    target.write(b)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f'Source changed during packing: {source}')
            if e.get('expected_sha256') and h.hexdigest() != e['expected_sha256']:
                raise ValueError(f'Base checkpoint checksum mismatch: {source}')
            files.append(dict(path=e['path'], bytes=after.st_size, sha256=h.hexdigest()))
    temporary.replace(archive)
    result = dict(file=archive.name, bytes=archive.stat().st_size, sha256=digest(archive),
                  specification=specification, files=files)
    atomic(meta, result)
    print(json.dumps(dict(event='shard_complete', file=archive.name, bytes=result['bytes'])), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--shard-gib', type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.pack.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        src = args.source_run.resolve()
        data = src / 'data'
        index = json.loads((data / 'index.json').read_text())
        assert index['production'] and len(index['episodes']) == 27500
        paths = {'index.json', index['numeric_path']}
        for e in index['episodes']:
            paths.update([e['raw_path'], e['instruction_path']])
        entries = [dict(source=str((data / r).resolve()), path='data/' + r) for r in sorted(paths)]
        base = json.loads((src / 'manifests/pi05_base_verified.json').read_text())
        entries += [dict(source=str((src / 'checkpoints/pi05_base' / e['path']).resolve()),
                         path='base/' + e['path'], expected_sha256=e['sha256']) for e in base['verified_files']]
        tokenizer = Path(__file__).resolve().parents[1] / 'app/tokenizer/paligemma_tokenizer.model'
        entries += [dict(source=str(tokenizer), path='tokenizer/paligemma_tokenizer.model')]
        groups, group, size = [], [], 0
        for e in entries:
            n = Path(e['source']).stat().st_size
            if group and size + n > args.shard_gib * 2**30:
                groups.append(group)
                group, size = [], 0
            group.append(e)
            size += n
        if group:
            groups.append(group)
        atomic(args.output / 'pack_plan.json', dict(shards=len(groups), files=len(entries),
            total_bytes=sum(Path(e['source']).stat().st_size for e in entries),
            source_index_sha256=digest(data / 'index.json')))
        results = []
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(build_shard, args.output, i, g) for i, g in enumerate(groups)]
            for f in as_completed(futures):
                results.append(f.result())
                atomic(args.output / 'pack_status.json', dict(phase='packing',
                    complete_shards=len(results), total_shards=len(groups),
                    elapsed_seconds=time.monotonic() - started))
        manifest = dict(schema='pi05_portable_zip_v1', complete=True,
            created_utc=datetime.now(timezone.utc).isoformat(),
            source_index_sha256=digest(data / 'index.json'),
            episodes=27500, shards=sorted(results, key=lambda e:e['file']))
        atomic(args.output / 'payload_manifest.json', manifest)
        atomic(args.output / 'pack_status.json', dict(phase='complete', complete_shards=len(results),
            total_shards=len(groups), compressed_bytes=sum(e['bytes'] for e in results),
            elapsed_seconds=time.monotonic()-started))
        print('PAYLOAD_PACKAGING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
