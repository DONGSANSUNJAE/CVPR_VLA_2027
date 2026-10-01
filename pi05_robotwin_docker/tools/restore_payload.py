"""Verify and restore independent ZIP shards, safely and resumably (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

if sys.version_info < (3, 9):
    raise SystemExit(
        f'Python 3.9+ required (PurePath.is_relative_to); found {sys.version.split()[0]}. '
        'Install a newer python3 on this host before restoring the payload.'
    )


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def restore(manifest_path, archives, destination):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'pi05_portable_zip_v1' or not manifest.get('complete'):
        raise ValueError('Complete payload manifest required')
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for shard in manifest['shards']:
        archive = archives / shard['file']
        if Path(shard['file']).name != shard['file']:
            raise ValueError('Invalid archive name')
        if archive.stat().st_size != shard['bytes'] or digest(archive) != shard['sha256']:
            raise ValueError(f'Archive checksum failed: {archive}')
        expected = {f['path']: f for f in shard['files']}
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist()) != len(expected) or set(z.namelist()) != set(expected):
                raise ValueError('Archive members differ from manifest')
            for name, entry in expected.items():
                relative = PurePosixPath(name)
                if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in {'data','base','tokenizer'}:
                    raise ValueError(f'Unsafe payload path: {name}')
                target = destination / name
                if not target.resolve().is_relative_to(destination):
                    raise ValueError('Destination symlink escapes payload')
                if target.exists():
                    if target.stat().st_size == entry['bytes'] and digest(target) == entry['sha256']:
                        continue
                    raise FileExistsError(f'Existing content differs; preserve and inspect it: {target}')
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_name(target.name + '.restore-partial')
                if temp.is_symlink():
                    raise ValueError('Refusing a symlink at temporary output')
                h = hashlib.sha256()
                size = 0
                with z.open(name) as source, temp.open('wb') as out:
                    for b in iter(lambda: source.read(8 << 20), b''):
                        h.update(b)
                        size += len(b)
                        out.write(b)
                if size != entry['bytes'] or h.hexdigest() != entry['sha256']:
                    raise ValueError(f'Extracted checksum failed: {name}')
                temp.replace(target)
        print('RESTORED', archive.name, flush=True)
    (destination / 'PAYLOAD_VERIFIED.json').write_text(json.dumps(dict(
        manifest_sha256=digest(manifest_path), episodes=manifest['episodes'])) + '\n')
    print('PAYLOAD_VERIFIED', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archives', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('payload'))
    args = p.parse_args()
    restore(args.archives / 'payload_manifest.json', args.archives, args.output)
