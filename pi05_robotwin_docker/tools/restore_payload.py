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


def restore(manifest_path, archives, destination, delete_after_restore=False):
    """Restore every shard in the manifest into `destination`.

    Default behaviour (delete_after_restore=False) is unchanged from the original
    design: every shard's archive must be present, is re-verified by size+hash on
    every call, and a missing archive is a hard failure. This is what the published
    build/A100 handoff documents and tests already rely on.

    delete_after_restore=True is a separate, opt-in mode for a destination disk too
    small to hold the ZIPs and the restored payload at the same time: once a shard
    is verified and extracted, its completion is recorded in a small state file next
    to the payload and its archive is deleted. Later calls trust that record instead
    of re-hashing already-restored data, and treat a shard whose archive has not
    arrived yet as "not yet transferred" rather than an error, so the function can be
    called again and again as archives are copied in one at a time. The trade-off:
    once a shard is recorded complete (and very possibly its ZIP deleted), this mode
    can no longer re-detect tampering with that shard's already-extracted files on a
    later call, because there is nothing left to re-verify it against — only use this
    mode when disk space genuinely requires it.
    """
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'pi05_portable_zip_v1' or not manifest.get('complete'):
        raise ValueError('Complete payload manifest required')
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / '.restore_state.json'
    completed = set()
    if delete_after_restore and state_path.is_file():
        completed = set(json.loads(state_path.read_text()))
    pending = []
    for shard in manifest['shards']:
        if delete_after_restore and shard['file'] in completed:
            continue  # verified, extracted and recorded in a previous pass
        archive = archives / shard['file']
        if Path(shard['file']).name != shard['file']:
            raise ValueError('Invalid archive name')
        if delete_after_restore and not archive.is_file():
            pending.append(shard['file'])  # not transferred yet; retry on a later pass
            continue
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
        if delete_after_restore:
            completed.add(shard['file'])
            state_path.write_text(json.dumps(sorted(completed)))
            archive.unlink()
    if pending:
        preview = ', '.join(sorted(pending)[:5]) + (', ...' if len(pending) > 5 else '')
        print(f'PARTIAL: {len(pending)} shard(s) not yet transferred: {preview}', flush=True)
        return
    (destination / 'PAYLOAD_VERIFIED.json').write_text(json.dumps(dict(
        manifest_sha256=digest(manifest_path), episodes=manifest['episodes'])) + '\n')
    print('PAYLOAD_VERIFIED', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archives', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('payload'))
    p.add_argument('--delete-after-restore', action='store_true',
                    help='Delete each archive immediately after its contents are verified and '
                         'extracted, and trust a small state file for shards already completed '
                         'in a previous pass instead of re-hashing them. Safe to call repeatedly '
                         'as more archives arrive one at a time. Use only when the destination '
                         'disk cannot hold the ZIPs and the restored payload at the same time; '
                         'once a shard is recorded complete this mode can no longer re-detect '
                         'later tampering with that shard, because its archive is gone.')
    args = p.parse_args()
    restore(args.archives / 'payload_manifest.json', args.archives, args.output,
            delete_after_restore=args.delete_after_restore)
