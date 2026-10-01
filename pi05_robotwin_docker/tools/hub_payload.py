"""Optional private Hugging Face transfer; no token is stored in this project.

Run only after explicitly choosing the destination. Code belongs in GitHub;
ZIPs belong in a private dataset repo or a direct server-to-server transfer.
Requires huggingface_hub (included in the container's dependency lock).
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['upload','download'])
    p.add_argument('--repo-id', required=True)
    p.add_argument('--directory', type=Path, default=Path('archives'))
    p.add_argument('--revision', help='Pinned commit SHA required for downloads')
    a = p.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    if a.action == 'upload':
        from package_payload import digest
        manifest_path = a.directory / 'payload_manifest.json'
        m = json.loads(manifest_path.read_text())
        if not m.get('complete'):
            raise ValueError('Payload is not complete')
        for s in m['shards']:
            if digest(a.directory / s['file']) != s['sha256']:
                raise ValueError('Archive checksum failed before upload')
        api.create_repo(a.repo_id, repo_type='dataset', private=True, exist_ok=True)
        if not api.repo_info(a.repo_id, repo_type='dataset').private:
            raise ValueError('Refusing to upload this migration payload to a public repository')
        api.upload_large_folder(repo_id=a.repo_id, repo_type='dataset', folder_path=a.directory,
            allow_patterns=['payload-*.zip', 'payload_manifest.json'], num_workers=4)
        print('PIN_THIS_REVISION', api.repo_info(a.repo_id, repo_type='dataset').sha)
    else:
        if not a.revision or len(a.revision) != 40 or any(c not in '0123456789abcdef' for c in a.revision):
            raise ValueError('Pass the 40-character commit SHA from upload')
        snapshot_download(a.repo_id, repo_type='dataset', revision=a.revision,
            local_dir=str(a.directory), allow_patterns=['payload-*.zip','payload_manifest.json'], max_workers=4)


if __name__ == '__main__':
    main()
