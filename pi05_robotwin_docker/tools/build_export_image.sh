#!/usr/bin/env bash
# Run on a Docker-capable Linux x86_64 build host. No GPU or payload required.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

for executable in docker git python3 sha256sum; do
  command -v "$executable" >/dev/null || { echo "Missing executable: $executable" >&2; exit 1; }
done
docker info >/dev/null
docker buildx version
if [[ "$(docker info --format '{{.OSType}}/{{.Architecture}}')" != 'linux/x86_64' && \
      "$(docker info --format '{{.OSType}}/{{.Architecture}}')" != 'linux/amd64' ]]; then
  echo 'This handoff requires a native Linux x86_64 Docker builder.' >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo 'Use a clean Git clone so the image can be matched to an exact commit.' >&2
  exit 1
fi
PI05_SOURCE_COMMIT="$(git rev-parse HEAD)"
PI05_IMAGE='pi05-robotwin-false:60k'
PI05_EXPORT_DIR="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${1:-build/image_export}")"
mkdir -p -- "$PI05_EXPORT_DIR"
if [[ -n "$(ls -A -- "$PI05_EXPORT_DIR")" ]]; then
  echo 'Export directory must be empty; keep any previous build and choose a new directory.' >&2
  exit 1
fi
printf '%s\n' "$PI05_SOURCE_COMMIT" > "$PI05_EXPORT_DIR/source_commit.txt"
docker version > "$PI05_EXPORT_DIR/docker_version.txt"
docker buildx build --platform linux/amd64 --load --progress plain \
  --label "org.opencontainers.image.revision=$PI05_SOURCE_COMMIT" \
  --label 'org.opencontainers.image.source=https://github.com/DONGSANSUNJAE/CVPR_VLA_2027' \
  --tag "$PI05_IMAGE" . 2>&1 | tee "$PI05_EXPORT_DIR/build.log"

# Override the training entrypoint and require no mounts, network, GPU or weights.
docker run --rm --network none --entrypoint /opt/venv/bin/python \
  -e JAX_PLATFORMS=cpu -e CUDA_VISIBLE_DEVICES= -i "$PI05_IMAGE" \
  < tools/check_image_cpu.py > "$PI05_EXPORT_DIR/cpu_smoke.json" \
  2> "$PI05_EXPORT_DIR/cpu_smoke.stderr.log"
cat "$PI05_EXPORT_DIR/cpu_smoke.json"
docker image inspect "$PI05_IMAGE" > "$PI05_EXPORT_DIR/image_inspect.json"
docker image save --output "$PI05_EXPORT_DIR/pi05-robotwin-linux-amd64.tar.partial" "$PI05_IMAGE"
mv -- "$PI05_EXPORT_DIR/pi05-robotwin-linux-amd64.tar.partial" "$PI05_EXPORT_DIR/pi05-robotwin-linux-amd64.tar"

python3 - "$PI05_EXPORT_DIR" "$PI05_SOURCE_COMMIT" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
p = Path(sys.argv[1])
info = json.loads((p / 'image_inspect.json').read_text())[0]
assert (info['Os'], info['Architecture']) == ('linux', 'amd64')
assert info['Config']['Labels']['org.opencontainers.image.revision'] == sys.argv[2]
assert json.loads((p / 'cpu_smoke.json').read_text())['status'] == 'passed'
(p / 'image_manifest.json').write_text(json.dumps({
    'image_tag': 'pi05-robotwin-false:60k', 'image_id': info['Id'],
    'platform': 'linux/amd64', 'source_commit': sys.argv[2],
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'image_archive': 'pi05-robotwin-linux-amd64.tar',
    'archive_bytes': (p / 'pi05-robotwin-linux-amd64.tar').stat().st_size,
    'cpu_smoke': 'passed', 'gpu_training_tested': False,
    'raw_data_and_base_weights_included': False,
}, indent=2) + '\n')
PY
(
  cd -- "$PI05_EXPORT_DIR"
  sha256sum pi05-robotwin-linux-amd64.tar image_manifest.json image_inspect.json \
    source_commit.txt docker_version.txt build.log cpu_smoke.json cpu_smoke.stderr.log > SHA256SUMS
)
printf '\nIMAGE_EXPORT_COMPLETE: %s\n' "$PI05_EXPORT_DIR"
printf 'Send this whole directory to the A100 host. GPU training remains untested.\n'
