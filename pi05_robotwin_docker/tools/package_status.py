"""Local packaging progress; no chat notifications or automatic upload."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=root/'archives'
def read(p):
    return json.loads(p.read_text()) if p.exists() else None
pid_file=root/'build/package.pid'
pid=pid_file.read_text().strip() if pid_file.exists() else None
process=Path('/proc')/str(pid)/'cmdline'
alive=process.exists() and b'package_payload.py' in process.read_bytes()
print(json.dumps(dict(plan=read(out/'pack_plan.json'),status=read(out/'pack_status.json'),
    pid=pid,process_alive=alive,manifest_complete=(out/'payload_manifest.json').exists(),
    current_archive_bytes=sum(p.stat().st_size for p in out.glob('payload-*.zip*') if p.is_file())),indent=2))
