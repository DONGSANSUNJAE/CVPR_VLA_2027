"""CPU-only ETA observer for an already running, frozen training process.

Reads the original logs and writes separate *_eta files. Never modifies the
trainer, its source hashes, checkpoints, or its own log files. Rates describe
optimizer updates; a final checkpoint flush is not included in the forecast.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import time

UTC = timezone.utc
KST = timezone(timedelta(hours=9))
DEFAULT_LOGS = '/HW/pi05_robotwin_false_60k/outputs/logs'


def read_records(path):
    """Bound reads and tolerate a last JSON line still being written."""
    with path.open('rb') as stream:
        size = stream.seek(0, 2)
        stream.seek(max(0, size - 262144))
        if size > 262144:
            stream.readline()
        data = stream.read()
    records = []
    for line in data.splitlines(keepends=True):
        if not line.endswith(b'\n'):
            continue
        try:
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
        except (ValueError, UnicodeDecodeError):
            continue
    return records


def recent_rate(records, step, window=100):
    """Weight each interval by its update count; exclude startup compilation."""
    matches = [i for i, r in enumerate(records) if r.get('completed_updates') == step]
    if not matches:
        return None, 0
    intervals = []
    previous = None
    for record in records[:matches[-1] + 1]:
        current = record.get('completed_updates')
        rate = record.get('seconds_per_update')
        if not isinstance(current, int) or isinstance(current, bool):
            continue
        delta = current - previous if previous is not None else 0
        if delta <= 0 or delta > 10:
            intervals.clear()  # Restart, rollback, or missing interval boundary.
        elif current > 3 and previous >= 3 and isinstance(rate, (int, float)) and math.isfinite(rate) and rate > 0:
            intervals.append((delta, rate))
        previous = current
    seconds = 0.0
    count = 0
    for delta, rate in reversed(intervals):
        used = min(delta, window - count)
        seconds += used * rate
        count += used
        if count == window:
            break
    return (seconds / count if count >= 20 else None), count


def duration(seconds):
    minutes = math.ceil(max(0, seconds) / 60)
    days, minutes = divmod(minutes, 1440)
    hours, minutes = divmod(minutes, 60)
    return f'{days}d {hours:02d}h {minutes:02d}m'


def make_snapshot(status, records, running, now=None):
    now = now or datetime.now(UTC)
    result = dict(status)
    if 'completed_updates' not in result and isinstance(status.get('last_status'), dict):
        for name in ('completed_updates', 'target_updates'):
            if name in status['last_status']:
                result[name] = status['last_status'][name]
    result.update(
        eta_status='waiting_for_training',
        training_process_running=running,
        eta_seconds_per_update=None,
        eta_window_updates=0,
        estimated_remaining_seconds=None,
        estimated_remaining=None,
        estimated_finish_utc=None,
        estimated_finish_kst=None,
        eta_as_of_utc=status.get('time_utc'),
        eta_observed_utc=now.isoformat(),
        eta_scope='optimizer_updates; final checkpoint flush and future interruptions excluded',
    )
    phase = status.get('phase')
    if phase == 'complete':
        result.update(eta_status='complete', estimated_remaining_seconds=0, estimated_remaining='0d 00h 00m')
        return result
    if phase in {'failed', 'interrupted_checkpoint_saved'}:
        result['eta_status'] = 'stopped'
        return result
    if not running:
        result['eta_status'] = 'not_running'
        return result
    if phase != 'training':
        return result
    step = status.get('completed_updates')
    target = status.get('target_updates')
    if not isinstance(step, int) or not isinstance(target, int) or target < step or step < 0:
        result['eta_status'] = 'invalid_progress'
        return result
    if step == target:
        result['eta_status'] = 'final_checkpoint_pending'
        return result
    rate, count = recent_rate(records, step)
    result.update(eta_seconds_per_update=rate, eta_window_updates=count)
    if rate is None:
        result['eta_status'] = 'warming_up'
        return result
    try:
        stamp = datetime.fromisoformat(status['time_utc'])
        if stamp.tzinfo is None:
            raise ValueError('Timestamp must contain timezone')
        age = (now - stamp).total_seconds()
        if age < -30:
            raise ValueError('Future timestamp')
    except (KeyError, ValueError, TypeError):
        result['eta_status'] = 'invalid_timestamp'
        return result
    result['source_age_seconds'] = round(max(0, age), 1)
    if age > max(300, 50 * rate):
        result['eta_status'] = 'stale_progress'
        return result
    remaining = (target - step) * rate
    finish = stamp + timedelta(seconds=remaining)
    result.update(
        eta_status='estimated',
        estimated_remaining_seconds=round(remaining, 1),
        estimated_remaining=duration(remaining),
        estimated_finish_utc=finish.isoformat(timespec='seconds'),
        estimated_finish_kst=finish.astimezone(KST).isoformat(timespec='seconds'),
    )
    return result


def trainer_identities():
    found = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
            if any(a.endswith(b'train_robotwin_impl.py') for a in args) and b'--preflight-only' not in args:
                if (path.parent / 'cwd').resolve() == Path('/opt/pi05').resolve():
                    stat = (path.parent / 'stat').read_text().rsplit(')', 1)[1].split()
                    found.append((int(path.parent.name), stat[19]))
        except (OSError, IndexError):
            pass
    return sorted(found)


def atomic_json(path, record):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_status(logs):
    """An object mount may return an older status snapshot than launch.log."""
    candidates = []
    status_path = logs / 'train_status.json'
    if status_path.exists():
        try:
            candidates.append((json.loads(status_path.read_text()), status_path.name))
        except ValueError:
            pass
    launch = logs / 'launch.log'
    if launch.exists():
        candidates.extend((r, launch.name) for r in read_records(launch) if 'phase' in r)
    valid = []
    for record, source in candidates:
        try:
            stamp = datetime.fromisoformat(record['time_utc'])
            if stamp.tzinfo is not None:
                valid.append((stamp, record, source))
        except (KeyError, ValueError, TypeError):
            continue
    if not valid:
        raise ValueError('No readable timestamped training status')
    _, record, source = max(valid, key=lambda entry: entry[0])
    return dict(record, source_status_file=source)


def observe(logs, watch):
    last_key = None
    last_metric_key = None
    previous_trainers = None
    session_floor = None
    first_interval_pending = True
    while True:
        now = datetime.now(UTC)
        trainers = trainer_identities()
        try:
            status = read_status(logs)
            records = read_records(logs / 'metrics.jsonl') if (logs / 'metrics.jsonl').exists() else []
            step = status.get('completed_updates')
            # Once observing, discard pre-restart timing history and the first
            # logging interval of the new process (which includes compilation).
            if previous_trainers is not None and trainers != previous_trainers:
                first_interval_pending = True
            if status.get('phase') not in {'training', 'complete'}:
                first_interval_pending = True
            if status.get('phase') == 'training' and first_interval_pending and isinstance(step, int):
                session_floor = step
                first_interval_pending = False
            filtered = records if session_floor is None else [r for r in records if r.get('completed_updates', -1) > session_floor]
            snapshot = make_snapshot(status, filtered, len(trainers) == 1, now)
        except (OSError, ValueError, TypeError) as error:
            records = []
            snapshot = make_snapshot({'phase': 'logs_unavailable', 'read_error': str(error)}, [], len(trainers) == 1, now)
        previous_trainers = trainers
        snapshot['training_pids'] = [pid for pid, _ in trainers]
        key = (snapshot.get('time_utc'), snapshot.get('completed_updates'), snapshot.get('phase'), snapshot['eta_status'], tuple(trainers), snapshot.get('read_error'))
        if key != last_key:
            atomic_json(logs / 'train_status_eta.json', snapshot)
            metric_key = (snapshot.get('time_utc'), snapshot.get('completed_updates'), snapshot['eta_status'], tuple(trainers))
            matching = [r for r in records if r.get('completed_updates') == snapshot.get('completed_updates')]
            if matching and metric_key != last_metric_key:
                enriched = dict(matching[-1])
                enriched.update({k: v for k, v in snapshot.items() if k.startswith(('eta_', 'estimated_', 'training_', 'source_age'))})
                enriched['record_type'] = 'eta_observation'
                enriched['source_phase'] = snapshot.get('phase')
                enriched['source_status_time_utc'] = snapshot.get('time_utc')
                with (logs / 'metrics_eta.jsonl').open('a') as stream:
                    stream.write(json.dumps(enriched, allow_nan=False) + '\n')
                last_metric_key = metric_key
            print(json.dumps(snapshot, allow_nan=False), flush=True)
            last_key = key
        if not watch:
            return
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', type=Path, default=Path(DEFAULT_LOGS))
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if not args.logs.is_dir():
        raise SystemExit(f'Log directory does not exist: {args.logs}')
    import hashlib
    token = hashlib.sha256(str(args.logs.resolve()).encode()).hexdigest()[:16]
    with open(f'/tmp/pi05_eta_{token}.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('ETA observer already running; read train_status_eta.json')
        observe(args.logs, args.watch)


if __name__ == '__main__':
    main()
