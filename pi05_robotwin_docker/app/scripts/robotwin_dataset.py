"""Raw RoboTwin episodes with the public pi05 converter's temporal contract.

No LeRobot video transcoding is needed. Images retain the public converter's
OpenCV decoding without a channel swap and its 640x480 resize. The simulator
itself passes native RGB arrays directly to cv2.imencode, so this no-swap decode
restores their original numerical RGB ordering. Avoiding the second JPEG and
video encoding changes compression artifacts, not frame selection or channel
order. Adapter/delta/normalization transforms belong outside this dataset.
"""
from __future__ import annotations

import bisect
from collections import OrderedDict
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np


CAMERAS = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}
SCHEMA = "robotwin_raw_hdf5_v1"


def read_index(path: str | Path) -> dict:
    index = json.loads(Path(path).read_text())
    if index.get("schema") != SCHEMA or not index.get("complete"):
        raise ValueError("A complete robotwin_raw_hdf5_v1 index is required")
    if index.get("action_horizon") != 50:
        raise ValueError("This frozen recipe requires action_horizon=50")
    episodes = index["episodes"]
    raw_offset = train_offset = 0
    seen = set()
    counts = {}
    for ep in episodes:
        key = (ep["task"], ep["setting"], ep["episode_index"])
        if key in seen:
            raise ValueError(f"Duplicate episode: {key}")
        seen.add(key)
        if ep["state_offset"] != raw_offset or ep["training_offset"] != train_offset:
            raise ValueError("Non-contiguous numeric or training offsets")
        if ep["raw_num_frames"] < 2 or ep["training_length"] != ep["raw_num_frames"] - 1:
            raise ValueError("Episode length must preserve the one-step action shift")
        if not isinstance(ep["prompt"], str) or not ep["prompt"].strip():
            raise ValueError("Missing frozen seen instruction")
        raw_offset += ep["raw_num_frames"]
        train_offset += ep["training_length"]
        counts.setdefault(ep["task"], {"clean": 0, "randomized": 0})[ep["setting"]] += 1
    if raw_offset != index["num_raw_frames"] or train_offset != index["total_train_frames"]:
        raise ValueError("Index totals do not match episodes")
    if index.get("production"):
        expected = index["expected_tasks"]
        if len(expected) != 50 or set(counts) != set(expected) or len(episodes) != 27500:
            raise ValueError("Production requires all 50 tasks and 27,500 episodes")
        for task, count in counts.items():
            if count != {"clean": 50, "randomized": 500}:
                raise ValueError(f"Incomplete clean/randomized mix for {task}: {count}")
            for setting, n in [("clean", 50), ("randomized", 500)]:
                if {k[2] for k in seen if k[:2] == (task, setting)} != set(range(n)):
                    raise ValueError(f"Invalid episode IDs for {task}/{setting}")
    return index


class RobotwinRawDataset:
    """Frame-uniform random access; HDF5 handles and numeric mmap are worker-local."""

    def __init__(self, index_path: str | Path, action_horizon: int = 50, hdf5_cache_size: int = 16):
        if action_horizon != 50:
            raise ValueError("Frozen action horizon is 50")
        if hdf5_cache_size < 1:
            raise ValueError("hdf5_cache_size must be positive")
        self.index_path = Path(index_path).resolve()
        self.root = self.index_path.parent
        self.index = read_index(self.index_path)
        self.episodes = self.index["episodes"]
        self._ends = [e["training_offset"] + e["training_length"] for e in self.episodes]
        self.action_horizon = action_horizon
        self.hdf5_cache_size = hdf5_cache_size
        self._pid = None
        self._qpos = None
        self._files = OrderedDict()

    def __len__(self):
        return self.index["total_train_frames"]

    def _ensure_worker(self):
        if self._pid == os.getpid():
            return
        self.close()
        self._pid = os.getpid()
        self._qpos = np.load(self.root / self.index["numeric_path"], mmap_mode="r", allow_pickle=False)
        if self._qpos.shape != (self.index["num_raw_frames"], 14) or self._qpos.dtype != np.float32:
            raise ValueError("Numeric memmap shape/dtype differs from manifest")
        cv2.setNumThreads(1)

    def locate(self, index: int):
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep = self.episodes[bisect.bisect_right(self._ends, index)]
        return ep, index - ep["training_offset"]

    def numeric_sample(self, index: int):
        self._ensure_worker()
        ep, t = self.locate(index)
        offsets = ep["state_offset"] + np.minimum(t + 1 + np.arange(self.action_horizon), ep["raw_num_frames"] - 1)
        return {
            "state": np.array(self._qpos[ep["state_offset"] + t], copy=True),
            "actions": np.array(self._qpos[offsets], copy=True),
            "prompt": ep["prompt"],
        }

    def __getitem__(self, index: int):
        result = self.numeric_sample(index)
        ep, t = self.locate(index)
        name = ep["raw_path"]
        if name in self._files:
            h5 = self._files.pop(name)
        else:
            h5 = h5py.File(self.root / name, "r")
        self._files[name] = h5
        while len(self._files) > self.hdf5_cache_size:
            _, old = self._files.popitem(last=False)
            old.close()
        images = {}
        for output_name, camera in CAMERAS.items():
            encoded = h5[f"observation/{camera}/rgb"][t]
            image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Invalid image: {name}/{camera}/{t}")
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
            images[output_name] = np.ascontiguousarray(image.transpose(2, 0, 1))
        result["images"] = images
        return result

    def close(self):
        for h5 in getattr(self, "_files", {}).values():
            h5.close()
        self._files = OrderedDict()
        self._qpos = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_files=OrderedDict(), _qpos=None, _pid=None)
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
