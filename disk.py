import ctypes
import json
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes

import torch

from checkpoint import CKPT

SECTOR = 4096
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.CreateFileW.restype = wintypes.HANDLE
_k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
_k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]


class _Overlapped(ctypes.Structure):
    _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t), ("Offset", wintypes.DWORD),
                ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]


class DiskExperts:
    """Reads expert weights straight from NVMe, bypassing the Windows file cache (FILE_FLAG_NO_BUFFERING)."""

    def __init__(self, slots, threads=8):
        index = json.load(open(CKPT / "model.safetensors.index.json"))["weight_map"]
        self.where, self.scales, spans = {}, {}, {}
        for shard in sorted(set(index.values())):
            path = str(CKPT / shard)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
                header.pop("__metadata__", None)
                for key, h in header.items():
                    if ".mlp.experts." not in key:
                        continue
                    root, proj, kind = key.rsplit(".", 2)
                    start, end = (8 + n + o for o in h["data_offsets"])
                    if kind == "weight":
                        spans.setdefault(root, {}).setdefault(path, {})[proj] = (start, end)
                    else:
                        f.seek(start)
                        self.scales[(root, proj)] = torch.frombuffer(bytearray(f.read(end - start)), dtype=torch.float32).reshape(h["shape"])
        size = 0
        for root, files in spans.items():
            pieces = []
            for path, p in files.items():
                start, end = min(a for a, _ in p.values()), max(b for _, b in p.values())
                assert end - start == sum(b - a for a, b in p.values())
                pieces.append((path, start, end, {k: (a - start, b - start) for k, (a, b) in p.items()}))
            size = max(size, sum(end - start + 2 * SECTOR for _, start, end, _ in pieces))
            layer, e = root.split(".layers.")[1].split(".mlp.experts.")
            self.where[(int(layer), int(e))] = (root, pieces)
        self.stride = (size + SECTOR - 1) // SECTOR * SECTOR
        self.slots = slots
        self.buffer = torch.empty(slots * self.stride, dtype=torch.uint8, pin_memory=True)
        assert self.buffer.data_ptr() % SECTOR == 0
        self.pool = ThreadPoolExecutor(threads)
        self.local = threading.local()

    def _handle(self, path):
        handles = getattr(self.local, "handles", None)
        if handles is None:
            handles = self.local.handles = {}
        if path not in handles:
            h = _k32.CreateFileW(path, 0x80000000, 1, None, 3, 0x20000000, None)
            if h in (None, wintypes.HANDLE(-1).value):
                raise ctypes.WinError(ctypes.get_last_error())
            handles[path] = h
        return handles[path]

    def _read(self, slot, key):
        offsets, cursor = {}, slot * self.stride
        for path, start, end, projs in self.where[key][1]:
            aligned = start // SECTOR * SECTOR
            length = (end - aligned + SECTOR - 1) // SECTOR * SECTOR
            o = _Overlapped(Offset=aligned & 0xFFFFFFFF, OffsetHigh=aligned >> 32)
            read = wintypes.DWORD()
            if not _k32.ReadFile(self._handle(path), self.buffer.data_ptr() + cursor, length, ctypes.byref(read), ctypes.byref(o)):
                raise ctypes.WinError(ctypes.get_last_error())
            for proj, (a, b) in projs.items():
                offsets[proj] = (cursor + start - aligned + a, cursor + start - aligned + b)
            cursor += length
        return offsets

    def fetch(self, keys):
        """keys is a list of (layer, expert), read in order into buffer slots. Returns per-proj buffer offsets for each."""
        return list(self.pool.map(lambda ik: self._read(ik[0], ik[1]), enumerate(keys)))

    def view(self, offsets, proj):
        a, b = offsets[proj]
        return self.buffer[a:b]
