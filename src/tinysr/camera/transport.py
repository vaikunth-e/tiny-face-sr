"""USBDBG checksummed transfer collector for the camera-owned C16 experiment"""

import base64
import hashlib
import re

PROTOCOL = "TINYSR"


class Collector:
    """incrementally verify numbered transfers while retaining only a small log"""

    def __init__(self, *, max_jpeg_bytes: int = 3_000_000):
        self.pending = ""
        self.phase_log = []
        self.tail = []
        self.errors = []
        self.completed = False
        self.transfers = {}
        self.max_jpeg_bytes = max_jpeg_bytes
        self.lines = 0

    def feed(self, text: str) -> None:
        """accept arbitrary USB fragmentation, including split UTF-8 text lines"""
        self.pending += text.replace("\r", "")
        if len(self.pending) > 32768:
            raise ValueError("camera sent an excessively long line")
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self.consume(line)

    def consume(self, line: str) -> None:
        """check transfer ordering and bounds before saving decoded bytes"""
        self.lines += 1
        if line.startswith(PROTOCOL + "|PHASE|"):
            self.phase_log.append(line.split("|", 2)[2])
            return
        if line.startswith(PROTOCOL + "|ERROR|"):
            self.errors.append(line.split("|", 2)[2])
            return
        if line.startswith(PROTOCOL + "|FATAL|"):
            raise RuntimeError("camera transfer failure: " + line.split("|", 2)[2])
        if line == PROTOCOL + "|DONE":
            if "REPORT" not in self.transfers or any(
                not t["finished"] for t in self.transfers.values()
            ):
                raise ValueError("DONE before complete report or image transfer")
            self.completed = True
            return
        if not line.startswith(PROTOCOL + "|"):
            self.tail.append(line)
            self.tail = self.tail[-120:]
            return
        fields = line.split("|")
        if len(fields) < 4:
            raise ValueError("malformed transfer record")
        command, kind = fields[1], fields[2]
        if kind not in ("BASE", "NEURAL", "REPORT"):
            raise ValueError("unexpected transfer kind")
        if command == "START":
            if len(fields) != 6:
                raise ValueError("malformed transfer header")
            size, count = int(fields[3]), int(fields[4])
            digest = fields[5]
            limit = self.max_jpeg_bytes if kind in ("BASE", "NEURAL") else 32_768
            if kind in self.transfers or not 0 < size <= limit:
                raise ValueError("duplicate or oversized transfer")
            if not (0 < count <= size and re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise ValueError("invalid transfer header")
            self.transfers[kind] = {
                "length": size,
                "count": count,
                "sha256": digest,
                "next_index": 0,
                "data": bytearray(),
                "finished": False,
            }
        elif command == "DATA":
            if len(fields) != 5:
                raise ValueError("malformed data record")
            item = self.transfers.get(kind)
            if item is None or item["finished"]:
                raise ValueError("data outside transfer")
            idx = int(fields[3])
            encoded = fields[4]
            if idx != item["next_index"]:
                raise ValueError("missing or out-of-order camera chunk")
            payload = base64.b64decode(encoded, validate=True)
            if not payload or len(item["data"]) + len(payload) > item["length"]:
                raise ValueError("invalid chunk length")
            item["data"].extend(payload)
            item["next_index"] += 1
        elif command == "END":
            if len(fields) != 4:
                raise ValueError("malformed transfer terminator")
            item = self.transfers.get(kind)
            if item is None or item["finished"] or int(fields[3]) != item["count"]:
                raise ValueError("unexpected transfer end")
            if (
                item["next_index"] != item["count"]
                or len(item["data"]) != item["length"]
            ):
                raise ValueError("missing or incomplete camera data")
            if hashlib.sha256(item["data"]).hexdigest() != item["sha256"]:
                raise ValueError("camera output SHA256 mismatch")
            item["finished"] = True
        else:
            raise ValueError("unknown transfer record " + command)

    def artifact(self, kind: str) -> bytes:
        """return only a fully verified, completed transfer"""
        item = self.transfers.get(kind)
        if item is None or not item["finished"]:
            raise ValueError(kind + " transfer did not complete")
        return bytes(item["data"])
