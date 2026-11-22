"""Real-World Byte-Stream Domain: L7 Network Packet, AST & Industrial Telemetry Security Triage.

Generates authentic variable-length raw wire frames (32..96 bytes) combining binary L3/L4/L7
headers (IPv4 / 802.1Q VLAN / VXLAN encapsulation, ports, TCP control flags, Modbus MBAP, DNS)
with structured L7 ASCII/binary payloads (HTTP REST JSON ASTs, gRPC Protobuf varints, SQLi
tautologies, shell/path traversal payloads, malformed JSON ASTs, and SCADA coil overrides).

Supports:
  1. 8 Base Security Verdict Classes (`0..7`):
       0: ALLOW_HTTP_JSON       - Valid HTTP REST frame with balanced JSON AST
       1: ALLOW_GRPC_TELEMETRY  - Valid HTTP/2 gRPC length-prefixed Protobuf telemetry
       2: ALLOW_DNS_QUERY       - Valid UDP:53 DNS standard query with length-prefixed labels
       3: ALLOW_MODBUS_READ     - Valid TCP:502 Modbus FC=0x03/0x04 holding-register read
       4: DROP_SQLI_AST         - HTTP query/JSON containing SQL tautology / UNION AST injection
       5: DROP_CMD_TRAVERSAL    - Path traversal (`../`) or subshell/pipe command injection
       6: QUARANTINE_BAD_AST    - Malformed/truncated JSON AST (unbalanced braces/quotes)
       7: THROTTLE_SYN_SCAN     - Anomalous TCP SYN+FIN+URG Xmas/flood reconnaissance frame
  2. Live Protocol Encapsulation Drift (`mutate_regime()`):
       Simulates live enterprise network migration (e.g. raw IPv4 -> 802.1Q VLAN 4-byte tag shift
       + port remapping -> VXLAN/TLS-1.3 tunnel shift), invalidating static byte-offset rules so
       Rung 0 trips its RCE circuit breaker and re-crystallizes a new KV260 LUT6 firewall!
  3. Open-World Zero-Day Attack Spawn (`spawn_novel_class()` -> Class `8`):
       Spawns unseen `ISOLATE_ZERODAY_MODBUS_WRITE` (unauthorized SCADA FC=0x10 Write-Multiple-
       Registers override to safety PLC `0xFF00`), caught zero-shot by Rung 2.5 HC-SOINN!
  4. Zero-Tolerance Deterministic DPI / AST Contract Verifier (`verify` / `verify_subset`):
       Formal packet header + lexical AST balance & signature parser acting as the RCE gate.
"""

from __future__ import annotations

import numpy as np


CLASS_NAMES: dict[int, str] = {
    0: "ALLOW_HTTP_JSON",
    1: "ALLOW_GRPC_TELEMETRY",
    2: "ALLOW_DNS_QUERY",
    3: "ALLOW_MODBUS_READ",
    4: "DROP_SQLI_AST",
    5: "DROP_CMD_TRAVERSAL",
    6: "QUARANTINE_BAD_AST",
    7: "THROTTLE_SYN_SCAN",
    8: "ISOLATE_ZERODAY_MODBUS_WRITE",
}


class NetworkAstTelemetryEnv:
    """
    Real-world variable-length L7 packet + AST byte-stream environment implementing the exact
    same contract interface (`n_classes`, `regime_id`, `mutate_regime`, `spawn_novel_class`,
    `sample_microbatch`, `ground_truth`, `verify`, `verify_subset`) as `ModularConstraintEnv`.
    """

    def __init__(self, n_classes: int = 8, seed: int = 42):
        self.base_n_classes = n_classes
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)
        self.regime_id = 0

    @property
    def header_shift(self) -> int:
        """Simulates L2/tunnel encapsulation header shift (0 -> +4B 802.1Q VLAN -> +8B VXLAN)."""
        return (self.regime_id % 3) * 4

    @property
    def port_xor_key(self) -> int:
        """Simulates live service-mesh port / VIP remapping across regimes."""
        return (self.regime_id % 3) * 0x5A

    def mutate_regime(self) -> None:
        """Shifts wire encapsulation offset and service-mesh port mapping mid-stream."""
        self.regime_id += 1

    def spawn_novel_class(self) -> int:
        """Spawns unseen Zero-Day Class 8: unauthorized SCADA Modbus FC=0x10 PLC override."""
        if self.n_classes < 9:
            self.n_classes = 9
        return self.n_classes - 1

    def _build_packet_for_class(self, cls_id: int) -> np.ndarray:
        shift = self.header_shift
        pxor = self.port_xor_key
        extra_tail = int(self.rng.integers(0, 25))

        # Encapsulation prefix (0B raw IPv4, 4B 802.1Q VLAN, 8B VXLAN)
        prefix = bytearray()
        if shift == 4:
            vlan_id = int(self.rng.integers(10, 250))
            prefix.extend([0x81, 0x00, 0x00, vlan_id])
        elif shift == 8:
            vni = int(self.rng.integers(1, 250))
            prefix.extend([0x08, 0x00, 0x00, 0x00, 0x00, 0x01, vni, 0x00])

        hdr = bytearray(16)
        hdr[0] = 0x45 ^ (self.regime_id * 0x11)  # Version/IHL under active encapsulation
        hdr[1] = int(self.rng.integers(0, 4))    # DSCP/ECN

        # Protocol-specific L4 header + authentic variable-length L7 wire payload
        if cls_id == 0:
            # ALLOW_HTTP_JSON: TCP:443/80 + valid balanced JSON AST
            hdr[2], hdr[3] = 0x01 ^ pxor, 0xBB ^ pxor
            hdr[4] = 0x18  # PSH|ACK
            hdr[5] = 0x10  # L7=HTTP_REST
            hdr[6:10] = b"HTTP"
            item_id = int(self.rng.integers(10, 99))
            payload = (
                f'POST /v1/telemetry HTTP/1.1\r\n{{"sensor":{item_id},"ok":true}}'
            ).encode("ascii")
        elif cls_id == 1:
            # ALLOW_GRPC_TELEMETRY: TCP:50051 + HTTP/2 gRPC Protobuf varints
            hdr[2], hdr[3] = 0xC3 ^ pxor, 0x83 ^ pxor
            hdr[4] = 0x18  # PSH|ACK
            hdr[5] = 0x20  # L7=GRPC_H2
            hdr[6:10] = b"GRPC"
            metric_v = int(self.rng.integers(1, 127))
            payload = (
                b"\x00\x00\x00\x00\x16\x08"
                + bytes([metric_v])
                + b"\x12\x10clai.node.health"
            )
        elif cls_id == 2:
            # ALLOW_DNS_QUERY: UDP:53 + standard query A-record labels
            hdr[2], hdr[3] = 0x00 ^ pxor, 0x35 ^ pxor
            hdr[4] = 0x00  # UDP
            hdr[5] = 0x30  # L7=DNS_Q
            hdr[6:10] = b"DNSQ"
            txid = int(self.rng.integers(1, 255))
            payload = (
                bytes([txid, 0x42, 0x01, 0x00, 0x00, 0x01])
                + b"\x04edge\x05clai0\x03internal\x00\x00\x01\x00\x01"
            )
        elif cls_id == 3:
            # ALLOW_MODBUS_READ: TCP:502 + safe FC=0x03 Read Holding Registers
            hdr[2], hdr[3] = 0x01 ^ pxor, 0xF6 ^ pxor
            hdr[4] = 0x18  # PSH|ACK
            hdr[5] = 0x40  # L7=MODBUS_SAFE
            hdr[6:10] = b"MBAP"
            reg_addr = int(self.rng.integers(0x00, 0x40))
            payload = bytes([0x00, 0x01, 0x00, 0x00, 0x00, 0x06, 0x01, 0x03, 0x00, reg_addr, 0x00, 0x04])
        elif cls_id == 4:
            # DROP_SQLI_AST: HTTP request containing SQL injection tautology / UNION SELECT
            hdr[2], hdr[3] = 0x01 ^ pxor, 0xBB ^ pxor
            hdr[4] = 0x18
            hdr[5] = 0x10
            hdr[6:10] = b"HTTP"
            sqli_tok = "' OR 1=1 UNION SELECT password FROM users--"
            payload = f"GET /api/user?filter={sqli_tok} HTTP/1.1\r\n".encode("ascii")
        elif cls_id == 5:
            # DROP_CMD_TRAVERSAL: Path traversal `../` + shell command execution
            hdr[2], hdr[3] = 0x00 ^ pxor, 0x50 ^ pxor
            hdr[4] = 0x18
            hdr[5] = 0x10
            hdr[6:10] = b"HTTP"
            payload = b"GET /../../../../etc/shadow; $(cat /root/key|sh) HTTP/1.1\r\n"
        elif cls_id == 6:
            # QUARANTINE_BAD_AST: Truncated / unbalanced JSON AST payload
            hdr[2], hdr[3] = 0x01 ^ pxor, 0xBB ^ pxor
            hdr[4] = 0x18
            hdr[5] = 0x10
            hdr[6:10] = b"HTTP"
            payload = b'POST /v1/config HTTP/1.1\r\n{"policy":[{"allow":true,"broken":[1,2'
        elif cls_id == 7:
            # THROTTLE_SYN_SCAN: Anomalous TCP SYN|FIN|URG (0x2B) flood/Xmas scan
            scan_port = int(self.rng.integers(1024, 65000))
            hdr[2], hdr[3] = ((scan_port >> 8) & 0xFF) ^ pxor, (scan_port & 0xFF) ^ pxor
            hdr[4] = 0x2B  # SYN|FIN|URG|ACK illegal TCP flag combination!
            hdr[5] = 0xFF  # Unknown/probe
            hdr[6:10] = b"SCAN"
            payload = bytes(self.rng.integers(0x90, 0xFF, size=(20,), dtype=np.uint8).tolist())
        else:
            # Class 8 (Zero-Day): ISOLATE_ZERODAY_MODBUS_WRITE (FC=0x10 PLC safety override!)
            hdr[2], hdr[3] = 0x01 ^ pxor, 0xF6 ^ pxor
            hdr[4] = 0x18
            hdr[5] = 0x40
            hdr[6:10] = b"MBAP"
            payload = b"\x00\x09\x00\x00\x00\x0B\x01\x10\xFF\x00\x00\x02\x04\xDE\xAD\xBE\xEF"

        # Compute deterministic AST / lexical header summary slots (emulating hardware parser Stage 0)
        open_braces = payload.count(b"{") + payload.count(b"[")
        close_braces = payload.count(b"}") + payload.count(b"]")
        hdr[10] = (128 + open_braces - close_braces) & 0xFF
        hdr[11] = 0xE0 if (b"UNION SELECT" in payload or b"OR 1=1" in payload) else 0x10
        hdr[12] = 0xE0 if (b"../" in payload or b"$(" in payload) else 0x10
        hdr[13] = len(payload) & 0xFF

        pad_bytes = bytes(self.rng.integers(0x20, 0x5A, size=(extra_tail,), dtype=np.uint8).tolist())
        frame = bytes(prefix) + bytes(hdr) + payload + pad_bytes
        arr = np.frombuffer(frame, dtype=np.uint8).copy()
        return arr

    def ground_truth(self, x: np.ndarray) -> int:
        """
        Deterministic Deep-Packet & Lexical AST Contract Inspector (RCE Ground Truth).
        Evaluates packet under the CURRENT active network encapsulation regime (`header_shift`
        and `port_xor_key`). Stale rules from a prior encapsulation regime misparse offsets!
        """
        shift = self.header_shift
        pxor = self.port_xor_key
        if len(x) < shift + 20:
            return 6

        hdr = x[shift : shift + 16]
        expected_ver = 0x45 ^ (self.regime_id * 0x11)
        if int(hdr[0]) != expected_ver:
            return 6

        tcp_flags = int(hdr[4])
        magic = bytes(hdr[6:10].tolist())
        port_hi = int(hdr[2]) ^ pxor
        port_lo = int(hdr[3]) ^ pxor
        dst_port = (port_hi << 8) | port_lo
        payload = bytes(x[shift + 16 :].tolist())

        # 1. L4 Anomalous TCP Flag Reconnaissance Check (SYN+FIN+URG = 0x2B)
        if (tcp_flags & 0x23) == 0x23 or magic == b"SCAN":
            return 7

        # 2. Industrial Modbus TCP (Port 502 / MBAP)
        if magic == b"MBAP" and dst_port == 502 and len(payload) >= 10:
            func_code = payload[7]
            reg_hi = payload[8]
            if func_code == 0x10 and reg_hi == 0xFF and self.n_classes >= 9:
                return 8  # Zero-Day Unauthorized Safety PLC Write!
            if func_code in (0x03, 0x04):
                return 3
            return 6

        # 3. DNS UDP:53 Standard Query
        if magic == b"DNSQ" and dst_port == 53:
            return 2

        # 4. gRPC HTTP/2 Length-Prefixed Protobuf Telemetry
        if magic == b"GRPC" and dst_port == 50051:
            return 1

        # 5. HTTP/1.1 & JSON AST Deep Inspection
        if magic == b"HTTP":
            if b"UNION SELECT" in payload or b"OR 1=1" in payload:
                return 4
            if b"../" in payload or b"$(" in payload:
                return 5
            if b"{" in payload:
                opens = payload.count(b"{") + payload.count(b"[")
                closes = payload.count(b"}") + payload.count(b"]")
                if opens != closes:
                    return 6
            return 0

        return 6

    def sample_episode(self) -> tuple[np.ndarray, int]:
        cls_id = int(self.rng.integers(0, self.n_classes))
        pkt = self._build_packet_for_class(cls_id)
        return pkt, self.ground_truth(pkt)

    def sample_microbatch(self, batch_size: int = 64) -> tuple[list[np.ndarray], np.ndarray]:
        episodes: list[np.ndarray] = []
        targets = np.empty(batch_size, dtype=np.int64)
        for i in range(batch_size):
            pkt, gt = self.sample_episode()
            episodes.append(pkt)
            targets[i] = gt
        return episodes, targets

    def verify(self, x: np.ndarray, proposed_action: int) -> bool:
        """Zero-tolerance deterministic DPI + AST RCE contract verifier."""
        return int(proposed_action) == self.ground_truth(x)

    def verify_subset(
        self, episodes: list[np.ndarray], indices: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        return np.array(
            [self.verify(episodes[int(idx)], int(act)) for idx, act in zip(indices, actions)],
            dtype=bool,
        )
