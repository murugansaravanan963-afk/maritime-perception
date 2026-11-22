import argparse
import socket
import struct
import sys
from pathlib import Path


# ============================================================
# Automatic project paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

FRAME_DIR = ROOT_DIR / "RTL_v6" / "frame_0000"


# Maritime960 parameters
N_ANCHORS = 18900
NUM_CLASSES = 6

SCORE_THRESHOLD = 64

BBOX_VALUES_PER_ANCHOR = 64

MODE_B_RECORD_BYTES = 68


# ============================================================
# CRC-16 / CCITT-FALSE
#
# poly = 0x1021
# init = 0xFFFF
# ============================================================

def crc16_ccitt_false(data):
    crc = 0xFFFF

    for byte in data:
        crc ^= byte << 8

        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF

    return crc


# ============================================================
# TCP helper
# ============================================================

def recv_exact(sock, size):
    data = bytearray()

    while len(data) < size:
        chunk = sock.recv(size - len(data))

        if not chunk:
            raise ConnectionError(
                "Connection closed before reply completed."
            )

        data.extend(chunk)

    return bytes(data)


# ============================================================
# Load hex vectors
# ============================================================

def load_hex_file(path):
    values = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            values.append(
                int(line, 16)
            )

    return values


# ============================================================
# Build Mode-B packet
# ============================================================

def build_mode_b_packet(frame_dir, frame_id=0):
    bbox_path = frame_dir / "in_bbox.hex"
    score_path = frame_dir / "in_score.hex"

    if not bbox_path.exists():
        raise FileNotFoundError(
            f"Missing {bbox_path}"
        )

    if not score_path.exists():
        raise FileNotFoundError(
            f"Missing {score_path}"
        )

    print("[CLIENT] Loading frame vectors...")

    bbox = load_hex_file(bbox_path)
    scores = load_hex_file(score_path)

    expected_bbox = (
        N_ANCHORS *
        BBOX_VALUES_PER_ANCHOR
    )

    expected_scores = (
        N_ANCHORS *
        NUM_CLASSES
    )

    if len(bbox) != expected_bbox:
        raise ValueError(
            f"bbox count mismatch: "
            f"{len(bbox)} != {expected_bbox}"
        )

    if len(scores) != expected_scores:
        raise ValueError(
            f"score count mismatch: "
            f"{len(scores)} != {expected_scores}"
        )

    candidates = []

    # --------------------------------------------------------
    # Pi-style Mode-B prefilter
    #
    # argmax:
    #   lowest class index wins ties
    #
    # pass:
    #   score >= 64
    # --------------------------------------------------------

    for anchor in range(N_ANCHORS):
        base = anchor * NUM_CLASSES

        best_cls = 0
        best_score = scores[base]

        for cls in range(1, NUM_CLASSES):
            score = scores[base + cls]

            if score > best_score:
                best_score = score
                best_cls = cls

        if best_score >= SCORE_THRESHOLD:
            bbox_base = (
                anchor *
                BBOX_VALUES_PER_ANCHOR
            )

            bbox_values = bbox[
                bbox_base:
                bbox_base +
                BBOX_VALUES_PER_ANCHOR
            ]

            candidates.append(
                (
                    anchor,
                    best_cls,
                    best_score,
                    bbox_values,
                )
            )

    print(
        f"[CLIENT] candidates={len(candidates)}"
    )

    if candidates:
        first = candidates[0]
        last = candidates[-1]

        print(
            "[CLIENT] first candidate: "
            f"anchor={first[0]} "
            f"cls={first[1]} "
            f"score={first[2]}"
        )

        print(
            "[CLIENT] last candidate : "
            f"anchor={last[0]} "
            f"cls={last[1]} "
            f"score={last[2]}"
        )

    # --------------------------------------------------------
    # Build payload
    #
    # Per candidate = 68 B:
    #
    # anchor_idx : u16 BE
    # cls        : u8
    # score_q    : u8
    # bbox[64]   : 64 × u8
    # --------------------------------------------------------

    payload = bytearray()

    for anchor, cls, score, bbox_values in candidates:
        payload.extend(
            struct.pack(">H", anchor)
        )

        payload.append(cls & 0xFF)
        payload.append(score & 0xFF)

        payload.extend(
            bytes(v & 0xFF for v in bbox_values)
        )

    expected_payload = (
        len(candidates) *
        MODE_B_RECORD_BYTES
    )

    if len(payload) != expected_payload:
        raise RuntimeError(
            "Internal Mode-B payload size error."
        )

    # --------------------------------------------------------
    # 16-byte Y8PP header
    #
    # 0  : magic[4]
    # 4  : version
    # 5  : cmd = 0x03 FRAME_CANDIDATES
    # 6  : frame_id u16
    # 8  : candidate count u16
    # 10 : n_classes
    # 11 : bins
    # 12 : payload_len u32
    # --------------------------------------------------------

    header = bytearray()

    header.extend(b"Y8PP")

    header.append(1)
    header.append(0x03)

    header.extend(
        struct.pack(">H", frame_id)
    )

    header.extend(
        struct.pack(">H", len(candidates))
    )

    header.append(NUM_CLASSES)
    header.append(16)

    header.extend(
        struct.pack(">I", len(payload))
    )

    if len(header) != 16:
        raise RuntimeError(
            "Internal header size error."
        )

    packet_without_crc = (
        bytes(header) +
        bytes(payload)
    )

    crc = crc16_ccitt_false(
        packet_without_crc
    )

    packet = (
        packet_without_crc +
        struct.pack(">H", crc)
    )

    print(
        f"[CLIENT] Mode-B packet="
        f"{len(packet)} bytes"
    )

    print(
        f"[CLIENT] payload="
        f"{len(payload)} bytes"
    )

    print(
        f"[CLIENT] CRC16=0x{crc:04X}"
    )

    return packet


# ============================================================
# Parse FPGA reply
# ============================================================

def parse_reply(reply):
    if len(reply) < 14:
        raise ValueError(
            f"Reply too short: {len(reply)} bytes"
        )

    status = reply[0]

    frame_id = struct.unpack(
        ">H",
        reply[1:3]
    )[0]

    n_det = reply[3]

    cycles_first = struct.unpack(
        ">I",
        reply[4:8]
    )[0]

    cycles_last = struct.unpack(
        ">I",
        reply[8:12]
    )[0]

    expected_length = (
        12 +
        12 * n_det +
        2
    )

    if len(reply) != expected_length:
        raise ValueError(
            f"Reply length mismatch: "
            f"{len(reply)} != {expected_length}"
        )

    received_crc = struct.unpack(
        ">H",
        reply[-2:]
    )[0]

    calculated_crc = crc16_ccitt_false(
        reply[:-2]
    )

    print("")
    print("========================================================")
    print(" VIRTUAL FPGA MODE-B REPLY")
    print("========================================================")

    print(
        f" status               : 0x{status:02X}"
    )

    print(
        f" frame_id             : {frame_id}"
    )

    print(
        f" n_det                : {n_det}"
    )

    print(
        f" cycles first->done   : {cycles_first}"
    )

    print(
        f" cycles last->done    : {cycles_last}"
    )

    print(
        f" reply bytes          : {len(reply)}"
    )

    print(
        f" CRC received         : 0x{received_crc:04X}"
    )

    print(
        f" CRC calculated       : 0x{calculated_crc:04X}"
    )

    if received_crc != calculated_crc:
        raise ValueError(
            "Reply CRC mismatch."
        )

    detections = []

    offset = 12

    for i in range(n_det):
        record = reply[
            offset:
            offset + 12
        ]

        x1 = struct.unpack(
            ">H",
            record[0:2]
        )[0]

        y1 = struct.unpack(
            ">H",
            record[2:4]
        )[0]

        x2 = struct.unpack(
            ">H",
            record[4:6]
        )[0]

        y2 = struct.unpack(
            ">H",
            record[6:8]
        )[0]

        cls = record[8]
        score = record[9]

        anchor = struct.unpack(
            ">H",
            record[10:12]
        )[0]

        detections.append(
            (
                x1,
                y1,
                x2,
                y2,
                cls,
                score,
                anchor,
            )
        )

        print(
            f" det[{i}] "
            f"box=({x1},{y1})-({x2},{y2}) "
            f"cls={cls} "
            f"score={score} "
            f"anchor={anchor}"
        )

        offset += 12

    print("========================================================")

    if status & 0x01:
        print("[CLIENT] FPGA status: OK")
    else:
        print(
            "[CLIENT] WARNING: FPGA OK bit is not set."
        )

    return detections


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Virtual FPGA Mode-B localhost client"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bridge host"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Bridge TCP port"
    )

    args = parser.parse_args()

    if not FRAME_DIR.exists():
        print(
            f"[CLIENT] ERROR: frame directory not found:\n"
            f"{FRAME_DIR}"
        )

        sys.exit(1)

    print("")
    print("========================================================")
    print(" VIRTUAL FPGA MODE-B LOCAL CLIENT")
    print("========================================================")
    print(f" Frame directory : {FRAME_DIR}")
    print(f" Server          : {args.host}:{args.port}")
    print("========================================================")
    print("")

    try:
        packet = build_mode_b_packet(
            FRAME_DIR,
            frame_id=0
        )

        print("")
        print(
            f"[CLIENT] Connecting to "
            f"{args.host}:{args.port}..."
        )

        with socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        ) as sock:

            sock.connect(
                (args.host, args.port)
            )

            # TCP-only framing:
            # 4-byte packet length prefix
            sock.sendall(
                struct.pack("!I", len(packet))
            )

            sock.sendall(packet)

            print(
                f"[CLIENT] Sent "
                f"{len(packet)} byte Mode-B packet."
            )

            # Receive reply length
            reply_len_raw = recv_exact(
                sock,
                4
            )

            reply_length = struct.unpack(
                "!I",
                reply_len_raw
            )[0]

            if reply_length == 0:
                raise RuntimeError(
                    "Bridge reported ModelSim failure."
                )

            print(
                f"[CLIENT] Receiving "
                f"{reply_length} byte reply..."
            )

            reply = recv_exact(
                sock,
                reply_length
            )

        parse_reply(reply)

        print("")
        print(
            "[CLIENT] TCP -> ModelSim RTL -> TCP completed."
        )

    except Exception as exc:
        print("")
        print(
            f"[CLIENT] ERROR: {exc}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()