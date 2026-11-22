import argparse
import socket
import struct
import subprocess
import sys
from pathlib import Path


# ============================================================
# Automatic project paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

RTL_DIR = ROOT_DIR / "RTL_v6"
MODE_B_DIR = ROOT_DIR / "mode_b_v1"

RX_FILE = SCRIPT_DIR / "rx_packet.bin"
RX_HEX_FILE = SCRIPT_DIR / "rx_packet.hex"
TX_FILE = SCRIPT_DIR / "tx_reply.hex"
DO_FILE = SCRIPT_DIR / "run_mode_b.do"


# ============================================================
# Utility functions
# ============================================================

def recv_exact(sock, size):
    data = bytearray()

    while len(data) < size:
        chunk = sock.recv(size - len(data))

        if not chunk:
            raise ConnectionError("Connection closed before packet completed.")

        data.extend(chunk)

    return bytes(data)


def find_vsim(user_path=None):
    """
    Locate ModelSim vsim.exe.

    Priority:
      1. --vsim argument
      2. PATH
      3. Known installation used on this PC
    """

    if user_path:
        path = Path(user_path)

        if path.exists():
            return str(path)

        raise FileNotFoundError(
            f"Specified ModelSim executable does not exist:\n{path}"
        )

    # Try PATH first
    import shutil

    found = shutil.which("vsim")

    if found:
        return found

    # Known installation on current machine
    candidates = [
        Path(r"E:\Modelsim\modelsim_ase\win32aloem\vsim.exe"),
        Path(r"C:\Modeltech64_2020.4\win64\vsim.exe"),
        Path(r"C:\intelFPGA\20.1\modelsim_ase\win32aloem\vsim.exe"),
        Path(r"C:\intelFPGA_lite\20.1\modelsim_ase\win32aloem\vsim.exe"),
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "Could not find ModelSim vsim.exe.\n"
        "Use:\n"
        "python mode_b_bridge.py --vsim \"C:\\path\\to\\vsim.exe\""
    )


def check_project():
    required = [
        RTL_DIR,
        MODE_B_DIR,
        RTL_DIR / "rtl_params.vh",
        RTL_DIR / "dfl_exp_lut.mem",
        MODE_B_DIR / "mode_b_packet_parser.sv",
        MODE_B_DIR / "mode_b_sparse_decode.sv",
        MODE_B_DIR / "mode_b_compute_top.sv",
        MODE_B_DIR / "mode_b_result_builder.sv",
        MODE_B_DIR / "yolov8_mode_b_packet_top.sv",
        SCRIPT_DIR / "mode_b_socket_tb.sv",
        DO_FILE,
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        print("\n[BRIDGE] ERROR: Missing required files/directories:")

        for path in missing:
            print(f"  {path}")

        sys.exit(1)


def run_modelsim(vsim):
    # Remove old reply so we cannot accidentally return stale data
    if TX_FILE.exists():
        TX_FILE.unlink()

    print("[BRIDGE] Starting ModelSim...")

    command = [
        vsim,
        "-c",
        "-do",
        f'do "{DO_FILE.as_posix()}"'
    ]

    result = subprocess.run(
        command,
        cwd=str(SCRIPT_DIR)
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ModelSim failed with return code {result.returncode}"
        )

    if not TX_FILE.exists():
        raise RuntimeError(
            "ModelSim finished but tx_reply.hex was not generated."
        )

    # RTL TB writes one reply byte per line as hexadecimal text.
    reply = bytearray()

    with open(TX_FILE, "r", encoding="ascii") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = int(text, 16)
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid hex byte in tx_reply.hex at line {line_no}: {text!r}"
                ) from exc
            if not 0 <= value <= 0xFF:
                raise RuntimeError(
                    f"Out-of-range byte in tx_reply.hex at line {line_no}: {text!r}"
                )
            reply.append(value)

    if len(reply) == 0:
        raise RuntimeError("tx_reply.hex is empty.")

    return bytes(reply)


# ============================================================
# Main server
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Virtual FPGA Mode-B TCP bridge"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="TCP listen address (default: localhost)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="TCP port (default: 9000)"
    )

    parser.add_argument(
        "--vsim",
        default=None,
        help="Optional explicit path to vsim.exe"
    )

    args = parser.parse_args()

    check_project()

    try:
        vsim = find_vsim(args.vsim)
    except FileNotFoundError as exc:
        print(f"\n[BRIDGE] ERROR:\n{exc}")
        sys.exit(1)

    print("")
    print("========================================================")
    print(" VIRTUAL FPGA MODE-B TCP BRIDGE")
    print("========================================================")
    print(f" Project root : {ROOT_DIR}")
    print(f" RTL          : {RTL_DIR}")
    print(f" Mode-B RTL   : {MODE_B_DIR}")
    print(f" ModelSim     : {vsim}")
    print(f" RX file      : {RX_FILE}")
    print(f" TX file      : {TX_FILE}")
    print("========================================================")
    print("")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )

        server.bind((args.host, args.port))
        server.listen(1)

        print(
            f"[BRIDGE] listening on "
            f"{args.host}:{args.port}"
        )

        print(
            "[BRIDGE] Waiting for client..."
        )

        while True:
            conn, addr = server.accept()

            with conn:
                print("")
                print(
                    f"[BRIDGE] Client connected: "
                    f"{addr[0]}:{addr[1]}"
                )

                try:
                    # TCP framing:
                    # first 4 bytes = packet length, big endian
                    length_raw = recv_exact(conn, 4)

                    packet_length = struct.unpack(
                        "!I",
                        length_raw
                    )[0]

                    if packet_length <= 0:
                        raise ValueError(
                            "Received invalid packet length."
                        )

                    if packet_length > 10_000_000:
                        raise ValueError(
                            f"Packet too large: {packet_length} bytes"
                        )

                    print(
                        f"[BRIDGE] Receiving "
                        f"{packet_length} byte Mode-B packet..."
                    )

                    packet = recv_exact(
                        conn,
                        packet_length
                    )

                    RX_FILE.write_bytes(packet)

                    print(
                        f"[BRIDGE] Saved rx_packet.bin "
                        f"({len(packet)} bytes)"
                    )

                    # ModelSim TB reads one byte per line from rx_packet.hex
                    with open(RX_HEX_FILE, "w") as f:
                        for byte in packet:
                            f.write(f"{byte:02X}\n")

                    print(
                        f"[BRIDGE] Saved rx_packet.hex "
                        f"({len(packet)} bytes)"
                    )

                    # Basic magic sanity check
                    if len(packet) < 6:
                        raise ValueError(
                            "Mode-B packet is shorter than header."
                        )

                    if packet[0:4] != b"Y8PP":
                        raise ValueError(
                            "Invalid Mode-B magic."
                        )

                    print(
                        f"[BRIDGE] Packet magic OK, "
                        f"cmd=0x{packet[5]:02X}"
                    )

                    # Run virtual FPGA
                    reply = run_modelsim(vsim)

                    print(
                        f"[BRIDGE] ModelSim produced "
                        f"{len(reply)} byte reply"
                    )

                    # Return:
                    # 4-byte TCP length + raw Mode-B reply
                    conn.sendall(
                        struct.pack("!I", len(reply))
                    )

                    conn.sendall(reply)

                    print(
                        "[BRIDGE] Reply sent to client."
                    )

                except Exception as exc:
                    print(
                        f"[BRIDGE] ERROR: {exc}"
                    )

                    # Return zero-length reply on bridge failure
                    try:
                        conn.sendall(
                            struct.pack("!I", 0)
                        )
                    except Exception:
                        pass

            print("")
            print(
                "[BRIDGE] Waiting for next client..."
            )


if __name__ == "__main__":
    main()