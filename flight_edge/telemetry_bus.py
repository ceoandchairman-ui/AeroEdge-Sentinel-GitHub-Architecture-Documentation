"""gRPC telemetry ingestion bus for HIL micro-satellite autonomy testing."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import importlib
import logging
from pathlib import Path
import random
import subprocess
import sys
from typing import AsyncIterator, Iterable, Sequence

import grpc

from flight_edge.flight_agent import AgentContext, FlightMitigationAgent
from flight_edge.inference_engine import InferenceEngine

LOGGER = logging.getLogger(__name__)
PROTO_IMPORT_ROOT = Path(__file__).resolve().parents[1] / "_generated"


def ensure_proto_modules() -> tuple[object, object]:
    """Import generated protobuf modules or build them from protos at runtime."""
    sys.path.insert(0, str(PROTO_IMPORT_ROOT))

    try:
        telemetry_pb2 = importlib.import_module("telemetry_pb2")
        telemetry_pb2_grpc = importlib.import_module("telemetry_pb2_grpc")
        return telemetry_pb2, telemetry_pb2_grpc
    except ModuleNotFoundError:
        LOGGER.info("Generated protobuf modules not found; generating from protos.")

    proto_file = Path(__file__).resolve().parents[1] / "protos" / "telemetry.proto"
    PROTO_IMPORT_ROOT.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{proto_file.parent}",
        f"--python_out={PROTO_IMPORT_ROOT}",
        f"--grpc_python_out={PROTO_IMPORT_ROOT}",
        str(proto_file),
    ]
    subprocess.run(cmd, check=True)

    telemetry_pb2 = importlib.import_module("telemetry_pb2")
    telemetry_pb2_grpc = importlib.import_module("telemetry_pb2_grpc")
    return telemetry_pb2, telemetry_pb2_grpc


class TelemetryIngestServicer:
    """Server-side telemetry stream processor."""

    def __init__(self, pb2: object) -> None:
        self._pb2 = pb2
        self._engine = InferenceEngine()
        self._agent = FlightMitigationAgent()

    async def StreamTelemetry(self, request_iterator, context):  # noqa: N802
        packets_received = 0
        async for packet in request_iterator:
            packets_received += 1
            vector = flatten_packet(packet)
            result = self._engine.infer(vector)
            confidence = min(result.reconstruction_error / max(result.threshold, 1e-6), 1.0)

            agent_context = AgentContext(
                component_state=packet.component_state,
                reconstruction_error=result.reconstruction_error,
                threshold=result.threshold,
                anomaly_detected=result.is_anomaly,
                confidence=confidence,
            )
            updated = self._agent.evaluate(agent_context)

            LOGGER.info(
                "pkt=%d ts=%d state=%s err=%.6f threshold=%.6f anomaly=%s actions=%s",
                packets_received,
                packet.timestamp,
                packet.component_state,
                result.reconstruction_error,
                result.threshold,
                result.is_anomaly,
                updated.actions_taken,
            )

        return self._pb2.IngestAck(
            accepted=True,
            packets_received=packets_received,
            message="Telemetry stream processed.",
        )


def flatten_packet(packet) -> Sequence[float]:
    return [
        float(packet.imu.accel.x),
        float(packet.imu.accel.y),
        float(packet.imu.accel.z),
        float(packet.thermal.board_c),
        float(packet.thermal.battery_c),
        float(packet.battery_bus.voltage_v),
        float(packet.battery_bus.current_a),
    ]


def current_timestamp_ns() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1_000_000_000)


def generate_packet(pb2: object, anomaly: bool = False):
    """Generate simulated telemetry packet with optional anomaly injection."""
    accel_base = [0.01, -0.02, 0.98]
    accel_noise = [random.uniform(-0.015, 0.015) for _ in range(3)]

    board_temp = random.uniform(35.5, 42.0)
    battery_temp = random.uniform(31.0, 38.0)
    voltage = random.uniform(27.6, 29.1)
    current = random.uniform(1.0, 2.0)

    component_state = "nominal"
    if anomaly:
        board_temp += random.uniform(14.0, 24.0)
        voltage -= random.uniform(4.0, 8.0)
        current += random.uniform(1.5, 3.0)
        component_state = "power_bus_warning"

    return pb2.TelemetryPacket(
        timestamp=current_timestamp_ns(),
        component_state=component_state,
        imu=pb2.IMU(
            accel=pb2.Vector3(
                x=accel_base[0] + accel_noise[0],
                y=accel_base[1] + accel_noise[1],
                z=accel_base[2] + accel_noise[2],
            )
        ),
        thermal=pb2.Thermal(board_c=board_temp, battery_c=battery_temp),
        battery_bus=pb2.BatteryBus(voltage_v=voltage, current_a=current),
    )


async def encoded_stream(
    pb2: object,
    packet_rate_hz: float,
    total_packets: int,
    anomaly_period: int,
) -> AsyncIterator:
    """Create serialized packets, decode them, and yield protobuf messages."""
    period = 1.0 / packet_rate_hz
    for idx in range(total_packets):
        anomaly = anomaly_period > 0 and idx > 0 and idx % anomaly_period == 0
        packet = generate_packet(pb2, anomaly=anomaly)

        # Explicit encode/decode path for bus-level packet validation.
        wire = packet.SerializeToString()
        decoded = pb2.TelemetryPacket()
        decoded.ParseFromString(wire)

        yield decoded
        await asyncio.sleep(period)


async def run_server(pb2_grpc: object, pb2: object, host: str, port: int) -> grpc.aio.Server:
    server = grpc.aio.server()
    servicer = TelemetryIngestServicer(pb2=pb2)
    pb2_grpc.add_TelemetryIngestServiceServicer_to_server(servicer, server)
    bind_addr = f"{host}:{port}"
    server.add_insecure_port(bind_addr)
    await server.start()
    LOGGER.info("Telemetry gRPC server started on %s", bind_addr)
    return server


async def run_client(
    pb2_grpc: object,
    pb2: object,
    host: str,
    port: int,
    packet_rate_hz: float,
    total_packets: int,
    anomaly_period: int,
) -> None:
    target = f"{host}:{port}"
    async with grpc.aio.insecure_channel(target) as channel:
        stub = pb2_grpc.TelemetryIngestServiceStub(channel)
        ack = await stub.StreamTelemetry(
            encoded_stream(
                pb2=pb2,
                packet_rate_hz=packet_rate_hz,
                total_packets=total_packets,
                anomaly_period=anomaly_period,
            )
        )
        LOGGER.info(
            "Stream complete: accepted=%s packets=%d message=%s",
            ack.accepted,
            ack.packets_received,
            ack.message,
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="AeroEdge Sentinel telemetry bus")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50061)
    parser.add_argument("--packet-rate", type=float, default=25.0)
    parser.add_argument("--total-packets", type=int, default=200)
    parser.add_argument(
        "--anomaly-period",
        type=int,
        default=37,
        help="Inject one anomaly every N packets. Set 0 to disable.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    pb2, pb2_grpc = ensure_proto_modules()
    server = await run_server(pb2_grpc=pb2_grpc, pb2=pb2, host=args.host, port=args.port)

    try:
        await run_client(
            pb2_grpc=pb2_grpc,
            pb2=pb2,
            host=args.host,
            port=args.port,
            packet_rate_hz=args.packet_rate,
            total_packets=args.total_packets,
            anomaly_period=args.anomaly_period,
        )
    finally:
        await server.stop(grace=None)
        LOGGER.info("Telemetry gRPC server stopped.")


if __name__ == "__main__":
    asyncio.run(main())
