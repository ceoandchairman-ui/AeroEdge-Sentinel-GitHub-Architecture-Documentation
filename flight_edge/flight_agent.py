"""Graph-driven flight mitigation agent for anomaly response."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Callable, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


class FlightState(str, Enum):
    NOMINAL = "nominal"
    INVESTIGATE = "investigate"
    MITIGATE = "mitigate"
    SAFE_MODE = "safe_mode"
    RECOVERY = "recovery"


@dataclass
class AgentContext:
    component_state: str
    reconstruction_error: float
    threshold: float
    anomaly_detected: bool
    confidence: float = 0.0
    actions_taken: List[str] = field(default_factory=list)


class FlightMitigationAgent:
    """State-machine style autonomous mitigation agent.

    The state graph is intentionally explicit and inspectable for flight reviews.
    """

    def __init__(self) -> None:
        self.state = FlightState.NOMINAL
        self._tools: Dict[str, Callable[[], str]] = {
            "isolate_faulty_bus": isolate_faulty_bus,
            "load_shed_payload": load_shed_payload,
            "reorient_solar_arrays": reorient_solar_arrays,
        }

    def evaluate(self, context: AgentContext) -> AgentContext:
        """Run one planning cycle and return updated context."""
        LOGGER.info(
            "Agent cycle: state=%s anomaly=%s err=%.6f threshold=%.6f",
            self.state.value,
            context.anomaly_detected,
            context.reconstruction_error,
            context.threshold,
        )

        if self.state == FlightState.NOMINAL:
            if context.anomaly_detected:
                self.state = FlightState.INVESTIGATE
            return context

        if self.state == FlightState.INVESTIGATE:
            if context.anomaly_detected:
                self.state = FlightState.MITIGATE
                return self._execute_mitigation_plan(context)
            self.state = FlightState.NOMINAL
            return context

        if self.state == FlightState.MITIGATE:
            self.state = FlightState.SAFE_MODE
            return self._enter_safe_mode(context)

        if self.state == FlightState.SAFE_MODE:
            if not context.anomaly_detected:
                self.state = FlightState.RECOVERY
                context.actions_taken.append("stabilized_in_safe_mode")
            return context

        if self.state == FlightState.RECOVERY:
            context.actions_taken.append("resume_nominal_ops")
            self.state = FlightState.NOMINAL
            return context

        return context

    def _execute_mitigation_plan(self, context: AgentContext) -> AgentContext:
        """Select actions by component state and confidence."""
        state = context.component_state.lower()

        context.actions_taken.append(self._tools["isolate_faulty_bus"]())

        if "thermal" in state or context.reconstruction_error > (context.threshold * 1.4):
            context.actions_taken.append(self._tools["load_shed_payload"]())

        if "power" in state or context.confidence >= 0.75:
            context.actions_taken.append(self._tools["reorient_solar_arrays"]())

        return context

    def _enter_safe_mode(self, context: AgentContext) -> AgentContext:
        context.actions_taken.append("switch_to_safe_mode_profile")
        return context


def isolate_faulty_bus() -> str:
    """Mock command that isolates the suspected avionics bus."""
    LOGGER.warning("Mitigation tool invoked: isolate_faulty_bus")
    return "isolate_faulty_bus"


def load_shed_payload() -> str:
    """Mock command that sheds non-critical payload power load."""
    LOGGER.warning("Mitigation tool invoked: load_shed_payload")
    return "load_shed_payload"


def reorient_solar_arrays() -> str:
    """Mock command that reorients arrays to improve power margins."""
    LOGGER.warning("Mitigation tool invoked: reorient_solar_arrays")
    return "reorient_solar_arrays"


def default_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


if __name__ == "__main__":
    default_logger()
    agent = FlightMitigationAgent()
    test_context = AgentContext(
        component_state="power_bus_warning",
        reconstruction_error=0.18,
        threshold=0.08,
        anomaly_detected=True,
        confidence=0.82,
    )

    for _ in range(4):
        test_context = agent.evaluate(test_context)
        LOGGER.info("state=%s actions=%s", agent.state.value, test_context.actions_taken)
