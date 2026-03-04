"""Circuit breaker pattern for provider health tracking."""

import enum
import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    def __init__(self, provider: str, retry_after: float):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(
            f"Circuit open for {provider}, retry after {retry_after:.1f}s"
        )


@dataclass
class _ProviderCircuit:
    state: CircuitState = CircuitState.CLOSED
    failures: list[float] = field(default_factory=list)
    opened_at: float = 0.0
    success_count: int = 0
    total_failures: int = 0


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        window_seconds: float = 60.0,
        cooldown_seconds: float = 30.0,
    ):
        self._failure_threshold = failure_threshold
        self._window = window_seconds
        self._cooldown = cooldown_seconds
        self._circuits: dict[str, _ProviderCircuit] = {}
        self._lock = threading.Lock()

    def _get_circuit(self, provider: str) -> _ProviderCircuit:
        if provider not in self._circuits:
            self._circuits[provider] = _ProviderCircuit()
        return self._circuits[provider]

    def _prune_old_failures(self, circuit: _ProviderCircuit, now: float) -> None:
        cutoff = now - self._window
        circuit.failures = [t for t in circuit.failures if t > cutoff]

    def can_execute(self, provider: str) -> bool:
        with self._lock:
            circuit = self._get_circuit(provider)
            now = time.monotonic()

            if circuit.state is CircuitState.CLOSED:
                return True

            if circuit.state is CircuitState.OPEN:
                elapsed = now - circuit.opened_at
                if elapsed >= self._cooldown:
                    circuit.state = CircuitState.HALF_OPEN
                    logger.info(
                        "Circuit for %s transitioned to HALF_OPEN after %.1fs cooldown",
                        provider, elapsed,
                    )
                    return True
                return False

            # HALF_OPEN — allow the single probe request
            return True

    def record_success(self, provider: str) -> None:
        with self._lock:
            circuit = self._get_circuit(provider)
            circuit.success_count += 1

            if circuit.state is CircuitState.HALF_OPEN:
                circuit.state = CircuitState.CLOSED
                circuit.failures.clear()
                logger.info("Circuit for %s closed after successful probe", provider)

    def record_failure(self, provider: str) -> None:
        with self._lock:
            circuit = self._get_circuit(provider)
            now = time.monotonic()
            circuit.total_failures += 1

            if circuit.state is CircuitState.HALF_OPEN:
                circuit.state = CircuitState.OPEN
                circuit.opened_at = now
                logger.warning("Circuit for %s re-opened after failed probe", provider)
                return

            circuit.failures.append(now)
            self._prune_old_failures(circuit, now)

            if (
                circuit.state is CircuitState.CLOSED
                and len(circuit.failures) >= self._failure_threshold
            ):
                circuit.state = CircuitState.OPEN
                circuit.opened_at = now
                logger.warning(
                    "Circuit for %s tripped OPEN after %d failures in %.0fs window",
                    provider, len(circuit.failures), self._window,
                )

    def get_status(self, provider: str) -> dict:
        with self._lock:
            circuit = self._get_circuit(provider)
            now = time.monotonic()
            self._prune_old_failures(circuit, now)

            result = {
                "state": circuit.state.value,
                "failures_in_window": len(circuit.failures),
                "total_failures": circuit.total_failures,
                "success_count": circuit.success_count,
            }

            if circuit.state is CircuitState.OPEN:
                elapsed = now - circuit.opened_at
                result["retry_after"] = round(max(self._cooldown - elapsed, 0), 1)

            return result

    def get_all_status(self) -> dict:
        with self._lock:
            now = time.monotonic()
            dashboard: dict[str, dict] = {}
            for name, circuit in self._circuits.items():
                self._prune_old_failures(circuit, now)
                entry = {
                    "state": circuit.state.value,
                    "failures_in_window": len(circuit.failures),
                    "total_failures": circuit.total_failures,
                    "success_count": circuit.success_count,
                }
                if circuit.state is CircuitState.OPEN:
                    elapsed = now - circuit.opened_at
                    entry["retry_after"] = round(max(self._cooldown - elapsed, 0), 1)
                dashboard[name] = entry
            return dashboard
