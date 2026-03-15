"""
Circuit Breaker & Self-Healing Layer

Monitors the health of the entire data pipeline and automatically:

1. DETECTS problems:
   - Source going offline (consecutive fetch failures)
   - Source producing garbage (validation rate drops)
   - Sudden claim volume spikes (data poisoning attempt)
   - Confidence distribution shifts (something upstream changed)
   - Single source dominating predictions (concentration risk)

2. RESPONDS to problems:
   - Degrades unhealthy sources (reduces weight, doesn't remove)
   - Quarantines suspicious claim batches
   - Raises minimum consensus thresholds when sources are limited
   - Alerts on pipeline health degradation

3. HEALS:
   - Gradually restores degraded sources as they prove reliable again
   - Clears quarantine after manual or time-based review
   - Adjusts thresholds back as source coverage improves

This is the "actuary checking the actuary" layer.
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta

from .data_source import DataSource, SourceRegistry, TrustTier
from .cross_validator import ValidationResult, ConsensusLevel
from ..utils.logger import get_logger

logger = get_logger('mirofish.circuit_breaker')


class BreakerState(Enum):
    """Circuit breaker states per source."""
    CLOSED = "closed"           # Normal operation
    HALF_OPEN = "half_open"     # Testing if source recovered
    OPEN = "open"               # Source is cut off


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Pipeline health alert."""
    severity: AlertSeverity
    source_id: Optional[str]
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "source_id": self.source_id,
            "message": self.message,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


@dataclass
class SourceBreaker:
    """Circuit breaker state for a single source."""
    source_id: str
    state: BreakerState = BreakerState.CLOSED
    failure_count: int = 0
    success_count_since_half_open: int = 0
    last_state_change: datetime = field(default_factory=datetime.utcnow)
    degradation_factor: float = 1.0     # 1.0 = full trust, 0.0 = fully degraded
    quarantined_claim_ids: List[str] = field(default_factory=list)


class CircuitBreaker:
    """
    Monitors pipeline health and automatically responds to problems.

    Think of this as the immune system for the data pipeline.
    """

    # Thresholds
    MAX_CONSECUTIVE_FAILURES = 5        # Failures before OPEN
    HALF_OPEN_TEST_WINDOW = 3           # Successes needed to close
    HALF_OPEN_COOLDOWN_MINUTES = 30     # Wait before testing again
    MIN_VALIDATION_RATE = 0.3           # Below this, source is suspect
    MAX_CLAIM_SPIKE_RATIO = 3.0         # 3x normal volume = suspicious
    MAX_SOURCE_CONCENTRATION = 0.5      # No source > 50% of total claims
    DEGRADATION_STEP = 0.2              # How much to degrade per incident
    RECOVERY_STEP = 0.05                # How much to recover per success cycle
    MIN_HEALTHY_SOURCES = 2             # Pipeline won't run with fewer

    def __init__(self, registry: SourceRegistry):
        self.registry = registry
        self._breakers: Dict[str, SourceBreaker] = {}
        self._alerts: List[Alert] = []
        self._claim_volume_history: Dict[str, List[int]] = {}
        self._on_alert_callbacks: List[Callable[[Alert], None]] = []

    def on_alert(self, callback: Callable[[Alert], None]):
        """Register a callback for alerts."""
        self._on_alert_callbacks.append(callback)

    def get_breaker(self, source_id: str) -> SourceBreaker:
        """Get or create breaker for a source."""
        if source_id not in self._breakers:
            self._breakers[source_id] = SourceBreaker(source_id=source_id)
        return self._breakers[source_id]

    # ── Detection ──────────────────────────────────────────────

    def check_source_health(self, source: DataSource) -> List[Alert]:
        """Run all health checks on a single source."""
        alerts = []
        breaker = self.get_breaker(source.source_id)

        # Check consecutive failures
        if source.health.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            if breaker.state != BreakerState.OPEN:
                self._open_breaker(breaker)
                alerts.append(self._create_alert(
                    AlertSeverity.CRITICAL,
                    source.source_id,
                    f"Source {source.source_id} circuit OPENED after "
                    f"{source.health.consecutive_failures} consecutive failures",
                    "consecutive_failures",
                    float(source.health.consecutive_failures),
                    float(self.MAX_CONSECUTIVE_FAILURES),
                ))

        # Check validation rate
        if (source.health.total_claims >= 10 and
                source.health.validation_rate < self.MIN_VALIDATION_RATE):
            alerts.append(self._create_alert(
                AlertSeverity.WARNING,
                source.source_id,
                f"Source {source.source_id} validation rate dropped to "
                f"{source.health.validation_rate:.1%}",
                "validation_rate",
                source.health.validation_rate,
                self.MIN_VALIDATION_RATE,
            ))
            self._degrade_source(breaker)

        return alerts

    def check_claim_volume(self, source_id: str, claim_count: int) -> List[Alert]:
        """
        Detect suspicious spikes in claim volume.

        A sudden 3x spike could indicate data poisoning or a source
        returning garbage data.
        """
        alerts = []
        if source_id not in self._claim_volume_history:
            self._claim_volume_history[source_id] = []

        history = self._claim_volume_history[source_id]
        history.append(claim_count)

        # Need at least 3 data points to detect spikes
        if len(history) >= 3:
            recent_avg = sum(history[-3:]) / 3
            if recent_avg > 0 and claim_count > recent_avg * self.MAX_CLAIM_SPIKE_RATIO:
                alerts.append(self._create_alert(
                    AlertSeverity.WARNING,
                    source_id,
                    f"Claim volume spike from {source_id}: {claim_count} "
                    f"(avg: {recent_avg:.0f})",
                    "claim_volume_spike",
                    float(claim_count),
                    recent_avg * self.MAX_CLAIM_SPIKE_RATIO,
                ))

        # Keep only last 20 data points
        if len(history) > 20:
            self._claim_volume_history[source_id] = history[-20:]

        return alerts

    def check_source_concentration(
        self,
        results: List[ValidationResult]
    ) -> List[Alert]:
        """
        Detect if one source is dominating the validated claims.

        If >50% of actionable claims come from one source, that's
        concentration risk — one source failing would crater the pipeline.
        """
        alerts = []
        actionable = [r for r in results if r.is_actionable]
        if not actionable:
            return alerts

        source_counts: Dict[str, int] = {}
        for r in actionable:
            sid = r.claim.source_id
            source_counts[sid] = source_counts.get(sid, 0) + 1

        total = len(actionable)
        for sid, count in source_counts.items():
            concentration = count / total
            if concentration > self.MAX_SOURCE_CONCENTRATION:
                alerts.append(self._create_alert(
                    AlertSeverity.WARNING,
                    sid,
                    f"Source {sid} accounts for {concentration:.0%} of "
                    f"actionable claims — concentration risk",
                    "source_concentration",
                    concentration,
                    self.MAX_SOURCE_CONCENTRATION,
                ))

        return alerts

    def check_pipeline_viability(self) -> List[Alert]:
        """
        Check if the pipeline has enough healthy sources to operate.

        If too many sources are degraded/open, the pipeline should
        refuse to generate predictions rather than produce garbage.
        """
        alerts = []
        healthy = [
            s for s in self.registry.get_all_sources()
            if not s.health.is_degraded and
            self.get_breaker(s.source_id).state != BreakerState.OPEN
        ]

        if len(healthy) < self.MIN_HEALTHY_SOURCES:
            alerts.append(self._create_alert(
                AlertSeverity.CRITICAL,
                None,
                f"Pipeline has only {len(healthy)} healthy sources "
                f"(minimum: {self.MIN_HEALTHY_SOURCES}). "
                f"Predictions should be halted until sources recover.",
                "healthy_source_count",
                float(len(healthy)),
                float(self.MIN_HEALTHY_SOURCES),
            ))

        return alerts

    # ── Response ───────────────────────────────────────────────

    def _open_breaker(self, breaker: SourceBreaker):
        """Cut off a source."""
        breaker.state = BreakerState.OPEN
        breaker.last_state_change = datetime.utcnow()
        breaker.degradation_factor = 0.0

        # Mark source as degraded in registry
        source = self.registry.get_source(breaker.source_id)
        if source:
            source.health.is_degraded = True

        logger.warning(f"Circuit OPENED for source: {breaker.source_id}")

    def _degrade_source(self, breaker: SourceBreaker):
        """Reduce trust in a source without fully cutting it off."""
        breaker.degradation_factor = max(
            0.0,
            breaker.degradation_factor - self.DEGRADATION_STEP
        )
        if breaker.degradation_factor == 0.0:
            self._open_breaker(breaker)
        else:
            logger.info(
                f"Degraded source {breaker.source_id} to "
                f"{breaker.degradation_factor:.0%} trust"
            )

    def quarantine_claims(self, breaker: SourceBreaker, claim_ids: List[str]):
        """Hold claims for review instead of processing them."""
        breaker.quarantined_claim_ids.extend(claim_ids)
        logger.info(
            f"Quarantined {len(claim_ids)} claims from {breaker.source_id}"
        )

    def should_allow_source(self, source_id: str) -> bool:
        """
        Should we accept data from this source right now?

        Called before fetching from a source.
        """
        breaker = self.get_breaker(source_id)

        if breaker.state == BreakerState.CLOSED:
            return True

        if breaker.state == BreakerState.OPEN:
            # Check if enough time has passed to try again
            elapsed = datetime.utcnow() - breaker.last_state_change
            if elapsed > timedelta(minutes=self.HALF_OPEN_COOLDOWN_MINUTES):
                breaker.state = BreakerState.HALF_OPEN
                breaker.success_count_since_half_open = 0
                breaker.last_state_change = datetime.utcnow()
                logger.info(f"Circuit HALF-OPEN for source: {source_id}")
                return True
            return False

        if breaker.state == BreakerState.HALF_OPEN:
            return True

        return False

    def record_source_success(self, source_id: str):
        """Source successfully fetched and produced valid claims."""
        breaker = self.get_breaker(source_id)

        if breaker.state == BreakerState.HALF_OPEN:
            breaker.success_count_since_half_open += 1
            if breaker.success_count_since_half_open >= self.HALF_OPEN_TEST_WINDOW:
                # Source has recovered
                breaker.state = BreakerState.CLOSED
                breaker.failure_count = 0
                breaker.last_state_change = datetime.utcnow()
                source = self.registry.get_source(source_id)
                if source:
                    source.health.is_degraded = False
                logger.info(f"Circuit CLOSED for source: {source_id} (recovered)")

        # Gradually restore degradation
        if breaker.degradation_factor < 1.0:
            breaker.degradation_factor = min(
                1.0,
                breaker.degradation_factor + self.RECOVERY_STEP
            )

    def record_source_failure(self, source_id: str):
        """Source failed to fetch or produced invalid claims."""
        breaker = self.get_breaker(source_id)
        breaker.failure_count += 1

        if breaker.state == BreakerState.HALF_OPEN:
            # Failed during recovery test — back to OPEN
            self._open_breaker(breaker)
            logger.warning(
                f"Source {source_id} failed during HALF-OPEN test, "
                f"circuit re-OPENED"
            )

    # ── Healing ────────────────────────────────────────────────

    def run_health_cycle(self) -> Dict[str, Any]:
        """
        Run a full health check cycle across all sources.

        Call this periodically (e.g., every fetch cycle).
        Returns a health report.
        """
        all_alerts = []

        for source in self.registry.get_all_sources():
            alerts = self.check_source_health(source)
            all_alerts.extend(alerts)

        pipeline_alerts = self.check_pipeline_viability()
        all_alerts.extend(pipeline_alerts)

        # Fire callbacks
        for alert in all_alerts:
            self._alerts.append(alert)
            for cb in self._on_alert_callbacks:
                try:
                    cb(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "alerts": [a.to_dict() for a in all_alerts],
            "breaker_states": {
                sid: {
                    "state": b.state.value,
                    "degradation": round(b.degradation_factor, 2),
                    "quarantined_claims": len(b.quarantined_claim_ids),
                }
                for sid, b in self._breakers.items()
            },
            "pipeline_viable": not any(
                a.severity == AlertSeverity.CRITICAL and a.metric == "healthy_source_count"
                for a in all_alerts
            ),
        }

    def get_degradation_factor(self, source_id: str) -> float:
        """
        Get the current degradation factor for a source.

        Used by the confidence scorer to further adjust weights.
        """
        breaker = self.get_breaker(source_id)
        return breaker.degradation_factor

    def get_alerts(self, severity: Optional[AlertSeverity] = None,
                   unacknowledged_only: bool = False) -> List[Alert]:
        """Get alerts, optionally filtered."""
        alerts = self._alerts
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        return alerts

    def acknowledge_alert(self, index: int):
        """Mark an alert as acknowledged."""
        if 0 <= index < len(self._alerts):
            self._alerts[index].acknowledged = True

    def _create_alert(
        self,
        severity: AlertSeverity,
        source_id: Optional[str],
        message: str,
        metric: str,
        value: float,
        threshold: float,
    ) -> Alert:
        alert = Alert(
            severity=severity,
            source_id=source_id,
            message=message,
            metric=metric,
            value=value,
            threshold=threshold,
        )
        logger.log(
            40 if severity == AlertSeverity.CRITICAL else 30,
            f"ALERT [{severity.value}]: {message}"
        )
        return alert
