"""
Validated Data Store

The final gate before data enters the prediction pipeline.

Only ValidationResults that pass all checks flow through here.
This is the "clean room" — everything downstream can trust that
data in this store has been:

1. Sourced from a registered DataSource
2. Decomposed into atomic Claims
3. Cross-validated against multiple sources
4. Scored with confidence intervals
5. Checked by the CircuitBreaker

The store also provides the interface for the simulation/report
layers to query validated intelligence by category, confidence
level, or time range.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .data_source import ClaimCategory, ClaimType, SourceRegistry
from .cross_validator import ValidationResult, ConsensusLevel, CrossValidator
from .circuit_breaker import CircuitBreaker, AlertSeverity
from ..utils.logger import get_logger

logger = get_logger('mirofish.validated_store')


@dataclass
class PipelineGate:
    """
    Controls whether the pipeline should proceed.

    The store won't release data if the pipeline is unhealthy.
    """
    is_open: bool = True
    reason: str = ""
    min_actionable_claims: int = 5      # Won't release with fewer
    min_categories_covered: int = 2     # Need at least 2 categories
    max_avg_ci_width: float = 0.6       # CI too wide = too uncertain

    def evaluate(
        self,
        actionable_count: int,
        categories_covered: int,
        avg_ci_width: float,
        pipeline_viable: bool,
    ) -> Tuple[bool, str]:
        """Check if we have enough validated data to proceed."""
        if not pipeline_viable:
            return False, "Pipeline not viable — too many sources degraded"

        if actionable_count < self.min_actionable_claims:
            return False, (
                f"Only {actionable_count} actionable claims "
                f"(need {self.min_actionable_claims})"
            )

        if categories_covered < self.min_categories_covered:
            return False, (
                f"Only {categories_covered} categories covered "
                f"(need {self.min_categories_covered})"
            )

        if avg_ci_width > self.max_avg_ci_width:
            return False, (
                f"Average confidence interval width {avg_ci_width:.2f} "
                f"exceeds maximum {self.max_avg_ci_width}"
            )

        return True, "Pipeline gate OPEN — sufficient validated data"


class ValidatedDataStore:
    """
    The clean data gate.

    Accepts ValidationResults, stores them indexed by category,
    and provides query interfaces for downstream consumers.
    """

    def __init__(
        self,
        registry: SourceRegistry,
        cross_validator: CrossValidator,
        circuit_breaker: CircuitBreaker,
        store_dir: Optional[str] = None,
    ):
        self.registry = registry
        self.validator = cross_validator
        self.breaker = circuit_breaker
        self.gate = PipelineGate()
        self.store_dir = store_dir

        # In-memory store, indexed by category
        self._results: Dict[ClaimCategory, List[ValidationResult]] = {}
        self._all_results: List[ValidationResult] = []
        self._ingestion_log: List[Dict[str, Any]] = []

        if store_dir:
            os.makedirs(store_dir, exist_ok=True)

    def ingest(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Accept validated results into the store.

        Filters out non-actionable results and runs circuit breaker
        checks before storing.
        """
        # Run circuit breaker checks
        cb_alerts = self.breaker.check_source_concentration(results)

        # Filter: only store actionable results (HIGH or MEDIUM consensus)
        actionable = [r for r in results if r.is_actionable]
        conflicts = [r for r in results if r.needs_review]
        low_conf = [r for r in results if r.consensus == ConsensusLevel.LOW]
        unvalidated = [r for r in results if r.consensus == ConsensusLevel.UNVALIDATED]

        # Store actionable results
        for result in actionable:
            cat = result.claim.category
            if cat not in self._results:
                self._results[cat] = []
            self._results[cat].append(result)
            self._all_results.append(result)

        # Log ingestion
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_received": len(results),
            "actionable_stored": len(actionable),
            "conflicts_flagged": len(conflicts),
            "low_confidence_dropped": len(low_conf),
            "unvalidated_dropped": len(unvalidated),
            "alerts": len(cb_alerts),
        }
        self._ingestion_log.append(log_entry)

        logger.info(
            f"Ingested {len(actionable)}/{len(results)} claims into store "
            f"({len(conflicts)} conflicts, {len(low_conf)} low-conf dropped)"
        )

        return log_entry

    def check_gate(self) -> Tuple[bool, str]:
        """
        Should the pipeline proceed with current data?

        Returns (is_open, reason).
        """
        actionable_count = len(self._all_results)
        categories_covered = len(self._results)

        # Calculate average CI width
        avg_ci_width = 0.0
        if self._all_results:
            widths = [
                r.confidence_interval[1] - r.confidence_interval[0]
                for r in self._all_results
            ]
            avg_ci_width = sum(widths) / len(widths)

        # Check circuit breaker pipeline viability
        health = self.breaker.run_health_cycle()
        pipeline_viable = health.get("pipeline_viable", False)

        is_open, reason = self.gate.evaluate(
            actionable_count=actionable_count,
            categories_covered=categories_covered,
            avg_ci_width=avg_ci_width,
            pipeline_viable=pipeline_viable,
        )

        logger.info(f"Pipeline gate: {'OPEN' if is_open else 'CLOSED'} — {reason}")
        return is_open, reason

    # ── Query Interface ────────────────────────────────────────

    def get_by_category(
        self,
        category: ClaimCategory,
        min_confidence: float = 0.0,
    ) -> List[ValidationResult]:
        """Get validated results for a category, optionally filtered by confidence."""
        results = self._results.get(category, [])
        if min_confidence > 0:
            results = [r for r in results if r.confidence >= min_confidence]
        return sorted(results, key=lambda r: r.confidence, reverse=True)

    def get_by_naics(self, naics_code: str) -> List[ValidationResult]:
        """Get all validated results for a specific industry (NAICS code)."""
        return [
            r for r in self._all_results
            if r.claim.naics_code == naics_code
        ]

    def get_high_confidence(self, min_confidence: float = 0.7) -> List[ValidationResult]:
        """Get only high-confidence validated results."""
        return [
            r for r in self._all_results
            if r.confidence >= min_confidence
        ]

    def get_fresh(self, max_age_hours: float = 24.0) -> List[ValidationResult]:
        """Get results that aren't stale."""
        return [
            r for r in self._all_results
            if not r.claim.is_stale
        ]

    def get_all(self) -> List[ValidationResult]:
        """Get all stored results."""
        return list(self._all_results)

    # ── Export for Pipeline ────────────────────────────────────

    def export_for_simulation(self) -> Dict[str, Any]:
        """
        Package validated data for the simulation layer.

        Returns structured data organized by category with
        confidence metadata attached.
        """
        is_open, reason = self.check_gate()

        export = {
            "gate_status": "open" if is_open else "closed",
            "gate_reason": reason,
            "exported_at": datetime.utcnow().isoformat(),
            "total_claims": len(self._all_results),
            "categories": {},
            "confidence_summary": self._confidence_summary(),
        }

        if not is_open:
            logger.warning(f"Export blocked by pipeline gate: {reason}")
            return export

        for cat, results in self._results.items():
            export["categories"][cat.value] = {
                "claim_count": len(results),
                "avg_confidence": (
                    sum(r.confidence for r in results) / len(results)
                    if results else 0.0
                ),
                "claims": [
                    {
                        "statement": r.claim.statement,
                        "value": r.claim.value,
                        "direction": r.claim.direction,
                        "magnitude": r.claim.magnitude,
                        "confidence": round(r.confidence, 3),
                        "ci_lower": r.confidence_interval[0],
                        "ci_upper": r.confidence_interval[1],
                        "consensus": r.consensus.value,
                        "source_count": len(r.corroborating_sources) + 1,
                        "naics_code": r.claim.naics_code,
                        "time_period": r.claim.time_period,
                        "claim_type": r.claim.claim_type.value,
                    }
                    for r in sorted(results, key=lambda r: r.confidence, reverse=True)
                ]
            }

        return export

    def export_as_seed_text(self) -> str:
        """
        Convert validated data into seed text for the existing
        MiroFish pipeline.

        This bridges the new validation layer with the existing
        ontology → graph → simulation pipeline.
        """
        is_open, reason = self.check_gate()
        if not is_open:
            logger.warning(f"Seed text export blocked: {reason}")
            return ""

        sections = []
        sections.append(
            "# Validated Market Intelligence Report\n"
            f"Generated: {datetime.utcnow().isoformat()}\n"
            f"Total validated claims: {len(self._all_results)}\n"
        )

        for cat, results in self._results.items():
            if not results:
                continue

            section_title = cat.value.replace("_", " ").title()
            sections.append(f"\n## {section_title}\n")

            # Sort by confidence
            sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)

            for r in sorted_results:
                conf_pct = f"{r.confidence:.0%}"
                ci = f"[{r.confidence_interval[0]:.0%}-{r.confidence_interval[1]:.0%}]"
                source_count = len(r.corroborating_sources) + 1
                consensus = r.consensus.value.upper()

                line = (
                    f"- **[{consensus} | {conf_pct} confidence {ci} | "
                    f"{source_count} sources]** {r.claim.statement}"
                )
                if r.claim.value is not None:
                    line += f" (Value: {r.claim.value}"
                    if r.claim.unit:
                        line += f" {r.claim.unit}"
                    line += ")"
                if r.claim.naics_code:
                    line += f" [NAICS: {r.claim.naics_code}]"

                sections.append(line)

        return "\n".join(sections)

    # ── Diagnostics ────────────────────────────────────────────

    def _confidence_summary(self) -> Dict[str, Any]:
        """Summary stats on confidence distribution."""
        if not self._all_results:
            return {"count": 0}

        confidences = [r.confidence for r in self._all_results]
        ci_widths = [
            r.confidence_interval[1] - r.confidence_interval[0]
            for r in self._all_results
        ]

        return {
            "count": len(confidences),
            "mean_confidence": round(sum(confidences) / len(confidences), 3),
            "min_confidence": round(min(confidences), 3),
            "max_confidence": round(max(confidences), 3),
            "mean_ci_width": round(sum(ci_widths) / len(ci_widths), 3),
            "high_confidence_count": sum(1 for c in confidences if c >= 0.7),
            "medium_confidence_count": sum(1 for c in confidences if 0.4 <= c < 0.7),
            "low_confidence_count": sum(1 for c in confidences if c < 0.4),
        }

    def health_report(self) -> Dict[str, Any]:
        """Full diagnostic report on the validated data store."""
        is_open, reason = self.check_gate()
        return {
            "gate_status": "open" if is_open else "closed",
            "gate_reason": reason,
            "total_stored": len(self._all_results),
            "categories_covered": len(self._results),
            "category_breakdown": {
                cat.value: len(results)
                for cat, results in self._results.items()
            },
            "confidence_summary": self._confidence_summary(),
            "ingestion_log": self._ingestion_log[-10:],  # Last 10 ingestions
            "source_health": self.registry.health_report(),
            "circuit_breaker": self.breaker.run_health_cycle(),
        }

    def clear(self):
        """Reset the store. Use between prediction cycles."""
        self._results.clear()
        self._all_results.clear()
        logger.info("Validated data store cleared")
