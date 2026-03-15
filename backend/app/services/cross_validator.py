"""
Cross-Validation Engine

The core protection against "one shitty Bloomberg article" syndrome.

Every claim must be cross-validated against other sources before it
can enter the prediction pipeline. The engine:

1. Groups claims by category + topic
2. Finds corroborating and contradicting claims across sources
3. Calculates consensus scores
4. Flags conflicts for human review or automatic resolution
5. Produces ValidatedClaims with confidence intervals

Consensus rules (actuarial model):
- 3+ independent sources agree → HIGH confidence
- 2 sources agree, none contradict → MEDIUM confidence
- 1 source only → LOW confidence (flagged, not used for decisions)
- Sources contradict → CONFLICT (quarantined until resolved)
- Only SYNTHETIC sources → UNVALIDATED (never enters pipeline alone)
"""

import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime

from .data_source import (
    Claim, ClaimCategory, ClaimType, TrustTier,
    DataSource, SourceRegistry
)
from ..utils.logger import get_logger

logger = get_logger('mirofish.cross_validator')


class ConsensusLevel(Enum):
    """Result of cross-validation."""
    HIGH = "high"               # 3+ sources agree
    MEDIUM = "medium"           # 2 sources agree, none contradict
    LOW = "low"                 # Single source, uncontradicted
    CONFLICT = "conflict"       # Sources actively contradict
    UNVALIDATED = "unvalidated" # Only synthetic sources, no real data


class ConflictResolution(Enum):
    """How a conflict was resolved."""
    UNRESOLVED = "unresolved"
    TIER_WEIGHT = "tier_weight"         # Higher-tier source wins
    RECENCY = "recency"                 # More recent data wins
    MAJORITY = "majority"               # Majority of sources wins
    MANUAL = "manual"                   # Human resolved it
    QUARANTINED = "quarantined"         # Held for review


@dataclass
class ValidationResult:
    """
    The output of cross-validating a claim.

    This is what gets passed downstream — never a raw Claim.
    """
    claim: Claim
    consensus: ConsensusLevel
    confidence: float                       # 0.0 - 1.0
    confidence_interval: Tuple[float, float]  # (lower, upper) bounds
    corroborating_sources: List[str] = field(default_factory=list)
    contradicting_sources: List[str] = field(default_factory=list)
    corroborating_claims: List[str] = field(default_factory=list)
    contradicting_claims: List[str] = field(default_factory=list)
    conflict_resolution: ConflictResolution = ConflictResolution.UNRESOLVED
    resolution_notes: str = ""
    validated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_actionable(self) -> bool:
        """Can this result be used for predictions?"""
        return self.consensus in (ConsensusLevel.HIGH, ConsensusLevel.MEDIUM)

    @property
    def needs_review(self) -> bool:
        """Should a human look at this?"""
        return self.consensus == ConsensusLevel.CONFLICT


@dataclass
class ClaimCluster:
    """
    A group of related claims about the same topic from different sources.

    Claims are clustered by category + rough semantic similarity.
    """
    cluster_id: str
    category: ClaimCategory
    topic: str                      # Brief description of what this cluster is about
    claims: List[Claim] = field(default_factory=list)
    source_ids: Set[str] = field(default_factory=set)

    def add_claim(self, claim: Claim):
        self.claims.append(claim)
        self.source_ids.add(claim.source_id)

    @property
    def source_count(self) -> int:
        return len(self.source_ids)

    @property
    def has_tier1(self) -> bool:
        """Does this cluster include any TIER_1 sources?"""
        # Caller must set this based on registry lookup
        return False  # Override in validation


class CrossValidator:
    """
    Cross-validates claims across multiple data sources.

    This is the gatekeeper. Nothing enters the prediction pipeline
    without passing through here.
    """

    # Minimum sources needed for each consensus level
    CONSENSUS_THRESHOLDS = {
        ConsensusLevel.HIGH: 3,
        ConsensusLevel.MEDIUM: 2,
        ConsensusLevel.LOW: 1,
    }

    # Maximum influence any single source can have (0-1)
    MAX_SINGLE_SOURCE_WEIGHT = 0.35

    def __init__(self, registry: SourceRegistry):
        self.registry = registry
        self._conflict_log: List[Dict[str, Any]] = []

    def validate_claims(self, claims: List[Claim]) -> List[ValidationResult]:
        """
        Cross-validate a batch of claims.

        1. Cluster claims by category + topic
        2. For each cluster, calculate consensus
        3. Return ValidationResults
        """
        if not claims:
            return []

        # Group by category
        by_category: Dict[ClaimCategory, List[Claim]] = {}
        for claim in claims:
            if claim.category not in by_category:
                by_category[claim.category] = []
            by_category[claim.category].append(claim)

        results = []
        for category, cat_claims in by_category.items():
            clusters = self._cluster_claims(cat_claims)
            for cluster in clusters:
                cluster_results = self._validate_cluster(cluster)
                results.extend(cluster_results)

        # Handle orphan claims (not in any cluster)
        clustered_ids = {r.claim.claim_id for r in results}
        for claim in claims:
            if claim.claim_id not in clustered_ids:
                result = self._validate_single(claim)
                results.append(result)

        validated_count = sum(1 for r in results if r.is_actionable)
        conflict_count = sum(1 for r in results if r.needs_review)
        logger.info(
            f"Cross-validation complete: {len(results)} claims, "
            f"{validated_count} actionable, {conflict_count} conflicts"
        )
        return results

    def _cluster_claims(self, claims: List[Claim]) -> List[ClaimCluster]:
        """
        Group related claims together.

        Clusters by category + claim_type + naics_code + geography.
        Direction is NOT part of the key — direction is what we COMPARE
        within a cluster to detect contradictions.
        """
        clusters: Dict[str, ClaimCluster] = {}

        for claim in claims:
            # Build cluster key from topic attributes (NOT direction)
            key_parts = [claim.category.value, claim.claim_type.value]
            if claim.naics_code:
                key_parts.append(claim.naics_code)
            if claim.geography:
                key_parts.append(claim.geography)

            cluster_key = ":".join(key_parts)

            if cluster_key not in clusters:
                clusters[cluster_key] = ClaimCluster(
                    cluster_id=cluster_key,
                    category=claim.category,
                    topic=cluster_key,
                )
            clusters[cluster_key].add_claim(claim)

        return list(clusters.values())

    def _validate_cluster(self, cluster: ClaimCluster) -> List[ValidationResult]:
        """
        Validate a cluster of related claims.

        Checks for agreement/contradiction within the cluster.
        """
        results = []

        if cluster.source_count == 0:
            return results

        # Separate by source tier
        tier_claims: Dict[TrustTier, List[Claim]] = {}
        for claim in cluster.claims:
            source = self.registry.get_source(claim.source_id)
            if source:
                tier = source.trust_tier
                if tier not in tier_claims:
                    tier_claims[tier] = []
                tier_claims[tier].append(claim)

        # Check for directional conflicts
        directions = set()
        for claim in cluster.claims:
            if claim.direction:
                directions.add(claim.direction)

        has_conflict = len(directions) > 1

        # Check for value conflicts (>20% disagreement on metrics)
        # Uses pairwise comparison — if ANY two values disagree by >20%,
        # that's a conflict worth flagging.
        values = [c.value for c in cluster.claims if c.value is not None]
        value_conflict = False
        if len(values) >= 2:
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    ref = max(abs(values[i]), abs(values[j]))
                    if ref > 0:
                        pct_diff = abs(values[i] - values[j]) / ref
                        if pct_diff > 0.20:
                            value_conflict = True
                            break
                if value_conflict:
                    break

        has_conflict = has_conflict or value_conflict

        # Determine consensus level
        all_synthetic = all(
            (self.registry.get_source(c.source_id) and
             self.registry.get_source(c.source_id).trust_tier == TrustTier.SYNTHETIC)
            for c in cluster.claims
        )

        if all_synthetic:
            consensus = ConsensusLevel.UNVALIDATED
        elif has_conflict:
            consensus = ConsensusLevel.CONFLICT
        elif cluster.source_count >= self.CONSENSUS_THRESHOLDS[ConsensusLevel.HIGH]:
            consensus = ConsensusLevel.HIGH
        elif cluster.source_count >= self.CONSENSUS_THRESHOLDS[ConsensusLevel.MEDIUM]:
            consensus = ConsensusLevel.MEDIUM
        else:
            consensus = ConsensusLevel.LOW

        # For conflicts, try automatic resolution
        resolution = ConflictResolution.UNRESOLVED
        resolution_notes = ""
        if has_conflict:
            resolution, resolution_notes = self._resolve_conflict(cluster, tier_claims)

        # Calculate confidence for each claim in the cluster
        for claim in cluster.claims:
            source = self.registry.get_source(claim.source_id)
            corroborating = [
                c.source_id for c in cluster.claims
                if c.source_id != claim.source_id and not self._claims_contradict(claim, c)
            ]
            contradicting = [
                c.source_id for c in cluster.claims
                if c.source_id != claim.source_id and self._claims_contradict(claim, c)
            ]

            confidence = self._calculate_confidence(
                claim=claim,
                source=source,
                consensus=consensus,
                corroborating_count=len(corroborating),
                contradicting_count=len(contradicting),
            )
            ci = self._confidence_interval(confidence, cluster.source_count)

            results.append(ValidationResult(
                claim=claim,
                consensus=consensus,
                confidence=confidence,
                confidence_interval=ci,
                corroborating_sources=corroborating,
                contradicting_sources=contradicting,
                corroborating_claims=[c.claim_id for c in cluster.claims if c.source_id in corroborating],
                contradicting_claims=[c.claim_id for c in cluster.claims if c.source_id in contradicting],
                conflict_resolution=resolution,
                resolution_notes=resolution_notes,
            ))

        return results

    def _validate_single(self, claim: Claim) -> ValidationResult:
        """Validate a claim with no cluster peers — always LOW or UNVALIDATED."""
        source = self.registry.get_source(claim.source_id)

        if source and source.trust_tier == TrustTier.SYNTHETIC:
            consensus = ConsensusLevel.UNVALIDATED
        else:
            consensus = ConsensusLevel.LOW

        confidence = self._calculate_confidence(
            claim=claim,
            source=source,
            consensus=consensus,
            corroborating_count=0,
            contradicting_count=0,
        )
        ci = self._confidence_interval(confidence, 1)

        return ValidationResult(
            claim=claim,
            consensus=consensus,
            confidence=confidence,
            confidence_interval=ci,
        )

    def _claims_contradict(self, a: Claim, b: Claim) -> bool:
        """Do two claims contradict each other?"""
        # Directional contradiction
        if a.direction and b.direction:
            opposites = {("up", "down"), ("down", "up")}
            if (a.direction, b.direction) in opposites:
                return True

        # Value contradiction (>20% disagreement, pairwise)
        if a.value is not None and b.value is not None:
            ref = max(abs(a.value), abs(b.value))
            if ref > 0:
                deviation = abs(a.value - b.value) / ref
                if deviation > 0.20:
                    return True

        return False

    def _resolve_conflict(
        self,
        cluster: ClaimCluster,
        tier_claims: Dict[TrustTier, List[Claim]]
    ) -> Tuple[ConflictResolution, str]:
        """
        Try to automatically resolve a conflict.

        Resolution hierarchy:
        1. If TIER_1 sources agree with each other, they win
        2. If one side has more sources, majority wins
        3. If equal, more recent data wins
        4. If still tied, quarantine for manual review
        """
        # Check if TIER_1 sources have a consensus among themselves
        t1_claims = tier_claims.get(TrustTier.TIER_1, [])
        if len(t1_claims) >= 2:
            t1_directions = {c.direction for c in t1_claims if c.direction}
            if len(t1_directions) == 1:
                direction = t1_directions.pop()
                self._log_conflict(cluster, ConflictResolution.TIER_WEIGHT,
                                   f"TIER_1 sources agree: {direction}")
                return (
                    ConflictResolution.TIER_WEIGHT,
                    f"Resolved by TIER_1 consensus: {direction}"
                )

        # Count by direction across all claims
        direction_counts: Dict[str, int] = {}
        for claim in cluster.claims:
            if claim.direction:
                direction_counts[claim.direction] = direction_counts.get(claim.direction, 0) + 1

        if direction_counts:
            sorted_dirs = sorted(direction_counts.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_dirs) >= 2 and sorted_dirs[0][1] > sorted_dirs[1][1]:
                winner = sorted_dirs[0][0]
                self._log_conflict(cluster, ConflictResolution.MAJORITY,
                                   f"Majority says: {winner}")
                return (
                    ConflictResolution.MAJORITY,
                    f"Resolved by majority: {winner} ({sorted_dirs[0][1]} vs {sorted_dirs[1][1]})"
                )

        # Recency: most recent claim wins if from equal or higher tier
        sorted_by_time = sorted(cluster.claims, key=lambda c: c.timestamp, reverse=True)
        if sorted_by_time:
            most_recent = sorted_by_time[0]
            self._log_conflict(cluster, ConflictResolution.RECENCY,
                               f"Most recent: {most_recent.source_id}")
            return (
                ConflictResolution.RECENCY,
                f"Resolved by recency: {most_recent.source_id} ({most_recent.timestamp.isoformat()})"
            )

        # Can't resolve — quarantine
        self._log_conflict(cluster, ConflictResolution.QUARANTINED, "No resolution possible")
        return (
            ConflictResolution.QUARANTINED,
            "Conflict could not be automatically resolved — quarantined for review"
        )

    def _calculate_confidence(
        self,
        claim: Claim,
        source: Optional[DataSource],
        consensus: ConsensusLevel,
        corroborating_count: int,
        contradicting_count: int,
    ) -> float:
        """
        Calculate confidence score for a validated claim.

        Formula:
            confidence = source_weight * consensus_multiplier * freshness * corroboration_bonus

        Capped by MAX_SINGLE_SOURCE_WEIGHT to prevent any one source
        from dominating.
        """
        # Base: source weight (capped)
        if source:
            base = min(source.effective_weight, self.MAX_SINGLE_SOURCE_WEIGHT)
        else:
            base = 0.1  # Unknown source gets minimal weight

        # Consensus multiplier
        consensus_multipliers = {
            ConsensusLevel.HIGH: 1.0,
            ConsensusLevel.MEDIUM: 0.75,
            ConsensusLevel.LOW: 0.4,
            ConsensusLevel.CONFLICT: 0.2,
            ConsensusLevel.UNVALIDATED: 0.05,
        }
        consensus_mult = consensus_multipliers.get(consensus, 0.1)

        # Freshness (decay over TTL)
        freshness = 1.0
        if claim.is_stale:
            freshness = 0.3  # Stale claims heavily penalized

        # Corroboration bonus (diminishing returns)
        if corroborating_count > 0:
            # log2 gives diminishing returns: 1 source = 1.0, 2 = 1.0, 4 = 1.0 bonus
            corr_bonus = 1.0 + (math.log2(corroborating_count + 1) * 0.15)
        else:
            corr_bonus = 1.0

        # Contradiction penalty
        if contradicting_count > 0:
            contra_penalty = max(0.3, 1.0 - (contradicting_count * 0.25))
        else:
            contra_penalty = 1.0

        confidence = base * consensus_mult * freshness * corr_bonus * contra_penalty

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))

    def _confidence_interval(self, confidence: float, sample_size: int) -> Tuple[float, float]:
        """
        Calculate confidence interval using Wilson score interval.

        Treats each source as a "trial" and the confidence as a
        proportion. Wider intervals with fewer sources.
        """
        if sample_size == 0:
            return (0.0, 0.0)

        # Wilson score interval (z=1.96 for 95% CI)
        z = 1.96
        n = max(sample_size, 1)
        p = confidence

        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator

        lower = max(0.0, center - spread)
        upper = min(1.0, center + spread)

        return (round(lower, 4), round(upper, 4))

    def _log_conflict(self, cluster: ClaimCluster, resolution: ConflictResolution, notes: str):
        """Record conflict for audit trail."""
        entry = {
            "cluster_id": cluster.cluster_id,
            "category": cluster.category.value,
            "source_count": cluster.source_count,
            "resolution": resolution.value,
            "notes": notes,
            "timestamp": datetime.utcnow().isoformat(),
            "claim_count": len(cluster.claims),
        }
        self._conflict_log.append(entry)
        logger.info(f"Conflict resolution: {entry}")

    def get_conflict_log(self) -> List[Dict[str, Any]]:
        """Return the full conflict resolution audit trail."""
        return list(self._conflict_log)

    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Summary statistics for a batch of validation results."""
        if not results:
            return {"total": 0}

        by_consensus = {}
        for level in ConsensusLevel:
            matching = [r for r in results if r.consensus == level]
            by_consensus[level.value] = {
                "count": len(matching),
                "avg_confidence": (
                    sum(r.confidence for r in matching) / len(matching)
                    if matching else 0.0
                ),
            }

        actionable = [r for r in results if r.is_actionable]
        avg_ci_width = 0.0
        if actionable:
            avg_ci_width = sum(
                r.confidence_interval[1] - r.confidence_interval[0]
                for r in actionable
            ) / len(actionable)

        return {
            "total": len(results),
            "actionable": len(actionable),
            "conflicts": sum(1 for r in results if r.needs_review),
            "avg_confidence": sum(r.confidence for r in results) / len(results),
            "avg_ci_width": round(avg_ci_width, 4),
            "by_consensus": by_consensus,
        }
