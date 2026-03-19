"""
Actuarial Risk Assessment Engine

Provides portfolio-level risk scoring and prediction reliability grading
for the MiroFish prediction pipeline.

Sits between the data validation layer (cross_validator + validated_data_store)
and report generation. Takes validated claims and produces an ActuaryAssessment
that answers: "How trustworthy is this overall prediction?"

Grading follows actuarial credibility standards:
    A: Strong evidence — multiple corroborating sources, narrow CIs
    B: Good evidence — some gaps but generally reliable
    C: Mixed evidence — use with caution
    D: Weak evidence — significant uncertainty
    F: Insufficient data — do not rely on prediction
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .cross_validator import ValidationResult, ConsensusLevel
from .data_source import ClaimCategory
from ..utils.logger import get_logger

logger = get_logger('mirofish.actuary')


class RiskSeverity(Enum):
    """How serious a risk factor is."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskFactor:
    """A specific risk identified in the prediction data."""
    name: str
    severity: RiskSeverity
    description: str
    metric: Optional[float] = None  # Quantitative measure if applicable


@dataclass
class ActuaryAssessment:
    """
    Complete actuarial assessment of a prediction's reliability.

    This is the output of ActuaryEngine.assess() and gets embedded
    in reports and exposed via the API.
    """
    overall_reliability_grade: str       # A / B / C / D / F
    overall_confidence: float            # 0.0-1.0 weighted aggregate
    prediction_strength: str             # "Strong" / "Moderate" / "Weak" / "Insufficient"
    risk_factors: List[RiskFactor] = field(default_factory=list)
    probability_distribution: Dict[str, Any] = field(default_factory=dict)
    concentration_risk: Dict[str, Any] = field(default_factory=dict)
    data_sufficiency: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    assessed_at: str = ""
    claim_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_reliability_grade": self.overall_reliability_grade,
            "overall_confidence": round(self.overall_confidence, 3),
            "prediction_strength": self.prediction_strength,
            "risk_factors": [
                {
                    "name": rf.name,
                    "severity": rf.severity.value,
                    "description": rf.description,
                    "metric": rf.metric,
                }
                for rf in self.risk_factors
            ],
            "probability_distribution": self.probability_distribution,
            "concentration_risk": self.concentration_risk,
            "data_sufficiency": self.data_sufficiency,
            "recommendations": self.recommendations,
            "assessed_at": self.assessed_at,
            "claim_count": self.claim_count,
        }

    def to_markdown(self) -> str:
        """Render assessment as markdown for inclusion in reports."""
        lines = [
            "## Actuarial Risk Assessment",
            "",
            f"**Reliability Grade: {self.overall_reliability_grade}** | "
            f"Overall Confidence: {self.overall_confidence:.1%} | "
            f"Prediction Strength: {self.prediction_strength}",
            "",
        ]

        # Risk factors
        if self.risk_factors:
            lines.append("### Risk Factors")
            lines.append("")
            for rf in sorted(self.risk_factors, key=lambda r: _severity_order(r.severity)):
                icon = {"critical": "[!]", "high": "[!]", "medium": "[~]", "low": "[-]"}.get(
                    rf.severity.value, "[-]"
                )
                metric_str = f" ({rf.metric:.1%})" if rf.metric is not None else ""
                lines.append(f"- {icon} **{rf.severity.value.upper()}**: {rf.description}{metric_str}")
            lines.append("")

        # Distribution
        dist = self.probability_distribution
        if dist:
            lines.append("### Confidence Distribution")
            lines.append("")
            lines.append(f"- Shape: **{dist.get('shape', 'unknown')}**")
            lines.append(f"- Mean: {dist.get('mean', 0):.1%} | Median: {dist.get('median', 0):.1%}")
            lines.append(f"- Std Dev: {dist.get('std_dev', 0):.3f} | IQR: {dist.get('iqr', 0):.3f}")
            buckets = dist.get("buckets", {})
            if buckets:
                lines.append(f"- Distribution: {_render_buckets(buckets)}")
            lines.append("")

        # Data sufficiency
        ds = self.data_sufficiency
        if ds:
            lines.append("### Data Sufficiency")
            lines.append("")
            lines.append(f"- Claims: {ds.get('claim_count', 0)} (minimum: {ds.get('min_required', 5)})")
            lines.append(f"- Categories covered: {ds.get('categories_covered', 0)}/{ds.get('expected_categories', 0)}")
            lines.append(f"- Unique sources: {ds.get('unique_sources', 0)}")
            lines.append(f"- Fresh data ratio: {ds.get('freshness_ratio', 0):.0%}")
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("### Recommendations")
            lines.append("")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        return "\n".join(lines)


def _severity_order(severity: RiskSeverity) -> int:
    return {RiskSeverity.CRITICAL: 0, RiskSeverity.HIGH: 1,
            RiskSeverity.MEDIUM: 2, RiskSeverity.LOW: 3}.get(severity, 4)


def _render_buckets(buckets: Dict[str, int]) -> str:
    total = sum(buckets.values())
    if total == 0:
        return "no data"
    parts = []
    for label in ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]:
        count = buckets.get(label, 0)
        pct = count / total * 100
        parts.append(f"{label}: {count} ({pct:.0f}%)")
    return " | ".join(parts)


class ActuaryEngine:
    """
    Actuarial assessment engine for the prediction pipeline.

    Usage:
        engine = ActuaryEngine()
        assessment = engine.assess(validated_results)
    """

    # Grading thresholds
    GRADE_A_MIN_CONFIDENCE = 0.80
    GRADE_A_MIN_HIGH_CONSENSUS_RATIO = 0.70
    GRADE_B_MIN_CONFIDENCE = 0.65
    GRADE_B_MIN_GOOD_CONSENSUS_RATIO = 0.50
    GRADE_C_MIN_CONFIDENCE = 0.50
    GRADE_C_MIN_GOOD_CONSENSUS_RATIO = 0.30
    GRADE_D_MIN_CONFIDENCE = 0.35

    # Concentration thresholds
    SOURCE_CONCENTRATION_THRESHOLD = 0.40
    CATEGORY_CONCENTRATION_THRESHOLD = 0.50
    WEAK_CONSENSUS_THRESHOLD = 0.60

    # Data sufficiency
    MIN_CLAIMS = 5
    MIN_SOURCES = 3
    EXPECTED_CATEGORIES = len(ClaimCategory)

    def assess(self, results: List[ValidationResult]) -> ActuaryAssessment:
        """
        Produce a complete actuarial assessment from validated claims.

        Args:
            results: List of ValidationResult from the cross-validator

        Returns:
            ActuaryAssessment with grade, risks, distribution, and recommendations
        """
        assessment = ActuaryAssessment(
            assessed_at=datetime.utcnow().isoformat(),
            claim_count=len(results),
            overall_reliability_grade="F",
            overall_confidence=0.0,
            prediction_strength="Insufficient",
        )

        if not results:
            assessment.risk_factors.append(RiskFactor(
                name="no_data",
                severity=RiskSeverity.CRITICAL,
                description="No validated claims available for assessment",
            ))
            assessment.recommendations.append("Pipeline produced no validated data. Check data sources and API keys.")
            assessment.data_sufficiency = self._assess_data_sufficiency(results)
            assessment.probability_distribution = self._empty_distribution()
            assessment.concentration_risk = {"source": {}, "category": {}, "consensus": {}}
            return assessment

        # Compute all components
        overall_confidence = self._compute_overall_confidence(results)
        distribution = self._compute_distribution(results)
        concentration = self._assess_concentration(results)
        sufficiency = self._assess_data_sufficiency(results)
        risk_factors = self._identify_risk_factors(results, concentration, sufficiency, distribution)
        grade = self._assign_grade(overall_confidence, results)
        strength = self._grade_to_strength(grade)
        recommendations = self._generate_recommendations(grade, risk_factors, sufficiency)

        assessment.overall_confidence = overall_confidence
        assessment.overall_reliability_grade = grade
        assessment.prediction_strength = strength
        assessment.probability_distribution = distribution
        assessment.concentration_risk = concentration
        assessment.data_sufficiency = sufficiency
        assessment.risk_factors = risk_factors
        assessment.recommendations = recommendations

        logger.info(
            f"Actuary assessment: grade={grade}, confidence={overall_confidence:.3f}, "
            f"claims={len(results)}, risks={len(risk_factors)}"
        )

        return assessment

    # ── Core Computations ────────────────────────────────────────

    def _compute_overall_confidence(self, results: List[ValidationResult]) -> float:
        """
        Weighted aggregate confidence across all claims.

        Higher-consensus claims get more weight — a HIGH consensus claim
        matters more than a LOW one for the overall assessment.
        """
        if not results:
            return 0.0

        consensus_weights = {
            ConsensusLevel.HIGH: 1.0,
            ConsensusLevel.MEDIUM: 0.75,
            ConsensusLevel.LOW: 0.4,
            ConsensusLevel.CONFLICT: 0.15,
            ConsensusLevel.UNVALIDATED: 0.05,
        }

        weighted_sum = 0.0
        weight_total = 0.0

        for r in results:
            w = consensus_weights.get(r.consensus, 0.1)
            weighted_sum += r.confidence * w
            weight_total += w

        return weighted_sum / weight_total if weight_total > 0 else 0.0

    def _compute_distribution(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Analyze the probability distribution of claim confidences.

        Returns bucket counts, shape classification, and descriptive stats.
        """
        if not results:
            return self._empty_distribution()

        confidences = sorted([r.confidence for r in results])
        n = len(confidences)

        # Bucket into quintiles
        buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for c in confidences:
            if c < 0.2:
                buckets["0.0-0.2"] += 1
            elif c < 0.4:
                buckets["0.2-0.4"] += 1
            elif c < 0.6:
                buckets["0.4-0.6"] += 1
            elif c < 0.8:
                buckets["0.6-0.8"] += 1
            else:
                buckets["0.8-1.0"] += 1

        # Descriptive statistics
        mean = sum(confidences) / n
        median = confidences[n // 2] if n % 2 == 1 else (confidences[n // 2 - 1] + confidences[n // 2]) / 2
        variance = sum((c - mean) ** 2 for c in confidences) / n
        std_dev = math.sqrt(variance)

        # IQR
        q1_idx = n // 4
        q3_idx = (3 * n) // 4
        q1 = confidences[q1_idx]
        q3 = confidences[min(q3_idx, n - 1)]
        iqr = q3 - q1

        # Skewness (Fisher-Pearson)
        skewness = 0.0
        if std_dev > 0 and n >= 3:
            skewness = (n / ((n - 1) * (n - 2))) * sum(
                ((c - mean) / std_dev) ** 3 for c in confidences
            ) if n > 2 else 0.0

        # Shape classification
        shape = self._classify_shape(buckets, skewness, n)

        return {
            "buckets": buckets,
            "shape": shape,
            "mean": round(mean, 4),
            "median": round(median, 4),
            "std_dev": round(std_dev, 4),
            "iqr": round(iqr, 4),
            "skewness": round(skewness, 4),
            "min": round(confidences[0], 4),
            "max": round(confidences[-1], 4),
        }

    def _classify_shape(self, buckets: Dict[str, int], skewness: float, n: int) -> str:
        """Classify the distribution shape."""
        if n < 3:
            return "insufficient_data"

        # Check for bimodality: significant counts at both extremes
        low_count = buckets.get("0.0-0.2", 0) + buckets.get("0.2-0.4", 0)
        high_count = buckets.get("0.6-0.8", 0) + buckets.get("0.8-1.0", 0)
        mid_count = buckets.get("0.4-0.6", 0)

        if n >= 6 and low_count >= n * 0.25 and high_count >= n * 0.25 and mid_count < n * 0.2:
            return "bimodal"

        if skewness < -0.5:
            return "right-skewed"  # Mostly high confidence (negative skew = tail on left)
        elif skewness > 0.5:
            return "left-skewed"   # Mostly low confidence (positive skew = tail on right)
        else:
            return "symmetric"

    def _assess_concentration(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Detect over-reliance on single sources, categories, or consensus levels.
        """
        n = len(results)
        if n == 0:
            return {"source": {}, "category": {}, "consensus": {}}

        # Source concentration
        source_counts: Dict[str, int] = {}
        for r in results:
            sid = r.claim.source_id
            source_counts[sid] = source_counts.get(sid, 0) + 1

        source_concentration = {
            sid: {"count": count, "ratio": round(count / n, 3)}
            for sid, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
        }

        # Category concentration
        cat_counts: Dict[str, int] = {}
        for r in results:
            cat = r.claim.category.value
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        category_concentration = {
            cat: {"count": count, "ratio": round(count / n, 3)}
            for cat, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
        }

        # Consensus level distribution
        consensus_counts: Dict[str, int] = {}
        for r in results:
            level = r.consensus.value
            consensus_counts[level] = consensus_counts.get(level, 0) + 1

        consensus_concentration = {
            level: {"count": count, "ratio": round(count / n, 3)}
            for level, count in consensus_counts.items()
        }

        return {
            "source": source_concentration,
            "category": category_concentration,
            "consensus": consensus_concentration,
        }

    def _assess_data_sufficiency(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Score whether we have enough data to make reliable predictions."""
        n = len(results)
        categories = set(r.claim.category for r in results) if results else set()
        sources = set(r.claim.source_id for r in results) if results else set()
        fresh_count = sum(1 for r in results if not r.claim.is_stale) if results else 0

        return {
            "claim_count": n,
            "min_required": self.MIN_CLAIMS,
            "sufficient": n >= self.MIN_CLAIMS,
            "categories_covered": len(categories),
            "expected_categories": self.EXPECTED_CATEGORIES,
            "category_coverage_ratio": round(len(categories) / self.EXPECTED_CATEGORIES, 3) if self.EXPECTED_CATEGORIES > 0 else 0,
            "unique_sources": len(sources),
            "min_sources_expected": self.MIN_SOURCES,
            "source_diversity_sufficient": len(sources) >= self.MIN_SOURCES,
            "freshness_ratio": round(fresh_count / n, 3) if n > 0 else 0,
            "fresh_count": fresh_count,
            "stale_count": n - fresh_count,
        }

    # ── Risk Identification ────────────────────────────────────────

    def _identify_risk_factors(
        self,
        results: List[ValidationResult],
        concentration: Dict[str, Any],
        sufficiency: Dict[str, Any],
        distribution: Dict[str, Any],
    ) -> List[RiskFactor]:
        """Identify all risk factors from the data."""
        risks: List[RiskFactor] = []
        n = len(results)

        # 1. Source concentration risk
        for sid, info in concentration.get("source", {}).items():
            if info["ratio"] > self.SOURCE_CONCENTRATION_THRESHOLD:
                risks.append(RiskFactor(
                    name="source_concentration",
                    severity=RiskSeverity.HIGH,
                    description=f"Source '{sid}' contributes {info['ratio']:.0%} of claims — over-reliance risk",
                    metric=info["ratio"],
                ))

        # 2. Category concentration risk
        for cat, info in concentration.get("category", {}).items():
            if info["ratio"] > self.CATEGORY_CONCENTRATION_THRESHOLD:
                risks.append(RiskFactor(
                    name="category_concentration",
                    severity=RiskSeverity.MEDIUM,
                    description=f"Category '{cat}' has {info['ratio']:.0%} of claims — blind spots in other areas",
                    metric=info["ratio"],
                ))

        # 3. Weak consensus dominance
        consensus = concentration.get("consensus", {})
        weak_count = sum(
            consensus.get(level, {}).get("count", 0)
            for level in ["low", "conflict", "unvalidated"]
        )
        if n > 0 and weak_count / n > self.WEAK_CONSENSUS_THRESHOLD:
            risks.append(RiskFactor(
                name="weak_consensus_dominance",
                severity=RiskSeverity.HIGH,
                description=f"{weak_count}/{n} claims ({weak_count/n:.0%}) have LOW, CONFLICT, or UNVALIDATED consensus",
                metric=weak_count / n,
            ))

        # 4. Data insufficiency
        if not sufficiency.get("sufficient", False):
            risks.append(RiskFactor(
                name="insufficient_claims",
                severity=RiskSeverity.CRITICAL,
                description=f"Only {sufficiency['claim_count']} claims (minimum {sufficiency['min_required']})",
                metric=sufficiency["claim_count"],
            ))

        if not sufficiency.get("source_diversity_sufficient", False):
            risks.append(RiskFactor(
                name="insufficient_sources",
                severity=RiskSeverity.HIGH,
                description=f"Only {sufficiency['unique_sources']} unique sources (minimum {self.MIN_SOURCES})",
                metric=sufficiency["unique_sources"],
            ))

        # 5. Category coverage gap
        coverage = sufficiency.get("category_coverage_ratio", 0)
        if coverage < 0.3:
            risks.append(RiskFactor(
                name="low_category_coverage",
                severity=RiskSeverity.HIGH,
                description=f"Only {sufficiency['categories_covered']}/{sufficiency['expected_categories']} categories covered ({coverage:.0%})",
                metric=coverage,
            ))
        elif coverage < 0.5:
            risks.append(RiskFactor(
                name="moderate_category_coverage",
                severity=RiskSeverity.MEDIUM,
                description=f"Only {sufficiency['categories_covered']}/{sufficiency['expected_categories']} categories covered ({coverage:.0%})",
                metric=coverage,
            ))

        # 6. Staleness risk
        freshness = sufficiency.get("freshness_ratio", 1.0)
        if freshness < 0.5:
            risks.append(RiskFactor(
                name="stale_data",
                severity=RiskSeverity.HIGH,
                description=f"{sufficiency.get('stale_count', 0)} of {n} claims are stale ({1-freshness:.0%})",
                metric=1 - freshness,
            ))
        elif freshness < 0.8:
            risks.append(RiskFactor(
                name="aging_data",
                severity=RiskSeverity.LOW,
                description=f"{sufficiency.get('stale_count', 0)} of {n} claims are stale ({1-freshness:.0%})",
                metric=1 - freshness,
            ))

        # 7. Wide confidence intervals
        mean_ci_width = 0.0
        if results:
            ci_widths = [r.confidence_interval[1] - r.confidence_interval[0] for r in results]
            mean_ci_width = sum(ci_widths) / len(ci_widths)
        if mean_ci_width > 0.5:
            risks.append(RiskFactor(
                name="wide_confidence_intervals",
                severity=RiskSeverity.MEDIUM,
                description=f"Mean CI width is {mean_ci_width:.3f} — high uncertainty in individual claims",
                metric=mean_ci_width,
            ))

        # 8. High conflict ratio
        conflict_count = sum(1 for r in results if r.consensus == ConsensusLevel.CONFLICT)
        if n > 0 and conflict_count / n > 0.2:
            risks.append(RiskFactor(
                name="high_conflict_ratio",
                severity=RiskSeverity.HIGH,
                description=f"{conflict_count}/{n} claims ({conflict_count/n:.0%}) have conflicting sources",
                metric=conflict_count / n,
            ))

        # 9. Distribution shape warning
        shape = distribution.get("shape", "")
        if shape == "left-skewed":
            risks.append(RiskFactor(
                name="low_confidence_skew",
                severity=RiskSeverity.MEDIUM,
                description="Confidence distribution is left-skewed — most claims have low confidence",
            ))
        elif shape == "bimodal":
            risks.append(RiskFactor(
                name="bimodal_distribution",
                severity=RiskSeverity.MEDIUM,
                description="Bimodal confidence distribution — data is polarized between high and low confidence",
            ))

        return risks

    # ── Grading ────────────────────────────────────────

    def _assign_grade(self, overall_confidence: float, results: List[ValidationResult]) -> str:
        """
        Assign reliability grade based on confidence and consensus quality.

        Modeled after actuarial credibility standards:
        - A: Fully credible — strong multi-source evidence
        - B: Partially credible — good but not complete
        - C: Limited credibility — proceed with caution
        - D: Minimal credibility — high uncertainty
        - F: Not credible — insufficient data
        """
        n = len(results)
        if n == 0:
            return "F"

        high_ratio = sum(1 for r in results if r.consensus == ConsensusLevel.HIGH) / n
        good_ratio = sum(
            1 for r in results
            if r.consensus in (ConsensusLevel.HIGH, ConsensusLevel.MEDIUM)
        ) / n

        if overall_confidence >= self.GRADE_A_MIN_CONFIDENCE and high_ratio >= self.GRADE_A_MIN_HIGH_CONSENSUS_RATIO:
            return "A"
        if overall_confidence >= self.GRADE_B_MIN_CONFIDENCE and good_ratio >= self.GRADE_B_MIN_GOOD_CONSENSUS_RATIO:
            return "B"
        if overall_confidence >= self.GRADE_C_MIN_CONFIDENCE and good_ratio >= self.GRADE_C_MIN_GOOD_CONSENSUS_RATIO:
            return "C"
        if overall_confidence >= self.GRADE_D_MIN_CONFIDENCE:
            return "D"
        return "F"

    def _grade_to_strength(self, grade: str) -> str:
        return {
            "A": "Strong",
            "B": "Moderate",
            "C": "Weak",
            "D": "Weak",
            "F": "Insufficient",
        }.get(grade, "Unknown")

    # ── Recommendations ────────────────────────────────────────

    def _generate_recommendations(
        self,
        grade: str,
        risk_factors: List[RiskFactor],
        sufficiency: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable recommendations based on the assessment."""
        recs: List[str] = []

        critical_risks = [rf for rf in risk_factors if rf.severity == RiskSeverity.CRITICAL]
        high_risks = [rf for rf in risk_factors if rf.severity == RiskSeverity.HIGH]

        if grade == "F":
            recs.append("Prediction reliability is insufficient. Do not use this prediction for decision-making.")
        elif grade == "D":
            recs.append("Prediction has weak evidence. Treat all findings as directional indicators only.")
        elif grade == "C":
            recs.append("Mixed evidence quality. Cross-reference key findings with additional sources before acting.")

        if any(rf.name == "insufficient_claims" for rf in critical_risks):
            recs.append(f"Add more data sources to reach minimum {sufficiency.get('min_required', 5)} validated claims.")

        if any(rf.name == "insufficient_sources" for rf in high_risks):
            recs.append("Diversify data sources — current prediction relies on too few independent sources.")

        source_risks = [rf for rf in risk_factors if rf.name == "source_concentration"]
        if source_risks:
            recs.append("Reduce reliance on dominant data source to avoid single-point-of-failure bias.")

        if any(rf.name == "stale_data" for rf in high_risks):
            recs.append("Refresh stale data — a significant portion of claims may no longer reflect current conditions.")

        if any(rf.name == "high_conflict_ratio" for rf in high_risks):
            recs.append("Resolve data conflicts — contradicting sources reduce prediction reliability.")

        coverage_risks = [rf for rf in risk_factors if rf.name in ("low_category_coverage", "moderate_category_coverage")]
        if coverage_risks:
            recs.append("Expand data collection to cover more market categories for a more complete picture.")

        if not recs and grade in ("A", "B"):
            recs.append("Data quality is strong. Prediction is suitable for informing business decisions.")

        return recs

    # ── Utilities ────────────────────────────────────────

    def _empty_distribution(self) -> Dict[str, Any]:
        return {
            "buckets": {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0},
            "shape": "no_data",
            "mean": 0, "median": 0, "std_dev": 0, "iqr": 0,
            "skewness": 0, "min": 0, "max": 0,
        }
