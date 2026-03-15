"""
Data Source Abstraction Layer

Every piece of data entering the prediction pipeline must come through
a registered DataSource. Each source has a trust tier, health tracking,
and produces structured Claims (not raw text).

Trust Tiers (actuarial model):
    TIER_1 (0.9): Government/institutional data - FRED, SBA, Census, BLS
    TIER_2 (0.7): Industry data providers - DealStats, BizBuySell, IBBA
    TIER_3 (0.4): Qualitative/sentiment - news, social media, forums
    SYNTHETIC (0.1): LLM-generated content - never trusted alone

A single source can never flip a prediction. Claims require cross-validation
before entering the pipeline.
"""

import time
import hashlib
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..utils.logger import get_logger

logger = get_logger('mirofish.data_source')


class TrustTier(Enum):
    """Source trust levels. Values are base confidence weights."""
    TIER_1 = 0.9       # Government / institutional
    TIER_2 = 0.7       # Industry data providers
    TIER_3 = 0.4       # Qualitative / sentiment
    SYNTHETIC = 0.1    # LLM-generated


class ClaimCategory(Enum):
    """What domain does a claim belong to."""
    SBA_LENDING = "sba_lending"
    DEAL_MULTIPLES = "deal_multiples"
    SECTOR_HEALTH = "sector_health"
    BUYER_DEMAND = "buyer_demand"
    SELLER_SUPPLY = "seller_supply"
    INTEREST_RATES = "interest_rates"
    BUSINESS_VALUATIONS = "business_valuations"
    MARKET_SENTIMENT = "market_sentiment"
    ECONOMIC_INDICATORS = "economic_indicators"
    REGULATORY = "regulatory"


class ClaimType(Enum):
    """Nature of the claim."""
    METRIC = "metric"           # Quantitative: "SBA 7(a) approvals down 15%"
    TREND = "trend"             # Directional: "HVAC multiples are rising"
    EVENT = "event"             # Discrete: "SBA changed underwriting rules"
    SENTIMENT = "sentiment"     # Opinion: "Brokers are bearish on restaurants"
    FORECAST = "forecast"       # Prediction: "Deal volume will decline Q3"


@dataclass
class Claim:
    """
    A single, atomic, verifiable assertion from a data source.

    Every piece of information is decomposed into Claims before it can
    enter the pipeline. This is the fundamental unit of truth.
    """
    claim_id: str
    source_id: str
    category: ClaimCategory
    claim_type: ClaimType
    statement: str                          # Human-readable claim
    value: Optional[Any] = None             # Numeric value if METRIC
    unit: Optional[str] = None              # Unit of measurement
    direction: Optional[str] = None         # "up", "down", "flat" for TREND
    magnitude: Optional[float] = None       # Percentage change if applicable
    naics_code: Optional[str] = None        # Industry code if sector-specific
    geography: Optional[str] = None         # Geographic scope
    time_period: Optional[str] = None       # What time range this covers
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_data: Optional[Dict] = None         # Original data for audit trail
    source_url: Optional[str] = None        # Where this came from
    ttl_hours: float = 24.0                 # How long before this claim is stale

    @property
    def is_stale(self) -> bool:
        age = datetime.utcnow() - self.timestamp
        return age > timedelta(hours=self.ttl_hours)

    @property
    def fingerprint(self) -> str:
        """Content hash for deduplication."""
        content = f"{self.category.value}:{self.claim_type.value}:{self.statement}:{self.value}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class SourceHealth:
    """Tracks the reliability of a data source over time."""
    source_id: str
    total_claims: int = 0
    validated_claims: int = 0           # Claims that passed cross-validation
    contradicted_claims: int = 0        # Claims that were contradicted
    stale_claims: int = 0               # Claims that expired before validation
    fetch_successes: int = 0
    fetch_failures: int = 0
    last_fetch: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    is_degraded: bool = False           # CircuitBreaker has flagged this source

    @property
    def validation_rate(self) -> float:
        """What % of this source's claims survive cross-validation."""
        if self.total_claims == 0:
            return 0.0
        return self.validated_claims / self.total_claims

    @property
    def availability(self) -> float:
        """What % of fetches succeed."""
        total = self.fetch_successes + self.fetch_failures
        if total == 0:
            return 1.0
        return self.fetch_successes / total

    @property
    def effective_trust(self) -> float:
        """Dynamic trust = base tier trust * validation rate * availability."""
        # This gets multiplied by the tier's base weight elsewhere
        return self.validation_rate * self.availability

    def record_fetch_success(self):
        self.fetch_successes += 1
        self.last_fetch = datetime.utcnow()
        self.last_success = datetime.utcnow()
        self.consecutive_failures = 0

    def record_fetch_failure(self, error: str = ""):
        self.fetch_failures += 1
        self.last_fetch = datetime.utcnow()
        self.last_failure = datetime.utcnow()
        self.consecutive_failures += 1
        logger.warning(
            f"Source {self.source_id} fetch failure #{self.consecutive_failures}: {error}"
        )

    def record_claim_validated(self):
        self.validated_claims += 1

    def record_claim_contradicted(self):
        self.contradicted_claims += 1


class DataSource(ABC):
    """
    Abstract base for all data sources.

    Every source must:
    1. Declare its trust tier
    2. Produce structured Claims (not raw text)
    3. Track its own health
    4. Declare what ClaimCategories it can speak to
    """

    def __init__(self, source_id: str, trust_tier: TrustTier,
                 categories: List[ClaimCategory],
                 ttl_hours: float = 24.0):
        self.source_id = source_id
        self.trust_tier = trust_tier
        self.categories = categories
        self.ttl_hours = ttl_hours
        self.health = SourceHealth(source_id=source_id)
        self._claim_cache: Dict[str, Claim] = {}

    @abstractmethod
    def fetch_claims(self, category: Optional[ClaimCategory] = None,
                     **kwargs) -> List[Claim]:
        """
        Fetch and return structured claims from this source.

        Implementations must:
        - Call self._record_fetch_success() or self._record_fetch_failure()
        - Return Claim objects, not raw data
        - Set appropriate TTL on claims
        """
        pass

    @property
    def base_weight(self) -> float:
        """Base confidence weight from trust tier."""
        return self.trust_tier.value

    @property
    def effective_weight(self) -> float:
        """
        Dynamic weight combining tier + track record.

        A TIER_1 source that keeps getting contradicted will lose weight.
        A TIER_3 source that consistently validates will gain some weight.
        But TIER_3 can never exceed TIER_1's floor.
        """
        base = self.base_weight
        track_record = self.health.effective_trust
        if self.health.total_claims < 10:
            # Not enough history, use base weight with penalty
            return base * 0.8
        # Blend: 60% tier, 40% track record — tier always dominates
        blended = (base * 0.6) + (track_record * 0.4)
        # Floor: never drop below 50% of tier weight
        return max(blended, base * 0.5)

    def _record_fetch_success(self):
        self.health.record_fetch_success()

    def _record_fetch_failure(self, error: str = ""):
        self.health.record_fetch_failure(error)

    def _make_claim_id(self, content: str) -> str:
        ts = datetime.utcnow().isoformat()
        raw = f"{self.source_id}:{content}:{ts}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class SourceRegistry:
    """
    Central registry of all data sources.

    Sources register here. The CrossValidator queries this registry
    to find corroborating/contradicting sources for any given claim.
    """

    def __init__(self):
        self._sources: Dict[str, DataSource] = {}
        self._category_index: Dict[ClaimCategory, List[str]] = {}

    def register(self, source: DataSource):
        """Register a data source."""
        if source.source_id in self._sources:
            logger.warning(f"Re-registering source: {source.source_id}")
        self._sources[source.source_id] = source
        for cat in source.categories:
            if cat not in self._category_index:
                self._category_index[cat] = []
            if source.source_id not in self._category_index[cat]:
                self._category_index[cat].append(source.source_id)
        logger.info(
            f"Registered source: {source.source_id} "
            f"(tier={source.trust_tier.name}, categories={[c.value for c in source.categories]})"
        )

    def get_source(self, source_id: str) -> Optional[DataSource]:
        return self._sources.get(source_id)

    def get_sources_for_category(self, category: ClaimCategory) -> List[DataSource]:
        """Get all sources that can speak to a given claim category."""
        source_ids = self._category_index.get(category, [])
        return [self._sources[sid] for sid in source_ids if sid in self._sources]

    def get_sources_by_tier(self, tier: TrustTier) -> List[DataSource]:
        return [s for s in self._sources.values() if s.trust_tier == tier]

    def get_all_sources(self) -> List[DataSource]:
        return list(self._sources.values())

    def get_healthy_sources(self) -> List[DataSource]:
        """Sources not flagged as degraded by the circuit breaker."""
        return [s for s in self._sources.values() if not s.health.is_degraded]

    @property
    def source_count(self) -> int:
        return len(self._sources)

    @property
    def category_coverage(self) -> Dict[str, int]:
        """How many sources cover each category."""
        return {
            cat.value: len(sids)
            for cat, sids in self._category_index.items()
        }

    def health_report(self) -> Dict[str, Any]:
        """Full health status of all registered sources."""
        return {
            "total_sources": self.source_count,
            "healthy_sources": len(self.get_healthy_sources()),
            "category_coverage": self.category_coverage,
            "sources": {
                sid: {
                    "tier": s.trust_tier.name,
                    "effective_weight": round(s.effective_weight, 3),
                    "availability": round(s.health.availability, 3),
                    "validation_rate": round(s.health.validation_rate, 3),
                    "is_degraded": s.health.is_degraded,
                    "consecutive_failures": s.health.consecutive_failures,
                    "total_claims": s.health.total_claims,
                }
                for sid, s in self._sources.items()
            }
        }
