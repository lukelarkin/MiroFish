"""
Data Pipeline Orchestrator

Wires together all validation layer components:
    DataSources → ClaimExtractor → CrossValidator → CircuitBreaker → ValidatedDataStore

This is the single entry point for the BrokerTrendService to get
validated, confidence-scored data. The pipeline:

1. Fetches from all healthy registered sources
2. Extracts claims
3. Cross-validates
4. Runs circuit breaker checks
5. Stores validated data
6. Checks the gate
7. Returns validated seed data for the simulation pipeline

The pipeline REFUSES to produce output if:
- Too few sources are healthy
- Confidence intervals are too wide
- Not enough categories are covered
"""

import os
from typing import Dict, List, Any, Optional

from .data_source import (
    DataSource, SourceRegistry, TrustTier,
    Claim, ClaimCategory, ClaimType,
)
from .claim_extractor import ClaimExtractor
from .cross_validator import CrossValidator, ValidationResult
from .circuit_breaker import CircuitBreaker
from .validated_data_store import ValidatedDataStore
from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.data_pipeline')


class DataPipeline:
    """
    Orchestrates the full data validation pipeline.

    Usage:
        pipeline = DataPipeline()
        pipeline.register_source(my_sba_source)
        pipeline.register_source(my_bizbuysell_source)

        result = pipeline.run()
        if result["gate_open"]:
            seed_text = result["seed_text"]
            # Feed into simulation pipeline
    """

    def __init__(self, store_dir: Optional[str] = None):
        self.registry = SourceRegistry()
        self.extractor = ClaimExtractor()
        self.validator = CrossValidator(self.registry)
        self.breaker = CircuitBreaker(self.registry)

        if store_dir is None:
            store_dir = os.path.join(Config.UPLOAD_FOLDER, 'validated_data')

        self.store = ValidatedDataStore(
            registry=self.registry,
            cross_validator=self.validator,
            circuit_breaker=self.breaker,
            store_dir=store_dir,
        )

    def register_source(self, source: DataSource):
        """Register a data source into the pipeline."""
        self.registry.register(source)

    def run(self, categories: Optional[List[ClaimCategory]] = None) -> Dict[str, Any]:
        """
        Execute the full pipeline: fetch → extract → validate → store → gate check.

        Args:
            categories: Optional filter — only fetch these categories.
                        If None, fetches all categories from all sources.

        Returns:
            Dict with:
                gate_open: bool — is there enough validated data?
                gate_reason: str — why the gate is open/closed
                seed_text: str — validated seed text (empty if gate closed)
                export: dict — structured export for simulation
                health: dict — pipeline health report
                stats: dict — validation statistics
        """
        self.store.clear()
        self.extractor.clear_seen()

        all_claims: List[Claim] = []

        # Step 1: Fetch from all healthy sources
        sources = self.registry.get_healthy_sources()
        if not sources:
            logger.warning("No healthy sources available")
            return self._empty_result("No healthy sources available")

        for source in sources:
            # Circuit breaker gate
            if not self.breaker.should_allow_source(source.source_id):
                logger.info(f"Skipping source {source.source_id} (circuit open)")
                continue

            try:
                if categories:
                    # Only fetch categories this source can speak to
                    source_cats = [c for c in categories if c in source.categories]
                    for cat in source_cats:
                        claims = source.fetch_claims(category=cat)
                        all_claims.extend(claims)
                else:
                    claims = source.fetch_claims()
                    all_claims.extend(claims)

                # Record success
                self.breaker.record_source_success(source.source_id)

                # Check for volume spikes
                volume_alerts = self.breaker.check_claim_volume(
                    source.source_id, len(claims)
                )

            except Exception as e:
                logger.error(f"Source {source.source_id} fetch failed: {e}")
                self.breaker.record_source_failure(source.source_id)
                source._record_fetch_failure(str(e))
                continue

        if not all_claims:
            logger.warning("No claims fetched from any source")
            return self._empty_result("No claims fetched from any source")

        logger.info(f"Fetched {len(all_claims)} claims from {len(sources)} sources")

        # Step 2: Cross-validate
        validation_results = self.validator.validate_claims(all_claims)
        stats = self.validator.get_validation_summary(validation_results)

        # Step 3: Update source health based on validation
        for result in validation_results:
            source = self.registry.get_source(result.claim.source_id)
            if source:
                source.health.total_claims += 1
                if result.is_actionable:
                    source.health.record_claim_validated()
                elif result.needs_review:
                    source.health.record_claim_contradicted()

        # Step 4: Ingest into store
        ingestion = self.store.ingest(validation_results)

        # Step 5: Check gate
        gate_open, gate_reason = self.store.check_gate()

        # Step 6: Export
        seed_text = ""
        export = {}
        if gate_open:
            seed_text = self.store.export_as_seed_text()
            export = self.store.export_for_simulation()

        return {
            "gate_open": gate_open,
            "gate_reason": gate_reason,
            "seed_text": seed_text,
            "export": export,
            "health": self.store.health_report(),
            "stats": stats,
            "ingestion": ingestion,
        }

    def run_health_check(self) -> Dict[str, Any]:
        """Run health checks without fetching new data."""
        return self.breaker.run_health_cycle()

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "gate_open": False,
            "gate_reason": reason,
            "seed_text": "",
            "export": {},
            "health": self.store.health_report(),
            "stats": {"total": 0},
            "ingestion": {},
        }


# ── Example Source Implementations ──────────────────────────────
# These are stubs showing how to implement real sources.
# Each one would make actual API calls when data feeds are configured.

class LLMSyntheticSource(DataSource):
    """
    The existing LLM-generated content, properly labeled as SYNTHETIC.

    This source exists so the current pipeline still works, but its
    claims will always be UNVALIDATED unless corroborated by real sources.
    """

    def __init__(self, llm_client):
        super().__init__(
            source_id="llm_synthetic",
            trust_tier=TrustTier.SYNTHETIC,
            categories=list(ClaimCategory),  # Claims to know everything
            ttl_hours=1.0,  # Very short TTL — synthetic data decays fast
        )
        self.llm_client = llm_client
        self._extractor = ClaimExtractor()

    def fetch_claims(self, category=None, **kwargs):
        """
        Generate synthetic claims via LLM.

        The LLM decomposes its own output into structured claims.
        """
        prompt = kwargs.get("prompt", self._default_prompt(category))

        try:
            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4096,
            )
            self._record_fetch_success()

            # Parse LLM response into claim dicts
            import json
            try:
                claims_data = json.loads(response)
            except json.JSONDecodeError:
                # LLM didn't return JSON — wrap as single sentiment claim
                claims_data = [{
                    "statement": response[:500],
                    "claim_type": "sentiment",
                    "category": category.value if category else "market_sentiment",
                }]

            claims = self._extractor.extract_from_llm_decomposition(
                source_id=self.source_id,
                llm_claims=claims_data if isinstance(claims_data, list) else [claims_data],
                base_category=category or ClaimCategory.MARKET_SENTIMENT,
                ttl_hours=self.ttl_hours,
            )
            return claims

        except Exception as e:
            self._record_fetch_failure(str(e))
            raise

    def _default_prompt(self, category=None):
        cat_context = f" Focus on {category.value}." if category else ""
        return (
            "You are a business brokerage market analyst. "
            "Return a JSON array of market claims, each with fields: "
            "statement (str), claim_type (metric|trend|event|sentiment|forecast), "
            "value (number or null), direction (up|down|flat or null), "
            "magnitude (percentage or null), naics_code (str or null), "
            "category (sba_lending|deal_multiples|sector_health|buyer_demand|"
            "seller_supply|interest_rates|business_valuations|market_sentiment|"
            "economic_indicators|regulatory)."
            f"{cat_context} "
            "Return 10-15 claims covering the current US small business "
            "acquisition market. Return ONLY valid JSON, no markdown."
        )


class SBADataSource(DataSource):
    """
    Stub for SBA.gov lending data.

    When implemented, fetches:
    - 7(a) loan approval volumes by NAICS code
    - Average loan amounts
    - Default rates
    - Geographic distribution

    TIER_1 trust — this is government data.
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            source_id="sba_gov",
            trust_tier=TrustTier.TIER_1,
            categories=[
                ClaimCategory.SBA_LENDING,
                ClaimCategory.SECTOR_HEALTH,
                ClaimCategory.ECONOMIC_INDICATORS,
            ],
            ttl_hours=168.0,  # Weekly data
        )
        self.api_key = api_key
        self._extractor = ClaimExtractor()

    def fetch_claims(self, category=None, **kwargs):
        # TODO: Implement actual SBA API calls
        # For now, returns empty — pipeline will work with whatever
        # sources ARE configured
        logger.info("SBA data source not yet configured — returning no claims")
        self._record_fetch_success()
        return []


class FREDDataSource(DataSource):
    """
    Stub for FRED (Federal Reserve Economic Data).

    When implemented, fetches:
    - Prime rate (drives SBA loan rates)
    - GDP growth
    - Unemployment by state
    - Consumer spending indices

    TIER_1 trust — Federal Reserve data.
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            source_id="fred",
            trust_tier=TrustTier.TIER_1,
            categories=[
                ClaimCategory.INTEREST_RATES,
                ClaimCategory.ECONOMIC_INDICATORS,
            ],
            ttl_hours=24.0,
        )
        self.api_key = api_key
        self._extractor = ClaimExtractor()

    def fetch_claims(self, category=None, **kwargs):
        # TODO: Implement actual FRED API calls
        logger.info("FRED data source not yet configured — returning no claims")
        self._record_fetch_success()
        return []


class BizBuySellSource(DataSource):
    """
    Stub for BizBuySell / BizQuest listing data.

    When implemented, fetches:
    - Listing counts by NAICS code
    - Asking price multiples (revenue, SDE, EBITDA)
    - Time on market
    - Geographic distribution

    TIER_2 trust — industry data provider.
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            source_id="bizbuysell",
            trust_tier=TrustTier.TIER_2,
            categories=[
                ClaimCategory.DEAL_MULTIPLES,
                ClaimCategory.SELLER_SUPPLY,
                ClaimCategory.BUSINESS_VALUATIONS,
                ClaimCategory.SECTOR_HEALTH,
            ],
            ttl_hours=48.0,
        )
        self.api_key = api_key
        self._extractor = ClaimExtractor()

    def fetch_claims(self, category=None, **kwargs):
        # TODO: Implement actual BizBuySell data integration
        logger.info("BizBuySell data source not yet configured — returning no claims")
        self._record_fetch_success()
        return []


class IBBAMarketPulseSource(DataSource):
    """
    Stub for IBBA Market Pulse Survey data.

    When implemented, fetches:
    - Broker sentiment on deal activity
    - Buyer/seller ratios
    - Industry sector trends
    - Deal size distribution

    TIER_2 trust — industry association data.
    """

    def __init__(self):
        super().__init__(
            source_id="ibba_market_pulse",
            trust_tier=TrustTier.TIER_2,
            categories=[
                ClaimCategory.BUYER_DEMAND,
                ClaimCategory.SELLER_SUPPLY,
                ClaimCategory.MARKET_SENTIMENT,
                ClaimCategory.DEAL_MULTIPLES,
            ],
            ttl_hours=720.0,  # Quarterly survey
        )
        self._extractor = ClaimExtractor()

    def fetch_claims(self, category=None, **kwargs):
        # TODO: Implement actual IBBA data integration
        logger.info("IBBA Market Pulse source not yet configured — returning no claims")
        self._record_fetch_success()
        return []
