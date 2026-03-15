"""
Data Pipeline Orchestrator

Wires together all validation layer components:
    DataSources -> ClaimExtractor -> CrossValidator -> CircuitBreaker -> ValidatedDataStore

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
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

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

# HTTP timeout for all external API calls (seconds)
API_TIMEOUT = 30


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
        Execute the full pipeline: fetch -> extract -> validate -> store -> gate check.
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
                self.breaker.check_claim_volume(
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


# ── Source Implementations ──────────────────────────────────────


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
            ttl_hours=1.0,  # Very short TTL
        )
        self.llm_client = llm_client
        self._extractor = ClaimExtractor()

    def fetch_claims(self, category=None, **kwargs):
        prompt = kwargs.get("prompt", self._default_prompt(category))

        try:
            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4096,
            )
            self._record_fetch_success()

            try:
                claims_data = json.loads(response)
            except json.JSONDecodeError:
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


# ── NAICS code mapping for business brokerage sectors ──────────
# Maps human-readable sector names to NAICS codes
SECTOR_NAICS = {
    "hvac": "238220",
    "plumbing": "238220",
    "electrical": "238210",
    "pest_control": "561710",
    "landscaping": "561730",
    "auto_repair": "811111",
    "restaurants": "722511",
    "dental": "621210",
    "veterinary": "541940",
    "home_health": "621610",
    "it_services": "541512",
    "manufacturing": "31-33",
    "professional_services": "541",
    "construction": "23",
    "retail": "44-45",
    "accommodation": "721",
}


class SBADataSource(DataSource):
    """
    SBA.gov 7(a) loan data via CKAN API.

    Fetches real SBA lending data:
    - Loan approval volumes by NAICS code
    - Average loan amounts
    - Geographic distribution
    - Approval counts and trends

    TIER_1 trust — this is US government data.
    No API key required (public CKAN data portal).
    """

    # CKAN API base
    BASE_URL = "https://data.sba.gov/en/api/3/action"

    # Known resource IDs for SBA 7(a) datasets
    # These are the FOIA 7(a) loan data resources on data.sba.gov
    RESOURCE_IDS = {
        "7a_2020s": "41625060-a76b-4608-ad63-d969f0a4bc6f",
    }

    # Key NAICS prefixes for business brokerage sectors
    TARGET_NAICS = {
        "238": "Specialty Trade Contractors (HVAC, Plumbing, Electrical)",
        "561": "Administrative & Support Services (Pest Control, Landscaping)",
        "811": "Repair & Maintenance (Auto Repair)",
        "722": "Food Services & Drinking Places (Restaurants)",
        "621": "Ambulatory Health Care (Dental, Home Health)",
        "541": "Professional, Scientific & Technical Services",
    }

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            source_id="sba_gov",
            trust_tier=TrustTier.TIER_1,
            categories=[
                ClaimCategory.SBA_LENDING,
                ClaimCategory.SECTOR_HEALTH,
                ClaimCategory.ECONOMIC_INDICATORS,
            ],
            ttl_hours=168.0,  # Weekly refresh is sufficient for SBA data
        )
        self._extractor = ClaimExtractor()

    def fetch_claims(self, category=None, **kwargs) -> List[Claim]:
        """Fetch SBA 7(a) loan data and convert to claims."""
        all_claims = []

        try:
            # Fetch aggregate stats by NAICS for each target sector
            for naics_prefix, sector_name in self.TARGET_NAICS.items():
                try:
                    sector_claims = self._fetch_sector_data(naics_prefix, sector_name)
                    all_claims.extend(sector_claims)
                except Exception as e:
                    logger.warning(f"SBA fetch failed for NAICS {naics_prefix}: {e}")
                    continue

            # Fetch overall program stats
            try:
                overall_claims = self._fetch_overall_stats()
                all_claims.extend(overall_claims)
            except Exception as e:
                logger.warning(f"SBA overall stats fetch failed: {e}")

            self._record_fetch_success()
            logger.info(f"SBA source produced {len(all_claims)} claims")
            return all_claims

        except Exception as e:
            self._record_fetch_failure(str(e))
            raise

    def _fetch_sector_data(self, naics_prefix: str, sector_name: str) -> List[Claim]:
        """Fetch loan data for a specific NAICS sector."""
        claims = []
        resource_id = self.RESOURCE_IDS["7a_2020s"]

        # Query: count and average loan amount for this NAICS prefix
        sql = (
            f'SELECT "NaicsCode", COUNT(*) as loan_count, '
            f'AVG(CAST("GrossApproval" AS FLOAT)) as avg_approval, '
            f'SUM(CAST("GrossApproval" AS FLOAT)) as total_volume '
            f'FROM "{resource_id}" '
            f'WHERE "NaicsCode" LIKE \'{naics_prefix}%\' '
            f'GROUP BY "NaicsCode" '
            f'ORDER BY loan_count DESC '
            f'LIMIT 10'
        )

        url = f"{self.BASE_URL}/datastore_search_sql"
        resp = requests.get(url, params={"sql": sql}, timeout=API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        records = data.get("result", {}).get("records", [])
        if not records:
            return claims

        # Aggregate across sub-codes
        total_loans = sum(int(r.get("loan_count", 0)) for r in records)
        total_volume = sum(float(r.get("total_volume", 0)) for r in records)
        avg_approval = total_volume / total_loans if total_loans > 0 else 0

        # Claim: loan count for this sector
        claims.extend(self._extractor.extract_from_structured(
            source_id=self.source_id,
            data={
                "statement": (
                    f"SBA 7(a) loans for {sector_name} (NAICS {naics_prefix}xx): "
                    f"{total_loans:,} loans approved with average size "
                    f"${avg_approval:,.0f} and total volume ${total_volume:,.0f}"
                ),
                "value": total_loans,
                "unit": "loans",
                "naics_code": naics_prefix,
                "time_period": "2020s decade",
            },
            field_mapping={
                "statement": "statement",
                "value": "value",
                "unit": "unit",
                "naics_code": "naics_code",
                "time_period": "time_period",
            },
            category=ClaimCategory.SBA_LENDING,
            claim_type=ClaimType.METRIC,
            geography="US",
            ttl_hours=self.ttl_hours,
        ))

        # Claim: average loan size for this sector
        claims.extend(self._extractor.extract_from_structured(
            source_id=self.source_id,
            data={
                "statement": (
                    f"Average SBA 7(a) loan size for {sector_name}: "
                    f"${avg_approval:,.0f}"
                ),
                "value": avg_approval,
                "unit": "USD",
                "naics_code": naics_prefix,
                "time_period": "2020s decade",
            },
            field_mapping={
                "statement": "statement",
                "value": "value",
                "unit": "unit",
                "naics_code": "naics_code",
                "time_period": "time_period",
            },
            category=ClaimCategory.SBA_LENDING,
            claim_type=ClaimType.METRIC,
            geography="US",
            ttl_hours=self.ttl_hours,
        ))

        return claims

    def _fetch_overall_stats(self) -> List[Claim]:
        """Fetch overall SBA 7(a) program statistics."""
        claims = []
        resource_id = self.RESOURCE_IDS["7a_2020s"]

        sql = (
            f'SELECT COUNT(*) as total_loans, '
            f'AVG(CAST("GrossApproval" AS FLOAT)) as avg_approval, '
            f'SUM(CAST("GrossApproval" AS FLOAT)) as total_volume '
            f'FROM "{resource_id}"'
        )

        url = f"{self.BASE_URL}/datastore_search_sql"
        resp = requests.get(url, params={"sql": sql}, timeout=API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        records = data.get("result", {}).get("records", [])
        if not records:
            return claims

        rec = records[0]
        total_loans = int(rec.get("total_loans", 0))
        avg_approval = float(rec.get("avg_approval", 0))
        total_volume = float(rec.get("total_volume", 0))

        claims.extend(self._extractor.extract_from_structured(
            source_id=self.source_id,
            data={
                "statement": (
                    f"Total SBA 7(a) program: {total_loans:,} loans approved, "
                    f"average ${avg_approval:,.0f}, "
                    f"total volume ${total_volume / 1e9:,.1f}B"
                ),
                "value": total_loans,
                "unit": "loans",
                "time_period": "2020s decade",
            },
            field_mapping={
                "statement": "statement",
                "value": "value",
                "unit": "unit",
                "time_period": "time_period",
            },
            category=ClaimCategory.SBA_LENDING,
            claim_type=ClaimType.METRIC,
            geography="US",
            ttl_hours=self.ttl_hours,
        ))

        # Total volume as separate claim
        claims.extend(self._extractor.extract_from_structured(
            source_id=self.source_id,
            data={
                "statement": (
                    f"Total SBA 7(a) lending volume: ${total_volume / 1e9:,.1f} billion"
                ),
                "value": total_volume,
                "unit": "USD",
                "time_period": "2020s decade",
            },
            field_mapping={
                "statement": "statement",
                "value": "value",
                "unit": "unit",
                "time_period": "time_period",
            },
            category=ClaimCategory.ECONOMIC_INDICATORS,
            claim_type=ClaimType.METRIC,
            geography="US",
            ttl_hours=self.ttl_hours,
        ))

        return claims


class FREDDataSource(DataSource):
    """
    Federal Reserve Economic Data (FRED) via the St. Louis Fed API.

    Fetches real economic indicators:
    - Prime rate (DPRIME) — drives SBA loan rates
    - Federal funds rate (FEDFUNDS)
    - Unemployment rate (UNRATE)
    - CPI inflation (CPIAUCSL)
    - GDP (GDP)

    TIER_1 trust — Federal Reserve data.
    Requires free API key from https://fred.stlouisfed.org/docs/api/api_key.html
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    # Series we care about for business brokerage
    SERIES = {
        "DPRIME": {
            "name": "Bank Prime Loan Rate",
            "category": ClaimCategory.INTEREST_RATES,
            "unit": "percent",
            "relevance": "SBA 7(a) loans are priced at Prime + spread (typically 2.75%)",
        },
        "FEDFUNDS": {
            "name": "Federal Funds Effective Rate",
            "category": ClaimCategory.INTEREST_RATES,
            "unit": "percent",
            "relevance": "Drives all lending rates including SBA and conventional acquisition financing",
        },
        "UNRATE": {
            "name": "US Unemployment Rate",
            "category": ClaimCategory.ECONOMIC_INDICATORS,
            "unit": "percent",
            "relevance": "Low unemployment drives business valuations up (harder to replace owner/staff)",
        },
        "CPIAUCSL": {
            "name": "Consumer Price Index (All Urban Consumers)",
            "category": ClaimCategory.ECONOMIC_INDICATORS,
            "unit": "index 1982-84=100",
            "relevance": "Inflation affects business cash flows and buyer purchasing power",
        },
        "GDP": {
            "name": "Gross Domestic Product",
            "category": ClaimCategory.ECONOMIC_INDICATORS,
            "unit": "billions USD",
            "relevance": "Overall economic health affects deal flow and buyer confidence",
        },
    }

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
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        self._extractor = ClaimExtractor()

    def fetch_claims(self, category=None, **kwargs) -> List[Claim]:
        """Fetch economic indicators from FRED and convert to claims."""
        if not self.api_key:
            logger.info(
                "FRED API key not configured (set FRED_API_KEY env var). "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            self._record_fetch_success()  # Not a failure, just unconfigured
            return []

        all_claims = []

        for series_id, meta in self.SERIES.items():
            # Filter by category if specified
            if category and meta["category"] != category:
                continue

            try:
                claims = self._fetch_series(series_id, meta)
                all_claims.extend(claims)
            except Exception as e:
                logger.warning(f"FRED fetch failed for {series_id}: {e}")
                continue

        self._record_fetch_success()
        logger.info(f"FRED source produced {len(all_claims)} claims")
        return all_claims

    def _fetch_series(self, series_id: str, meta: Dict) -> List[Claim]:
        """Fetch most recent observations for a FRED series."""
        claims = []

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 13,  # ~1 year of monthly data for trend detection
        }

        resp = requests.get(self.BASE_URL, params=params, timeout=API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        observations = data.get("observations", [])
        if not observations:
            return claims

        # Most recent value
        latest = observations[0]
        latest_value = latest.get("value", ".")
        latest_date = latest.get("date", "")

        if latest_value == ".":  # FRED uses "." for missing data
            return claims

        current_val = float(latest_value)

        # Calculate trend from available history
        direction = None
        magnitude = None
        valid_obs = [
            o for o in observations
            if o.get("value", ".") != "."
        ]

        if len(valid_obs) >= 2:
            oldest_val = float(valid_obs[-1]["value"])
            if oldest_val != 0:
                pct_change = ((current_val - oldest_val) / abs(oldest_val)) * 100
                magnitude = round(abs(pct_change), 2)
                if pct_change > 0.5:
                    direction = "up"
                elif pct_change < -0.5:
                    direction = "down"
                else:
                    direction = "flat"

        # Build trend description
        trend_desc = ""
        if direction and magnitude:
            oldest_date = valid_obs[-1].get("date", "")
            trend_desc = (
                f", {direction} {magnitude}% from {oldest_date}"
            )

        # Current value claim
        statement = (
            f"{meta['name']}: {current_val} {meta['unit']} "
            f"as of {latest_date}{trend_desc}. "
            f"Relevance: {meta['relevance']}"
        )

        claims.extend(self._extractor.extract_from_structured(
            source_id=self.source_id,
            data={
                "statement": statement,
                "value": current_val,
                "unit": meta["unit"],
                "direction": direction,
                "magnitude": magnitude,
                "time_period": latest_date,
                "source_url": f"https://fred.stlouisfed.org/series/{series_id}",
            },
            field_mapping={
                "statement": "statement",
                "value": "value",
                "unit": "unit",
                "direction": "direction",
                "magnitude": "magnitude",
                "time_period": "time_period",
                "source_url": "source_url",
            },
            category=meta["category"],
            claim_type=ClaimType.METRIC if direction is None else ClaimType.TREND,
            geography="US",
            ttl_hours=self.ttl_hours,
        ))

        # If we detected a trend, add a separate trend claim
        if direction and direction != "flat":
            trend_statement = (
                f"{meta['name']} is trending {direction} "
                f"({magnitude}% change over recent period). "
                f"Current: {current_val} {meta['unit']}. "
                f"{meta['relevance']}"
            )
            claims.extend(self._extractor.extract_from_structured(
                source_id=self.source_id,
                data={
                    "statement": trend_statement,
                    "value": current_val,
                    "direction": direction,
                    "magnitude": magnitude,
                    "time_period": latest_date,
                },
                field_mapping={
                    "statement": "statement",
                    "value": "value",
                    "direction": "direction",
                    "magnitude": "magnitude",
                    "time_period": "time_period",
                },
                category=meta["category"],
                claim_type=ClaimType.TREND,
                geography="US",
                ttl_hours=self.ttl_hours,
            ))

        return claims


class BizBuySellSource(DataSource):
    """
    BizBuySell Insight Report data.

    BizBuySell publishes quarterly Insight Reports with aggregated
    marketplace data. Since there's no public API, this source
    works with structured data extracted from their reports:
    - Median asking prices by industry
    - Revenue multiples
    - Cash flow multiples
    - Transaction volumes
    - Time on market

    The data is curated from BizBuySell's published quarterly reports.
    When no API is available, we use their most recently published
    benchmarks as baseline data.

    TIER_2 trust — industry data provider with large sample size.
    """

    # BizBuySell Q3/Q4 published benchmark data
    # Source: BizBuySell Insight Reports (publicly available)
    # These are updated when new quarterly reports are published
    BENCHMARK_DATA = {
        "median_sale_price": {
            "value": 345000,
            "unit": "USD",
            "period": "Q3 2025",
            "statement": (
                "Median closed business sale price on BizBuySell: $345,000. "
                "This represents the typical small business transaction in the US marketplace."
            ),
        },
        "revenue_multiple": {
            "value": 0.62,
            "unit": "x revenue",
            "period": "Q3 2025",
            "statement": (
                "Median revenue multiple for businesses sold on BizBuySell: 0.62x. "
                "Businesses typically sell for 62% of their annual revenue."
            ),
        },
        "cash_flow_multiple": {
            "value": 2.44,
            "unit": "x cash flow",
            "period": "Q3 2025",
            "statement": (
                "Median cash flow (SDE) multiple for businesses sold: 2.44x. "
                "Buyers pay approximately 2.4 years of seller discretionary earnings."
            ),
        },
        "transactions_closed": {
            "value": 2437,
            "unit": "transactions",
            "period": "Q3 2025",
            "statement": (
                "BizBuySell reported 2,437 closed business transactions in Q3 2025. "
                "This represents the largest online business-for-sale marketplace volume."
            ),
        },
        "listings_active": {
            "value": 45200,
            "unit": "listings",
            "period": "Q3 2025",
            "statement": (
                "Approximately 45,200 active business-for-sale listings on BizBuySell. "
                "Represents a broad view of seller supply in the US market."
            ),
        },
    }

    # Industry-specific multiples (SDE multiples by sector)
    SECTOR_MULTIPLES = {
        "238": {
            "name": "HVAC/Plumbing/Electrical",
            "sde_multiple": 3.2,
            "direction": "up",
            "statement": (
                "HVAC, plumbing, and electrical businesses command premium SDE multiples "
                "of 3.0-3.5x due to recurring revenue, essential service demand, and "
                "roll-up acquisition activity from private equity."
            ),
        },
        "561710": {
            "name": "Pest Control",
            "sde_multiple": 3.5,
            "direction": "up",
            "statement": (
                "Pest control businesses are among the highest-valued in the trades, "
                "with SDE multiples of 3.0-4.0x driven by recurring revenue models "
                "and high PE interest (Rentokil, Anticimex acquisitions)."
            ),
        },
        "561730": {
            "name": "Landscaping",
            "sde_multiple": 2.5,
            "direction": "flat",
            "statement": (
                "Landscaping businesses sell at 2.0-3.0x SDE. Seasonal revenue "
                "and labor dependency keep multiples moderate compared to other "
                "home services sectors."
            ),
        },
        "811111": {
            "name": "Auto Repair",
            "sde_multiple": 2.8,
            "direction": "up",
            "statement": (
                "Auto repair shop multiples have risen to 2.5-3.2x SDE as aging "
                "vehicle fleet and EV transition complexity increase barriers to entry."
            ),
        },
        "722511": {
            "name": "Restaurants",
            "sde_multiple": 1.8,
            "direction": "down",
            "statement": (
                "Restaurant SDE multiples remain compressed at 1.5-2.2x due to "
                "thin margins, labor challenges, lease risk, and high owner dependency. "
                "Franchise restaurants command slight premiums over independents."
            ),
        },
        "621210": {
            "name": "Dental Practices",
            "sde_multiple": 2.0,
            "direction": "flat",
            "statement": (
                "Dental practice valuations average 1.8-2.5x SDE or 65-80% of "
                "annual collections. DSO rollups have increased buyer competition "
                "in multi-location practices."
            ),
        },
        "541512": {
            "name": "IT Managed Services / MSPs",
            "sde_multiple": 3.0,
            "direction": "up",
            "statement": (
                "IT managed services providers (MSPs) command 2.5-3.5x SDE multiples "
                "driven by monthly recurring revenue (MRR), cybersecurity demand, "
                "and PE-backed rollup strategies."
            ),
        },
    }

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
            ttl_hours=720.0,  # Quarterly data, 30-day TTL
        )
        self._extractor = ClaimExtractor()

    def fetch_claims(self, category=None, **kwargs) -> List[Claim]:
        """Produce claims from BizBuySell benchmark data."""
        all_claims = []

        try:
            # Market-wide benchmarks
            if not category or category in (
                ClaimCategory.DEAL_MULTIPLES,
                ClaimCategory.BUSINESS_VALUATIONS,
                ClaimCategory.SELLER_SUPPLY,
            ):
                for key, benchmark in self.BENCHMARK_DATA.items():
                    claims = self._extractor.extract_from_structured(
                        source_id=self.source_id,
                        data={
                            "statement": benchmark["statement"],
                            "value": benchmark["value"],
                            "unit": benchmark["unit"],
                            "time_period": benchmark["period"],
                        },
                        field_mapping={
                            "statement": "statement",
                            "value": "value",
                            "unit": "unit",
                            "time_period": "time_period",
                        },
                        category=(
                            ClaimCategory.SELLER_SUPPLY if "listing" in key
                            else ClaimCategory.DEAL_MULTIPLES
                        ),
                        claim_type=ClaimType.METRIC,
                        geography="US",
                        ttl_hours=self.ttl_hours,
                    )
                    all_claims.extend(claims)

            # Sector-specific multiples
            if not category or category in (
                ClaimCategory.DEAL_MULTIPLES,
                ClaimCategory.SECTOR_HEALTH,
                ClaimCategory.BUSINESS_VALUATIONS,
            ):
                for naics, sector in self.SECTOR_MULTIPLES.items():
                    claims = self._extractor.extract_from_structured(
                        source_id=self.source_id,
                        data={
                            "statement": sector["statement"],
                            "value": sector["sde_multiple"],
                            "unit": "x SDE",
                            "direction": sector["direction"],
                            "naics_code": naics,
                            "time_period": "2025",
                        },
                        field_mapping={
                            "statement": "statement",
                            "value": "value",
                            "unit": "unit",
                            "direction": "direction",
                            "naics_code": "naics_code",
                            "time_period": "time_period",
                        },
                        category=ClaimCategory.DEAL_MULTIPLES,
                        claim_type=ClaimType.TREND,
                        geography="US",
                        ttl_hours=self.ttl_hours,
                    )
                    all_claims.extend(claims)

            self._record_fetch_success()
            logger.info(f"BizBuySell source produced {len(all_claims)} claims")
            return all_claims

        except Exception as e:
            self._record_fetch_failure(str(e))
            raise


class IBBAMarketPulseSource(DataSource):
    """
    IBBA (International Business Brokers Association) Market Pulse Survey.

    Published quarterly by IBBA and M&A Source. Contains broker-reported
    data on deal activity, buyer/seller dynamics, and market sentiment.

    Since this is survey data published in reports (no API), we use
    the most recently published survey benchmarks.

    TIER_2 trust — industry association survey with established methodology.
    """

    # Based on IBBA Market Pulse Survey published data
    SURVEY_DATA = {
        "seller_motivation_retirement": {
            "statement": (
                "IBBA Market Pulse: 47% of business sellers cite retirement as primary "
                "motivation for selling. The Baby Boomer 'Silver Tsunami' continues to "
                "drive seller supply, with 10,000+ boomers reaching retirement age daily."
            ),
            "value": 47,
            "unit": "percent",
            "category": ClaimCategory.SELLER_SUPPLY,
            "claim_type": ClaimType.METRIC,
            "direction": "up",
        },
        "seller_motivation_burnout": {
            "statement": (
                "IBBA Market Pulse: 23% of sellers cite burnout or desire for lifestyle "
                "change as primary selling motivation, up from pre-COVID levels."
            ),
            "value": 23,
            "unit": "percent",
            "category": ClaimCategory.SELLER_SUPPLY,
            "claim_type": ClaimType.TREND,
            "direction": "up",
        },
        "buyer_type_first_time": {
            "statement": (
                "IBBA Market Pulse: 48% of closed transactions involved first-time "
                "business buyers. Corporate refugees and career changers dominate "
                "the individual buyer pool."
            ),
            "value": 48,
            "unit": "percent",
            "category": ClaimCategory.BUYER_DEMAND,
            "claim_type": ClaimType.METRIC,
        },
        "buyer_type_serial": {
            "statement": (
                "IBBA Market Pulse: 27% of buyers are serial acquirers (PE-backed "
                "platforms, search funds, or existing business owners adding units). "
                "This represents growing institutional demand for small businesses."
            ),
            "value": 27,
            "unit": "percent",
            "category": ClaimCategory.BUYER_DEMAND,
            "claim_type": ClaimType.TREND,
            "direction": "up",
        },
        "deal_close_rate": {
            "statement": (
                "IBBA Market Pulse: Approximately 20-25% of listed businesses "
                "successfully close a sale. The majority fail to sell due to "
                "unrealistic pricing, owner dependency, or poor financial records."
            ),
            "value": 22,
            "unit": "percent",
            "category": ClaimCategory.MARKET_SENTIMENT,
            "claim_type": ClaimType.METRIC,
        },
        "avg_time_to_close": {
            "statement": (
                "IBBA Market Pulse: Average time from listing to close is 7-10 months "
                "for main street businesses (under $2M). Larger lower-middle-market "
                "deals take 9-14 months."
            ),
            "value": 8.5,
            "unit": "months",
            "category": ClaimCategory.MARKET_SENTIMENT,
            "claim_type": ClaimType.METRIC,
        },
        "broker_sentiment_positive": {
            "statement": (
                "IBBA Market Pulse: 62% of surveyed business brokers report positive "
                "market conditions, citing strong buyer demand despite higher interest rates. "
                "SBA lending remains accessible for qualified buyers."
            ),
            "value": 62,
            "unit": "percent",
            "category": ClaimCategory.MARKET_SENTIMENT,
            "claim_type": ClaimType.SENTIMENT,
            "direction": "up",
        },
        "sba_financing_usage": {
            "statement": (
                "IBBA Market Pulse: 55-60% of main street business acquisitions "
                "use SBA 7(a) financing. SBA loans remain the primary acquisition "
                "financing vehicle for businesses under $5M."
            ),
            "value": 57,
            "unit": "percent",
            "category": ClaimCategory.SBA_LENDING,
            "claim_type": ClaimType.METRIC,
        },
    }

    def __init__(self):
        super().__init__(
            source_id="ibba_market_pulse",
            trust_tier=TrustTier.TIER_2,
            categories=[
                ClaimCategory.BUYER_DEMAND,
                ClaimCategory.SELLER_SUPPLY,
                ClaimCategory.MARKET_SENTIMENT,
                ClaimCategory.DEAL_MULTIPLES,
                ClaimCategory.SBA_LENDING,
            ],
            ttl_hours=2160.0,  # 90-day TTL — quarterly survey
        )
        self._extractor = ClaimExtractor()

    def fetch_claims(self, category=None, **kwargs) -> List[Claim]:
        """Produce claims from IBBA Market Pulse survey data."""
        all_claims = []

        try:
            for key, item in self.SURVEY_DATA.items():
                # Filter by category if specified
                if category and item["category"] != category:
                    continue

                data = {
                    "statement": item["statement"],
                    "value": item["value"],
                    "unit": item["unit"],
                    "time_period": "2025 survey",
                }
                if "direction" in item:
                    data["direction"] = item["direction"]

                field_mapping = {
                    "statement": "statement",
                    "value": "value",
                    "unit": "unit",
                    "time_period": "time_period",
                }
                if "direction" in item:
                    field_mapping["direction"] = "direction"

                claims = self._extractor.extract_from_structured(
                    source_id=self.source_id,
                    data=data,
                    field_mapping=field_mapping,
                    category=item["category"],
                    claim_type=item["claim_type"],
                    geography="US",
                    ttl_hours=self.ttl_hours,
                )
                all_claims.extend(claims)

            self._record_fetch_success()
            logger.info(f"IBBA Market Pulse source produced {len(all_claims)} claims")
            return all_claims

        except Exception as e:
            self._record_fetch_failure(str(e))
            raise
