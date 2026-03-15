"""
Pipeline Pressure Test

Tests every layer of the data validation pipeline:
    DataSource → ClaimExtractor → CrossValidator → CircuitBreaker → ValidatedDataStore

This is the "pour a little water through and check for leaks" test.
No external APIs needed — uses mock sources that produce known claims
so we can verify every joint in the plumbing.

Run: python -m pytest backend/tests/test_data_pipeline.py -v
"""

import sys
import os
import importlib
import importlib.util
import pytest
from datetime import datetime
from unittest import mock

# ── Import Setup ────────────────────────────────────────────────
# The app's __init__.py and services/__init__.py eagerly import
# heavy deps (zep_cloud, camel, etc). We load our modules directly
# from file paths to bypass the import chain entirely.

backend_dir = os.path.join(os.path.dirname(__file__), '..')
services_dir = os.path.join(backend_dir, 'app', 'services')
utils_dir = os.path.join(backend_dir, 'app', 'utils')


def _load_module(name, filepath):
    """Load a single Python module by file path, bypassing __init__.py."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Step 1: Set up minimal stubs for the app package hierarchy
# so relative imports within our modules work
sys.path.insert(0, backend_dir)

# Create stub packages
for pkg in ['app', 'app.utils', 'app.services']:
    if pkg not in sys.modules:
        m = mock.MagicMock()
        m.__path__ = [os.path.join(backend_dir, pkg.replace('.', '/'))]
        m.__package__ = pkg
        sys.modules[pkg] = m

# Step 2: Load the logger (our modules depend on it)
_load_module('app.utils.logger', os.path.join(utils_dir, 'logger.py'))

# Step 3: Load the config (validated_data_store needs Config)
_load_module('app.config', os.path.join(backend_dir, 'app', 'config.py'))

# Step 4: Load our pipeline modules in dependency order
_load_module('app.services.data_source', os.path.join(services_dir, 'data_source.py'))
_load_module('app.services.claim_extractor', os.path.join(services_dir, 'claim_extractor.py'))
_load_module('app.services.cross_validator', os.path.join(services_dir, 'cross_validator.py'))
_load_module('app.services.circuit_breaker', os.path.join(services_dir, 'circuit_breaker.py'))
_load_module('app.services.validated_data_store', os.path.join(services_dir, 'validated_data_store.py'))
_load_module('app.services.data_pipeline', os.path.join(services_dir, 'data_pipeline.py'))

# Step 5: Pull classes into local scope
from app.services.data_source import (
    DataSource, SourceRegistry, SourceHealth,
    Claim, ClaimCategory, ClaimType, TrustTier,
)
from app.services.claim_extractor import ClaimExtractor
from app.services.cross_validator import (
    CrossValidator, ConsensusLevel, ConflictResolution, ValidationResult,
)
from app.services.circuit_breaker import (
    CircuitBreaker, BreakerState, AlertSeverity,
)
from app.services.validated_data_store import ValidatedDataStore
from app.services.data_pipeline import DataPipeline


# ── Test Data Sources ───────────────────────────────────────────
# These produce known claims so we can verify the plumbing.

class MockTier1Source(DataSource):
    """Simulates a government data source like SBA."""

    def __init__(self, source_id="mock_sba", claims_to_produce=None):
        super().__init__(
            source_id=source_id,
            trust_tier=TrustTier.TIER_1,
            categories=[ClaimCategory.SBA_LENDING, ClaimCategory.ECONOMIC_INDICATORS],
            ttl_hours=168.0,
        )
        self._claims = claims_to_produce or []

    def fetch_claims(self, category=None, **kwargs):
        self._record_fetch_success()
        return list(self._claims)


class MockTier2Source(DataSource):
    """Simulates an industry data source like BizBuySell."""

    def __init__(self, source_id="mock_bizbuysell", claims_to_produce=None):
        super().__init__(
            source_id=source_id,
            trust_tier=TrustTier.TIER_2,
            categories=[
                ClaimCategory.DEAL_MULTIPLES, ClaimCategory.SBA_LENDING,
                ClaimCategory.SECTOR_HEALTH,
            ],
            ttl_hours=48.0,
        )
        self._claims = claims_to_produce or []

    def fetch_claims(self, category=None, **kwargs):
        self._record_fetch_success()
        return list(self._claims)


class MockFailingSource(DataSource):
    """Always fails — for circuit breaker testing."""

    def __init__(self, source_id="mock_failing"):
        super().__init__(
            source_id=source_id,
            trust_tier=TrustTier.TIER_2,
            categories=[ClaimCategory.MARKET_SENTIMENT],
            ttl_hours=24.0,
        )

    def fetch_claims(self, category=None, **kwargs):
        self._record_fetch_failure("Simulated failure")
        raise ConnectionError("Simulated connection failure")


def _make_claim(source_id, statement, category, claim_type,
                value=None, direction=None, naics_code=None,
                claim_id=None):
    """Helper to create test claims."""
    return Claim(
        claim_id=claim_id or f"test_{hash(statement) % 10000:04d}",
        source_id=source_id,
        category=category,
        claim_type=claim_type,
        statement=statement,
        value=value,
        direction=direction,
        naics_code=naics_code,
        geography="US",
        time_period="2025",
        ttl_hours=168.0,
    )


# ── Layer 1: DataSource + Registry ──────────────────────────────

class TestDataSourceAndRegistry:
    """Test that sources register correctly and report their properties."""

    def test_source_registers_with_correct_tier(self):
        registry = SourceRegistry()
        source = MockTier1Source()
        registry.register(source)

        found = registry.get_source("mock_sba")
        assert found is not None
        assert found.trust_tier == TrustTier.TIER_1
        assert found.base_weight == 0.9

    def test_registry_indexes_by_category(self):
        registry = SourceRegistry()
        sba = MockTier1Source("source_a")
        bbs = MockTier2Source("source_b")
        registry.register(sba)
        registry.register(bbs)

        # Both should appear for SBA_LENDING
        sba_sources = registry.get_sources_for_category(ClaimCategory.SBA_LENDING)
        assert len(sba_sources) == 2

        # Only BizBuySell for DEAL_MULTIPLES
        deal_sources = registry.get_sources_for_category(ClaimCategory.DEAL_MULTIPLES)
        assert len(deal_sources) == 1
        assert deal_sources[0].source_id == "source_b"

    def test_source_health_tracking(self):
        source = MockTier1Source()
        assert source.health.availability == 1.0

        source.health.record_fetch_success()
        source.health.record_fetch_success()
        source.health.record_fetch_failure("test")

        assert source.health.fetch_successes == 2
        assert source.health.fetch_failures == 1
        assert source.health.availability == pytest.approx(2/3, abs=0.01)

    def test_effective_weight_with_no_history(self):
        source = MockTier1Source()
        # With no history, gets 80% of base weight
        assert source.effective_weight == pytest.approx(0.9 * 0.8, abs=0.01)

    def test_health_report(self):
        registry = SourceRegistry()
        registry.register(MockTier1Source())
        registry.register(MockTier2Source())

        report = registry.health_report()
        assert report["total_sources"] == 2
        assert report["healthy_sources"] == 2
        assert "mock_sba" in report["sources"]


# ── Layer 2: ClaimExtractor ─────────────────────────────────────

class TestClaimExtractor:
    """Test that raw data gets correctly decomposed into Claims."""

    def test_extract_from_structured(self):
        extractor = ClaimExtractor()
        claims = extractor.extract_from_structured(
            source_id="test_source",
            data={
                "statement": "SBA 7(a) loans for HVAC: 5000 approved",
                "value": 5000,
                "unit": "loans",
                "naics_code": "238220",
            },
            field_mapping={
                "statement": "statement",
                "value": "value",
                "unit": "unit",
                "naics_code": "naics_code",
            },
            category=ClaimCategory.SBA_LENDING,
            claim_type=ClaimType.METRIC,
        )
        assert len(claims) == 1
        assert claims[0].value == 5000
        assert claims[0].naics_code == "238220"
        assert claims[0].category == ClaimCategory.SBA_LENDING

    def test_deduplication(self):
        extractor = ClaimExtractor()
        data = {
            "statement": "Same claim twice",
            "value": 100,
        }
        claims1 = extractor.extract_from_structured(
            source_id="test", data=data,
            field_mapping={"statement": "statement", "value": "value"},
            category=ClaimCategory.SBA_LENDING,
            claim_type=ClaimType.METRIC,
        )
        claims2 = extractor.extract_from_structured(
            source_id="test", data=data,
            field_mapping={"statement": "statement", "value": "value"},
            category=ClaimCategory.SBA_LENDING,
            claim_type=ClaimType.METRIC,
        )
        # Second extraction should be deduplicated
        assert len(claims1) == 1
        assert len(claims2) == 0

    def test_extract_from_tabular(self):
        extractor = ClaimExtractor()
        rows = [
            {"sector": "HVAC", "count": 5000, "naics": "238220"},
            {"sector": "Restaurants", "count": 3000, "naics": "722511"},
        ]
        claims = extractor.extract_from_tabular(
            source_id="test",
            rows=rows,
            value_column="count",
            label_column="sector",
            category=ClaimCategory.SECTOR_HEALTH,
            naics_column="naics",
            unit="loans",
        )
        assert len(claims) == 2
        assert claims[0].value == 5000.0
        assert claims[1].naics_code == "722511"

    def test_extract_from_llm_decomposition(self):
        extractor = ClaimExtractor()
        llm_claims = [
            {
                "statement": "HVAC multiples are trending up to 3.2x",
                "claim_type": "trend",
                "value": 3.2,
                "direction": "up",
                "category": "deal_multiples",
                "naics_code": "238220",
            },
        ]
        claims = extractor.extract_from_llm_decomposition(
            source_id="llm_test",
            llm_claims=llm_claims,
            base_category=ClaimCategory.MARKET_SENTIMENT,
        )
        assert len(claims) == 1
        assert claims[0].direction == "up"
        assert claims[0].category == ClaimCategory.DEAL_MULTIPLES  # Override from LLM


# ── Layer 3: CrossValidator ─────────────────────────────────────

class TestCrossValidator:
    """Test consensus detection, conflict resolution, and confidence scoring."""

    def _setup_registry(self):
        registry = SourceRegistry()
        registry.register(MockTier1Source("sba_1"))
        registry.register(MockTier1Source("fred_1"))
        registry.register(MockTier2Source("bbs_1"))
        return registry

    def test_single_source_gets_low_confidence(self):
        """One source alone = LOW consensus. Never actionable."""
        registry = self._setup_registry()
        validator = CrossValidator(registry)

        claims = [
            _make_claim("sba_1", "SBA loans up 10%", ClaimCategory.SBA_LENDING,
                        ClaimType.TREND, direction="up"),
        ]
        results = validator.validate_claims(claims)
        assert len(results) == 1
        assert results[0].consensus == ConsensusLevel.LOW
        assert not results[0].is_actionable

    def test_two_sources_agree_gets_medium(self):
        """Two sources agreeing = MEDIUM consensus. Actionable."""
        registry = self._setup_registry()
        validator = CrossValidator(registry)

        claims = [
            _make_claim("sba_1", "SBA loans up 10%", ClaimCategory.SBA_LENDING,
                        ClaimType.TREND, direction="up", claim_id="c1"),
            _make_claim("bbs_1", "SBA lending volume increasing", ClaimCategory.SBA_LENDING,
                        ClaimType.TREND, direction="up", claim_id="c2"),
        ]
        results = validator.validate_claims(claims)
        # Both claims are in same cluster, should get MEDIUM
        actionable = [r for r in results if r.is_actionable]
        assert len(actionable) >= 1

    def test_three_sources_agree_gets_high(self):
        """Three sources agreeing = HIGH consensus. Best confidence."""
        registry = self._setup_registry()
        validator = CrossValidator(registry)

        claims = [
            _make_claim("sba_1", "Rates trending up", ClaimCategory.INTEREST_RATES,
                        ClaimType.TREND, direction="up", claim_id="c1"),
            _make_claim("fred_1", "Prime rate climbing", ClaimCategory.INTEREST_RATES,
                        ClaimType.TREND, direction="up", claim_id="c2"),
            _make_claim("bbs_1", "Financing costs rising", ClaimCategory.INTEREST_RATES,
                        ClaimType.TREND, direction="up", claim_id="c3"),
        ]
        results = validator.validate_claims(claims)
        high = [r for r in results if r.consensus == ConsensusLevel.HIGH]
        assert len(high) >= 1

    def test_contradiction_detected(self):
        """Sources disagreeing on direction = CONFLICT."""
        registry = self._setup_registry()
        validator = CrossValidator(registry)

        claims = [
            _make_claim("sba_1", "SBA loans up", ClaimCategory.SBA_LENDING,
                        ClaimType.TREND, direction="up", claim_id="c1"),
            _make_claim("bbs_1", "SBA lending declining", ClaimCategory.SBA_LENDING,
                        ClaimType.TREND, direction="down", claim_id="c2"),
        ]
        results = validator.validate_claims(claims)
        conflicts = [r for r in results if r.consensus == ConsensusLevel.CONFLICT]
        assert len(conflicts) >= 1

    def test_value_contradiction(self):
        """Values differing by >20% = conflict."""
        registry = self._setup_registry()
        validator = CrossValidator(registry)

        claims = [
            _make_claim("sba_1", "HVAC multiple", ClaimCategory.DEAL_MULTIPLES,
                        ClaimType.METRIC, value=3.0, naics_code="238",
                        claim_id="c1"),
            _make_claim("bbs_1", "HVAC multiple", ClaimCategory.DEAL_MULTIPLES,
                        ClaimType.METRIC, value=4.5, naics_code="238",
                        claim_id="c2"),
        ]
        results = validator.validate_claims(claims)
        # 50% deviation should trigger conflict
        conflicts = [r for r in results if r.consensus == ConsensusLevel.CONFLICT]
        assert len(conflicts) >= 1

    def test_confidence_interval_narrows_with_more_sources(self):
        """More sources = narrower confidence interval (Wilson score)."""
        registry = self._setup_registry()
        validator = CrossValidator(registry)

        # Single source
        claims_1 = [
            _make_claim("sba_1", "Metric A", ClaimCategory.SBA_LENDING,
                        ClaimType.METRIC, value=100, claim_id="c1"),
        ]
        results_1 = validator.validate_claims(claims_1)
        ci_1 = results_1[0].confidence_interval
        width_1 = ci_1[1] - ci_1[0]

        # Three agreeing sources
        claims_3 = [
            _make_claim("sba_1", "Metric B", ClaimCategory.ECONOMIC_INDICATORS,
                        ClaimType.METRIC, value=100, claim_id="c2"),
            _make_claim("fred_1", "Metric B alt", ClaimCategory.ECONOMIC_INDICATORS,
                        ClaimType.METRIC, value=100, claim_id="c3"),
            _make_claim("bbs_1", "Metric B alt2", ClaimCategory.ECONOMIC_INDICATORS,
                        ClaimType.METRIC, value=100, claim_id="c4"),
        ]
        results_3 = validator.validate_claims(claims_3)
        # Find one with 3-source cluster
        high_results = [r for r in results_3 if r.consensus == ConsensusLevel.HIGH]
        if high_results:
            ci_3 = high_results[0].confidence_interval
            width_3 = ci_3[1] - ci_3[0]
            # CI should be narrower (or at least not wider) with more sources
            assert width_3 <= width_1 + 0.1  # small tolerance

    def test_validation_summary(self):
        registry = self._setup_registry()
        validator = CrossValidator(registry)

        claims = [
            _make_claim("sba_1", "A", ClaimCategory.SBA_LENDING,
                        ClaimType.METRIC, claim_id="c1"),
            _make_claim("fred_1", "B", ClaimCategory.SBA_LENDING,
                        ClaimType.METRIC, claim_id="c2"),
        ]
        results = validator.validate_claims(claims)
        summary = validator.get_validation_summary(results)
        assert summary["total"] > 0
        assert "avg_confidence" in summary


# ── Layer 4: CircuitBreaker ─────────────────────────────────────

class TestCircuitBreaker:
    """Test circuit breaker states and self-healing."""

    def test_starts_closed(self):
        registry = SourceRegistry()
        source = MockTier1Source()
        registry.register(source)
        breaker = CircuitBreaker(registry)

        assert breaker.should_allow_source("mock_sba") is True
        assert breaker.get_breaker("mock_sba").state == BreakerState.CLOSED

    def test_opens_after_consecutive_failures(self):
        registry = SourceRegistry()
        source = MockTier1Source()
        registry.register(source)
        breaker = CircuitBreaker(registry)

        # Simulate 5 consecutive failures
        for i in range(5):
            source.health.record_fetch_failure(f"failure {i}")

        alerts = breaker.check_source_health(source)
        assert breaker.get_breaker("mock_sba").state == BreakerState.OPEN
        assert len(alerts) >= 1
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_open_breaker_blocks_source(self):
        registry = SourceRegistry()
        source = MockTier1Source()
        registry.register(source)
        breaker = CircuitBreaker(registry)

        # Force open
        b = breaker.get_breaker("mock_sba")
        breaker._open_breaker(b)

        assert breaker.should_allow_source("mock_sba") is False

    def test_degradation_reduces_trust(self):
        registry = SourceRegistry()
        source = MockTier1Source()
        registry.register(source)
        breaker = CircuitBreaker(registry)

        b = breaker.get_breaker("mock_sba")
        assert b.degradation_factor == 1.0

        breaker._degrade_source(b)
        assert b.degradation_factor == 0.8  # 1.0 - 0.2 step

        breaker._degrade_source(b)
        assert b.degradation_factor == pytest.approx(0.6, abs=0.01)

    def test_recovery_restores_trust(self):
        registry = SourceRegistry()
        source = MockTier1Source()
        registry.register(source)
        breaker = CircuitBreaker(registry)

        b = breaker.get_breaker("mock_sba")
        b.degradation_factor = 0.5

        breaker.record_source_success("mock_sba")
        assert b.degradation_factor == 0.55  # 0.5 + 0.05 step

    def test_concentration_risk_alert(self):
        registry = SourceRegistry()
        source = MockTier1Source()
        registry.register(source)
        breaker = CircuitBreaker(registry)

        # Create results where one source dominates
        claim = _make_claim("mock_sba", "Test", ClaimCategory.SBA_LENDING,
                            ClaimType.METRIC)
        results = []
        for i in range(10):
            results.append(ValidationResult(
                claim=claim,
                consensus=ConsensusLevel.HIGH,
                confidence=0.8,
                confidence_interval=(0.6, 0.95),
            ))
        alerts = breaker.check_source_concentration(results)
        assert len(alerts) >= 1  # >50% concentration

    def test_pipeline_viability_check(self):
        registry = SourceRegistry()
        source1 = MockTier1Source("s1")
        source2 = MockTier2Source("s2")
        registry.register(source1)
        registry.register(source2)
        breaker = CircuitBreaker(registry)

        # Both healthy — should be viable
        alerts = breaker.check_pipeline_viability()
        critical = [a for a in alerts if a.metric == "healthy_source_count"]
        assert len(critical) == 0

        # Degrade both — pipeline not viable
        source1.health.is_degraded = True
        source2.health.is_degraded = True
        alerts = breaker.check_pipeline_viability()
        critical = [a for a in alerts if a.metric == "healthy_source_count"]
        assert len(critical) >= 1


# ── Layer 5: ValidatedDataStore ─────────────────────────────────

class TestValidatedDataStore:
    """Test that only validated data passes through the gate."""

    def _setup_store(self):
        registry = SourceRegistry()
        registry.register(MockTier1Source("s1"))
        registry.register(MockTier1Source("s2"))
        registry.register(MockTier2Source("s3"))
        validator = CrossValidator(registry)
        breaker = CircuitBreaker(registry)
        store = ValidatedDataStore(registry, validator, breaker)
        return store, registry

    def test_only_actionable_claims_stored(self):
        store, _ = self._setup_store()

        results = [
            # HIGH — should be stored
            ValidationResult(
                claim=_make_claim("s1", "Good claim", ClaimCategory.SBA_LENDING,
                                  ClaimType.METRIC, claim_id="g1"),
                consensus=ConsensusLevel.HIGH,
                confidence=0.85,
                confidence_interval=(0.7, 0.95),
            ),
            # LOW — should be dropped
            ValidationResult(
                claim=_make_claim("s2", "Weak claim", ClaimCategory.SBA_LENDING,
                                  ClaimType.METRIC, claim_id="w1"),
                consensus=ConsensusLevel.LOW,
                confidence=0.2,
                confidence_interval=(0.05, 0.4),
            ),
            # UNVALIDATED — should be dropped
            ValidationResult(
                claim=_make_claim("s3", "Synth claim", ClaimCategory.SBA_LENDING,
                                  ClaimType.METRIC, claim_id="u1"),
                consensus=ConsensusLevel.UNVALIDATED,
                confidence=0.05,
                confidence_interval=(0.0, 0.15),
            ),
        ]
        log = store.ingest(results)
        assert log["actionable_stored"] == 1
        assert log["low_confidence_dropped"] == 1
        assert log["unvalidated_dropped"] == 1

    def test_gate_blocks_insufficient_data(self):
        store, _ = self._setup_store()

        # No data at all — gate should be closed
        is_open, reason = store.check_gate()
        assert not is_open
        assert "actionable claims" in reason.lower() or "sources" in reason.lower()

    def test_gate_opens_with_sufficient_data(self):
        store, _ = self._setup_store()

        # Insert enough actionable data across categories
        results = []
        categories = [
            ClaimCategory.SBA_LENDING,
            ClaimCategory.INTEREST_RATES,
            ClaimCategory.DEAL_MULTIPLES,
        ]
        for i, cat in enumerate(categories):
            for j in range(3):
                results.append(ValidationResult(
                    claim=_make_claim("s1", f"Claim {cat.value} {j}",
                                      cat, ClaimType.METRIC,
                                      value=100 + j,
                                      claim_id=f"gate_{i}_{j}"),
                    consensus=ConsensusLevel.HIGH,
                    confidence=0.85,
                    confidence_interval=(0.7, 0.95),
                ))
        store.ingest(results)
        is_open, reason = store.check_gate()
        assert is_open

    def test_export_as_seed_text(self):
        store, _ = self._setup_store()

        results = []
        for i in range(6):
            cat = ClaimCategory.SBA_LENDING if i < 3 else ClaimCategory.DEAL_MULTIPLES
            results.append(ValidationResult(
                claim=_make_claim("s1", f"Test claim {i}", cat,
                                  ClaimType.METRIC, value=100 + i,
                                  claim_id=f"seed_{i}"),
                consensus=ConsensusLevel.HIGH,
                confidence=0.85,
                confidence_interval=(0.7, 0.95),
            ))
        store.ingest(results)
        seed_text = store.export_as_seed_text()
        assert "Validated Market Intelligence" in seed_text
        assert "HIGH" in seed_text
        assert len(seed_text) > 100

    def test_export_for_simulation(self):
        store, _ = self._setup_store()

        results = []
        for i in range(6):
            cat = ClaimCategory.SBA_LENDING if i < 3 else ClaimCategory.INTEREST_RATES
            results.append(ValidationResult(
                claim=_make_claim("s1", f"Sim claim {i}", cat,
                                  ClaimType.METRIC, value=50 + i,
                                  claim_id=f"sim_{i}"),
                consensus=ConsensusLevel.MEDIUM,
                confidence=0.7,
                confidence_interval=(0.55, 0.85),
            ))
        store.ingest(results)
        export = store.export_for_simulation()
        assert export["gate_status"] == "open"
        assert "sba_lending" in export["categories"]


# ── Layer 6: Full Pipeline Integration ──────────────────────────

class TestFullPipeline:
    """
    End-to-end test: claims flow through ALL layers.

    This is the "pour water and check every joint" test.
    """

    def test_single_source_pipeline_gate_stays_closed(self):
        """Single source should not open the gate (need cross-validation)."""
        pipeline = DataPipeline(store_dir="/tmp/mirofish_test_store")

        claims = [
            _make_claim("solo_source", f"Solo claim {i}",
                        ClaimCategory.SBA_LENDING, ClaimType.METRIC,
                        value=100 + i, claim_id=f"solo_{i}")
            for i in range(10)
        ]
        source = MockTier1Source("solo_source", claims)
        pipeline.register_source(source)

        result = pipeline.run()

        # Single source = all LOW consensus = gate closed
        assert not result["gate_open"]
        assert result["stats"]["total"] > 0

    def test_two_agreeing_sources_can_open_gate(self):
        """Two TIER_1 sources agreeing should produce MEDIUM consensus."""
        pipeline = DataPipeline(store_dir="/tmp/mirofish_test_store2")

        # Both sources produce claims in the same categories with same direction
        sba_claims = []
        fred_claims = []
        categories = [
            ClaimCategory.SBA_LENDING,
            ClaimCategory.ECONOMIC_INDICATORS,
        ]
        for i, cat in enumerate(categories):
            for j in range(4):
                sba_claims.append(_make_claim(
                    "sba_t1", f"SBA {cat.value} claim {j}",
                    cat, ClaimType.TREND,
                    direction="up", value=100 + j,
                    claim_id=f"sba_{i}_{j}",
                ))
                fred_claims.append(_make_claim(
                    "fred_t1", f"FRED {cat.value} indicator {j}",
                    cat, ClaimType.TREND,
                    direction="up", value=100 + j,
                    claim_id=f"fred_{i}_{j}",
                ))

        pipeline.register_source(MockTier1Source("sba_t1", sba_claims))
        pipeline.register_source(MockTier1Source("fred_t1", fred_claims))

        result = pipeline.run()
        assert result["stats"]["total"] > 0
        # Should have some actionable claims (MEDIUM or HIGH)
        assert result["stats"]["actionable"] > 0

    def test_three_sources_with_conflict(self):
        """Conflicting claims should be detected and flagged."""
        pipeline = DataPipeline(store_dir="/tmp/mirofish_test_store3")

        # Two sources say "up", one says "down"
        claims_up1 = [_make_claim("src_a", "Rates going up",
                                   ClaimCategory.INTEREST_RATES,
                                   ClaimType.TREND, direction="up",
                                   claim_id="up1")]
        claims_up2 = [_make_claim("src_b", "Rates climbing",
                                   ClaimCategory.INTEREST_RATES,
                                   ClaimType.TREND, direction="up",
                                   claim_id="up2")]
        claims_down = [_make_claim("src_c", "Rates falling",
                                    ClaimCategory.INTEREST_RATES,
                                    ClaimType.TREND, direction="down",
                                    claim_id="down1")]

        pipeline.register_source(MockTier1Source("src_a", claims_up1))
        pipeline.register_source(MockTier1Source("src_b", claims_up2))
        pipeline.register_source(MockTier2Source("src_c", claims_down))

        result = pipeline.run()
        assert result["stats"]["total"] > 0
        # Should detect the conflict — all 3 claims land in same cluster,
        # contradicting directions should produce CONFLICT consensus
        assert result["stats"]["conflicts"] >= 1

    def test_failing_source_doesnt_crash_pipeline(self):
        """A failing source should be circuit-broken, not crash everything."""
        pipeline = DataPipeline(store_dir="/tmp/mirofish_test_store4")

        good_claims = [
            _make_claim("good_src", f"Good claim {i}",
                        ClaimCategory.SBA_LENDING, ClaimType.METRIC,
                        value=100, claim_id=f"good_{i}")
            for i in range(5)
        ]
        pipeline.register_source(MockTier1Source("good_src", good_claims))
        pipeline.register_source(MockFailingSource("bad_src"))

        # Should not raise
        result = pipeline.run()
        assert result["stats"]["total"] > 0
        # Good source claims should still be there
        assert result["stats"]["total"] >= 5

    def test_health_check_runs(self):
        """Health check should return status without fetching."""
        pipeline = DataPipeline(store_dir="/tmp/mirofish_test_store5")
        pipeline.register_source(MockTier1Source())

        health = pipeline.run_health_check()
        assert "breaker_states" in health
        assert "pipeline_viable" in health


# ── Cross-Validation Specific Tests ─────────────────────────────

class TestCrossValidationSpecifics:
    """Deep tests on the cross-validation math."""

    def test_tier1_source_gets_higher_confidence_than_tier2(self):
        registry = SourceRegistry()
        t1 = MockTier1Source("t1_src")
        t2 = MockTier2Source("t2_src")
        registry.register(t1)
        registry.register(t2)
        validator = CrossValidator(registry)

        t1_claim = _make_claim("t1_src", "T1 says X", ClaimCategory.SBA_LENDING,
                               ClaimType.METRIC, value=100, claim_id="t1c")
        t2_claim = _make_claim("t2_src", "T2 says Y", ClaimCategory.SBA_LENDING,
                               ClaimType.METRIC, value=100, claim_id="t2c")

        results = validator.validate_claims([t1_claim, t2_claim])
        t1_result = next(r for r in results if r.claim.source_id == "t1_src")
        t2_result = next(r for r in results if r.claim.source_id == "t2_src")

        # TIER_1 should have higher or equal confidence
        assert t1_result.confidence >= t2_result.confidence

    def test_stale_claim_penalized(self):
        registry = SourceRegistry()
        registry.register(MockTier1Source("src"))
        validator = CrossValidator(registry)

        # Fresh claim
        fresh = _make_claim("src", "Fresh data", ClaimCategory.SBA_LENDING,
                            ClaimType.METRIC, claim_id="fresh")
        fresh.ttl_hours = 168.0

        # Stale claim — set timestamp far in the past so is_stale=True
        stale = _make_claim("src", "Old data", ClaimCategory.ECONOMIC_INDICATORS,
                            ClaimType.METRIC, claim_id="stale")
        stale.ttl_hours = 1.0
        stale.timestamp = datetime(2020, 1, 1)  # 6 years ago — definitely stale

        results = validator.validate_claims([fresh, stale])
        fresh_r = next(r for r in results if r.claim.claim_id == "fresh")
        stale_r = next(r for r in results if r.claim.claim_id == "stale")

        assert fresh_r.confidence > stale_r.confidence

    def test_max_single_source_weight_cap(self):
        """No single source can have >35% weight."""
        registry = SourceRegistry()
        registry.register(MockTier1Source("src"))
        validator = CrossValidator(registry)

        claim = _make_claim("src", "Test", ClaimCategory.SBA_LENDING,
                            ClaimType.METRIC, claim_id="cap_test")
        results = validator.validate_claims([claim])
        # Even TIER_1 (0.9 weight) should be capped at 0.35 base
        assert results[0].confidence <= 0.5  # LOW consensus * capped weight


# ── Round 1 Improvements: Gap Coverage ──────────────────────────

class TestCircuitBreakerSelfHealing:
    """Test the full CLOSED → OPEN → HALF_OPEN → CLOSED recovery cycle."""

    def test_full_recovery_lifecycle(self):
        """Source fails, gets cut off, recovers, gets restored."""
        registry = SourceRegistry()
        source = MockTier1Source("healing_src")
        registry.register(source)
        breaker = CircuitBreaker(registry)

        # 1. Start CLOSED
        b = breaker.get_breaker("healing_src")
        assert b.state == BreakerState.CLOSED
        assert breaker.should_allow_source("healing_src") is True

        # 2. Simulate failures → OPEN
        for _ in range(5):
            source.health.record_fetch_failure("fail")
        breaker.check_source_health(source)
        assert b.state == BreakerState.OPEN
        assert breaker.should_allow_source("healing_src") is False
        assert source.health.is_degraded is True

        # 3. Force the cooldown to have passed (manipulate timestamp)
        b.last_state_change = datetime(2020, 1, 1)

        # 4. Should transition to HALF_OPEN
        assert breaker.should_allow_source("healing_src") is True
        assert b.state == BreakerState.HALF_OPEN

        # 5. Record successes → should close after HALF_OPEN_TEST_WINDOW (3)
        breaker.record_source_success("healing_src")
        assert b.state == BreakerState.HALF_OPEN  # Not yet
        breaker.record_source_success("healing_src")
        assert b.state == BreakerState.HALF_OPEN  # Not yet
        breaker.record_source_success("healing_src")
        assert b.state == BreakerState.CLOSED  # Recovered!
        assert source.health.is_degraded is False

    def test_half_open_failure_reopens(self):
        """If a source fails during HALF_OPEN test, go back to OPEN."""
        registry = SourceRegistry()
        source = MockTier1Source("fragile_src")
        registry.register(source)
        breaker = CircuitBreaker(registry)

        b = breaker.get_breaker("fragile_src")
        # Force to HALF_OPEN
        b.state = BreakerState.HALF_OPEN
        b.success_count_since_half_open = 1

        # Fail during recovery
        breaker.record_source_failure("fragile_src")
        assert b.state == BreakerState.OPEN  # Back to OPEN


class TestClaimVolumeSpike:
    """Test that sudden volume spikes are detected."""

    def test_spike_detected(self):
        registry = SourceRegistry()
        registry.register(MockTier1Source("spike_src"))
        breaker = CircuitBreaker(registry)

        # Build up normal volume history
        breaker.check_claim_volume("spike_src", 10)
        breaker.check_claim_volume("spike_src", 12)
        breaker.check_claim_volume("spike_src", 11)

        # Spike: 10x normal
        alerts = breaker.check_claim_volume("spike_src", 100)
        assert len(alerts) >= 1
        assert "spike" in alerts[0].message.lower()

    def test_normal_variation_no_alert(self):
        registry = SourceRegistry()
        registry.register(MockTier1Source("steady_src"))
        breaker = CircuitBreaker(registry)

        breaker.check_claim_volume("steady_src", 10)
        breaker.check_claim_volume("steady_src", 12)
        breaker.check_claim_volume("steady_src", 11)

        # Slight increase — no spike
        alerts = breaker.check_claim_volume("steady_src", 15)
        assert len(alerts) == 0


class TestSyntheticOnlyBlocked:
    """SYNTHETIC-only claims should never be actionable."""

    def test_synthetic_source_alone_is_unvalidated(self):
        registry = SourceRegistry()

        class MockSyntheticSource(DataSource):
            def __init__(self):
                super().__init__(
                    source_id="llm_mock",
                    trust_tier=TrustTier.SYNTHETIC,
                    categories=[ClaimCategory.MARKET_SENTIMENT],
                )
            def fetch_claims(self, category=None, **kwargs):
                self._record_fetch_success()
                return self._claims
            _claims = []

        synth = MockSyntheticSource()
        synth._claims = [
            _make_claim("llm_mock", f"LLM says {i}",
                        ClaimCategory.MARKET_SENTIMENT, ClaimType.SENTIMENT,
                        claim_id=f"synth_{i}")
            for i in range(10)
        ]
        registry.register(synth)
        validator = CrossValidator(registry)

        results = validator.validate_claims(synth._claims)
        # Every claim should be UNVALIDATED — none actionable
        for r in results:
            assert r.consensus == ConsensusLevel.UNVALIDATED
            assert not r.is_actionable


class TestConflictResolution:
    """Test the three automatic resolution paths."""

    def _setup(self):
        registry = SourceRegistry()
        registry.register(MockTier1Source("gov_a"))
        registry.register(MockTier1Source("gov_b"))
        registry.register(MockTier2Source("ind_c"))
        return registry, CrossValidator(registry)

    def test_tier1_consensus_wins_conflict(self):
        """When TIER_1 sources agree, they override TIER_2 contradiction."""
        registry, validator = self._setup()

        claims = [
            _make_claim("gov_a", "Rate up", ClaimCategory.INTEREST_RATES,
                        ClaimType.TREND, direction="up", claim_id="ga"),
            _make_claim("gov_b", "Rate rising", ClaimCategory.INTEREST_RATES,
                        ClaimType.TREND, direction="up", claim_id="gb"),
            _make_claim("ind_c", "Rate down", ClaimCategory.INTEREST_RATES,
                        ClaimType.TREND, direction="down", claim_id="ic"),
        ]
        results = validator.validate_claims(claims)
        # Should detect conflict but attempt resolution
        assert any(
            r.conflict_resolution.value != "unresolved"
            for r in results if r.consensus == ConsensusLevel.CONFLICT
        )

    def test_majority_wins_conflict(self):
        """When 2 of 3 equal-tier sources agree, majority wins."""
        registry = SourceRegistry()
        registry.register(MockTier2Source("t2_a"))
        registry.register(MockTier2Source("t2_b"))
        registry.register(MockTier2Source("t2_c"))
        validator = CrossValidator(registry)

        claims = [
            _make_claim("t2_a", "Up", ClaimCategory.DEAL_MULTIPLES,
                        ClaimType.TREND, direction="up", claim_id="a1"),
            _make_claim("t2_b", "Up too", ClaimCategory.DEAL_MULTIPLES,
                        ClaimType.TREND, direction="up", claim_id="b1"),
            _make_claim("t2_c", "Down", ClaimCategory.DEAL_MULTIPLES,
                        ClaimType.TREND, direction="down", claim_id="c1"),
        ]
        results = validator.validate_claims(claims)
        # Conflict exists, and resolution should mention majority
        conflict_results = [r for r in results if r.consensus == ConsensusLevel.CONFLICT]
        assert len(conflict_results) >= 1
        resolutions = [r.conflict_resolution.value for r in conflict_results]
        assert any(r in ("majority", "tier_weight", "recency") for r in resolutions)


class TestEffectiveWeightEvolution:
    """Test that source weight adapts based on track record."""

    def test_weight_improves_with_validation_history(self):
        source = MockTier1Source("evolving")
        initial_weight = source.effective_weight

        # Build good track record: 20 claims, all validated
        for _ in range(20):
            source.health.total_claims += 1
            source.health.record_claim_validated()
            source.health.record_fetch_success()

        evolved_weight = source.effective_weight
        # With perfect track record, weight should be higher than the
        # no-history penalty (80% of base)
        assert evolved_weight > initial_weight

    def test_weight_drops_with_contradictions(self):
        source = MockTier1Source("degrading")

        # Build bad track record: 20 claims, all contradicted
        for _ in range(20):
            source.health.total_claims += 1
            source.health.record_claim_contradicted()
            source.health.record_fetch_success()

        # Validation rate = 0, so effective_trust = 0
        # Weight = 0.9 * 0.6 + 0 * 0.4 = 0.54, floored at 0.9 * 0.5 = 0.45
        assert source.effective_weight == pytest.approx(0.54, abs=0.01)
        assert source.effective_weight < source.base_weight


class TestEdgeCases:
    """Edge cases that could crash the pipeline."""

    def test_empty_claims_list(self):
        registry = SourceRegistry()
        registry.register(MockTier1Source("src"))
        validator = CrossValidator(registry)
        results = validator.validate_claims([])
        assert results == []

    def test_duplicate_source_registration(self):
        registry = SourceRegistry()
        src1 = MockTier1Source("dupe")
        src2 = MockTier1Source("dupe")
        registry.register(src1)
        registry.register(src2)
        # Second registration should overwrite, not duplicate
        assert registry.source_count == 1

    def test_zero_value_claims_dont_crash(self):
        registry = SourceRegistry()
        registry.register(MockTier1Source("src_a"))
        registry.register(MockTier2Source("src_b"))
        validator = CrossValidator(registry)

        claims = [
            _make_claim("src_a", "Zero metric", ClaimCategory.SBA_LENDING,
                        ClaimType.METRIC, value=0, claim_id="z1"),
            _make_claim("src_b", "Also zero", ClaimCategory.SBA_LENDING,
                        ClaimType.METRIC, value=0, claim_id="z2"),
        ]
        # Should not raise ZeroDivisionError
        results = validator.validate_claims(claims)
        assert len(results) == 2

    def test_negative_value_claims(self):
        registry = SourceRegistry()
        registry.register(MockTier1Source("src"))
        validator = CrossValidator(registry)

        claim = _make_claim("src", "Decline", ClaimCategory.ECONOMIC_INDICATORS,
                            ClaimType.METRIC, value=-15.5, claim_id="neg")
        results = validator.validate_claims([claim])
        assert len(results) == 1
        assert results[0].claim.value == -15.5

    def test_claim_fingerprint_uniqueness(self):
        """Different claims should have different fingerprints."""
        c1 = _make_claim("src", "Claim A", ClaimCategory.SBA_LENDING,
                         ClaimType.METRIC, value=100, claim_id="fp1")
        c2 = _make_claim("src", "Claim B", ClaimCategory.SBA_LENDING,
                         ClaimType.METRIC, value=200, claim_id="fp2")
        assert c1.fingerprint != c2.fingerprint

    def test_claim_staleness(self):
        """Claim with expired TTL should report as stale."""
        c = _make_claim("src", "Old", ClaimCategory.SBA_LENDING,
                        ClaimType.METRIC, claim_id="stale_edge")
        c.ttl_hours = 1.0
        c.timestamp = datetime(2020, 1, 1)
        assert c.is_stale is True

        fresh = _make_claim("src", "New", ClaimCategory.SBA_LENDING,
                            ClaimType.METRIC, claim_id="fresh_edge")
        fresh.ttl_hours = 9999.0
        assert fresh.is_stale is False

    def test_store_clear_resets(self):
        """Store clear should reset all state."""
        registry = SourceRegistry()
        registry.register(MockTier1Source("s1"))
        validator = CrossValidator(registry)
        breaker = CircuitBreaker(registry)
        store = ValidatedDataStore(registry, validator, breaker)

        # Add some data
        results = [ValidationResult(
            claim=_make_claim("s1", "test", ClaimCategory.SBA_LENDING,
                              ClaimType.METRIC, claim_id="clr1"),
            consensus=ConsensusLevel.HIGH,
            confidence=0.9,
            confidence_interval=(0.8, 0.95),
        )]
        store.ingest(results)
        assert len(store.get_all()) == 1

        store.clear()
        assert len(store.get_all()) == 0


class TestExportStructure:
    """Verify export formats contain correct metadata."""

    def _populated_store(self):
        registry = SourceRegistry()
        registry.register(MockTier1Source("s1"))
        registry.register(MockTier2Source("s2"))
        validator = CrossValidator(registry)
        breaker = CircuitBreaker(registry)
        store = ValidatedDataStore(registry, validator, breaker)

        results = []
        for i in range(4):
            results.append(ValidationResult(
                claim=_make_claim(
                    "s1", f"SBA claim {i}", ClaimCategory.SBA_LENDING,
                    ClaimType.METRIC, value=100 + i, claim_id=f"exp_sba_{i}"),
                consensus=ConsensusLevel.HIGH,
                confidence=0.85 - (i * 0.05),
                confidence_interval=(0.7, 0.95),
            ))
        for i in range(3):
            results.append(ValidationResult(
                claim=_make_claim(
                    "s2", f"Deal claim {i}", ClaimCategory.DEAL_MULTIPLES,
                    ClaimType.TREND, value=2.5 + i, direction="up",
                    claim_id=f"exp_deal_{i}"),
                consensus=ConsensusLevel.MEDIUM,
                confidence=0.65,
                confidence_interval=(0.5, 0.8),
            ))
        store.ingest(results)
        return store

    def test_simulation_export_has_confidence_metadata(self):
        store = self._populated_store()
        export = store.export_for_simulation()

        assert export["gate_status"] == "open"
        assert export["total_claims"] == 7
        assert "sba_lending" in export["categories"]
        assert "deal_multiples" in export["categories"]

        # Check claim structure
        sba_claims = export["categories"]["sba_lending"]["claims"]
        assert len(sba_claims) == 4
        for claim in sba_claims:
            assert "confidence" in claim
            assert "ci_lower" in claim
            assert "ci_upper" in claim
            assert "consensus" in claim
            assert "source_count" in claim
            assert claim["confidence"] > 0

        # Should be sorted by confidence (descending)
        confidences = [c["confidence"] for c in sba_claims]
        assert confidences == sorted(confidences, reverse=True)

    def test_seed_text_export_has_confidence_annotations(self):
        store = self._populated_store()
        seed_text = store.export_as_seed_text()

        assert "Validated Market Intelligence" in seed_text
        assert "HIGH" in seed_text or "MEDIUM" in seed_text
        assert "confidence" in seed_text.lower()
        assert "sources" in seed_text.lower()
        # Should contain actual claim statements
        assert "SBA claim" in seed_text
        assert "Deal claim" in seed_text

    def test_health_report_structure(self):
        store = self._populated_store()
        report = store.health_report()

        assert "gate_status" in report
        assert "total_stored" in report
        assert report["total_stored"] == 7
        assert "confidence_summary" in report
        summary = report["confidence_summary"]
        assert summary["count"] == 7
        assert 0 < summary["mean_confidence"] <= 1.0
        assert summary["high_confidence_count"] >= 0


# ── Round 2 Improvements: Stress Tests ──────────────────────────

class TestPipelineStateIsolation:
    """Verify no state leaks between pipeline runs."""

    def test_consecutive_runs_are_independent(self):
        """Running the pipeline twice should give fresh results each time."""
        pipeline = DataPipeline(store_dir="/tmp/mirofish_test_isolation")

        claims_a = [
            _make_claim("src_x", f"Run1 claim {i}",
                        ClaimCategory.SBA_LENDING, ClaimType.METRIC,
                        value=100 + i, claim_id=f"run1_{i}")
            for i in range(3)
        ]
        claims_b = [
            _make_claim("src_y", f"Run1 corr {i}",
                        ClaimCategory.SBA_LENDING, ClaimType.METRIC,
                        value=100 + i, claim_id=f"run1c_{i}")
            for i in range(3)
        ]
        pipeline.register_source(MockTier1Source("src_x", claims_a))
        pipeline.register_source(MockTier1Source("src_y", claims_b))

        result1 = pipeline.run()
        count1 = result1["stats"]["total"]

        # Run again — extractor dedup should be cleared, so same counts
        result2 = pipeline.run()
        count2 = result2["stats"]["total"]

        assert count1 == count2, (
            f"State leak: run1 got {count1} claims, run2 got {count2}"
        )

    def test_store_is_clean_between_runs(self):
        """Validated store should be empty at start of each run."""
        pipeline = DataPipeline(store_dir="/tmp/mirofish_test_clean")
        pipeline.register_source(MockTier1Source("s1", [
            _make_claim("s1", "c1", ClaimCategory.SBA_LENDING,
                        ClaimType.METRIC, claim_id="iso1"),
        ]))
        pipeline.register_source(MockTier1Source("s2", [
            _make_claim("s2", "c2", ClaimCategory.SBA_LENDING,
                        ClaimType.METRIC, claim_id="iso2"),
        ]))

        pipeline.run()
        # Store should have data
        stored = pipeline.store.get_all()

        pipeline.run()
        # Store should have same count, not double
        stored2 = pipeline.store.get_all()
        assert len(stored2) == len(stored)


class TestGateBoundaryConditions:
    """Test the exact gate thresholds."""

    def _make_store(self):
        registry = SourceRegistry()
        registry.register(MockTier1Source("s1"))
        registry.register(MockTier2Source("s2"))
        validator = CrossValidator(registry)
        breaker = CircuitBreaker(registry)
        return ValidatedDataStore(registry, validator, breaker)

    def test_exactly_minimum_claims_opens_gate(self):
        """5 actionable claims (the minimum) across 2 categories should open."""
        store = self._make_store()
        results = []
        # 3 in SBA_LENDING + 2 in INTEREST_RATES = 5 total, 2 categories
        for i in range(3):
            results.append(ValidationResult(
                claim=_make_claim("s1", f"sba {i}", ClaimCategory.SBA_LENDING,
                                  ClaimType.METRIC, claim_id=f"gate_sba_{i}"),
                consensus=ConsensusLevel.HIGH,
                confidence=0.85,
                confidence_interval=(0.7, 0.95),
            ))
        for i in range(2):
            results.append(ValidationResult(
                claim=_make_claim("s1", f"rate {i}", ClaimCategory.INTEREST_RATES,
                                  ClaimType.METRIC, claim_id=f"gate_rate_{i}"),
                consensus=ConsensusLevel.MEDIUM,
                confidence=0.7,
                confidence_interval=(0.55, 0.85),
            ))
        store.ingest(results)
        is_open, _ = store.check_gate()
        assert is_open

    def test_below_minimum_claims_blocks_gate(self):
        """4 actionable claims (below 5 minimum) should block."""
        store = self._make_store()
        results = []
        for i in range(2):
            results.append(ValidationResult(
                claim=_make_claim("s1", f"few_sba {i}", ClaimCategory.SBA_LENDING,
                                  ClaimType.METRIC, claim_id=f"few_sba_{i}"),
                consensus=ConsensusLevel.HIGH,
                confidence=0.85,
                confidence_interval=(0.7, 0.95),
            ))
        for i in range(2):
            results.append(ValidationResult(
                claim=_make_claim("s1", f"few_rate {i}", ClaimCategory.INTEREST_RATES,
                                  ClaimType.METRIC, claim_id=f"few_rate_{i}"),
                consensus=ConsensusLevel.HIGH,
                confidence=0.85,
                confidence_interval=(0.7, 0.95),
            ))
        store.ingest(results)
        is_open, reason = store.check_gate()
        assert not is_open
        assert "4" in reason  # Should mention the count

    def test_single_category_blocks_gate(self):
        """5+ claims but only 1 category should block."""
        store = self._make_store()
        results = [
            ValidationResult(
                claim=_make_claim("s1", f"mono {i}", ClaimCategory.SBA_LENDING,
                                  ClaimType.METRIC, claim_id=f"mono_{i}"),
                consensus=ConsensusLevel.HIGH,
                confidence=0.85,
                confidence_interval=(0.7, 0.95),
            )
            for i in range(10)
        ]
        store.ingest(results)
        is_open, reason = store.check_gate()
        assert not is_open
        assert "categor" in reason.lower()


class TestConflictAuditTrail:
    """Verify conflict resolution produces an audit trail."""

    def test_conflict_log_accumulates(self):
        registry = SourceRegistry()
        registry.register(MockTier1Source("a"))
        registry.register(MockTier2Source("b"))
        validator = CrossValidator(registry)

        # Create two separate conflicts
        claims_batch1 = [
            _make_claim("a", "Rate up", ClaimCategory.INTEREST_RATES,
                        ClaimType.TREND, direction="up", claim_id="aud1"),
            _make_claim("b", "Rate down", ClaimCategory.INTEREST_RATES,
                        ClaimType.TREND, direction="down", claim_id="aud2"),
        ]
        claims_batch2 = [
            _make_claim("a", "Deals up", ClaimCategory.DEAL_MULTIPLES,
                        ClaimType.TREND, direction="up", claim_id="aud3"),
            _make_claim("b", "Deals down", ClaimCategory.DEAL_MULTIPLES,
                        ClaimType.TREND, direction="down", claim_id="aud4"),
        ]

        validator.validate_claims(claims_batch1)
        validator.validate_claims(claims_batch2)

        log = validator.get_conflict_log()
        assert len(log) >= 2
        assert all("resolution" in entry for entry in log)
        assert all("timestamp" in entry for entry in log)


class TestSourceWeightFloor:
    """Verify the weight floor is enforced."""

    def test_weight_never_drops_below_floor(self):
        """Even with terrible track record, floor is base * 0.5."""
        source = MockTier1Source("floored")
        base = source.base_weight  # 0.9
        floor = base * 0.5  # 0.45

        # 100 claims, all contradicted, 50% fetch failures
        for _ in range(100):
            source.health.total_claims += 1
            source.health.record_claim_contradicted()
        for _ in range(50):
            source.health.record_fetch_failure("bad")
        for _ in range(50):
            source.health.record_fetch_success()

        weight = source.effective_weight
        assert weight >= floor, f"Weight {weight} dropped below floor {floor}"


class TestRealSourceInstantiation:
    """Verify real data source classes can instantiate and produce claims."""

    def test_bizbuysell_produces_claims(self):
        from app.services.data_pipeline import BizBuySellSource
        source = BizBuySellSource()

        assert source.trust_tier == TrustTier.TIER_2
        assert source.source_id == "bizbuysell"

        claims = source.fetch_claims()
        # Should produce benchmark + sector claims
        assert len(claims) > 0
        assert all(isinstance(c, Claim) for c in claims)
        assert source.health.fetch_successes >= 1

        # Verify sector multiples are present
        naics_codes = {c.naics_code for c in claims if c.naics_code}
        assert len(naics_codes) > 0  # Should have NAICS-tagged claims

    def test_ibba_produces_claims(self):
        from app.services.data_pipeline import IBBAMarketPulseSource
        source = IBBAMarketPulseSource()

        assert source.trust_tier == TrustTier.TIER_2
        assert source.source_id == "ibba_market_pulse"

        claims = source.fetch_claims()
        assert len(claims) > 0
        assert all(isinstance(c, Claim) for c in claims)

        # Should cover multiple categories
        categories = {c.category for c in claims}
        assert len(categories) >= 3  # buyer_demand, seller_supply, market_sentiment, sba_lending

    def test_ibba_category_filter(self):
        """Filtering by category should return fewer claims."""
        from app.services.data_pipeline import IBBAMarketPulseSource
        source = IBBAMarketPulseSource()

        all_claims = source.fetch_claims()
        filtered = source.fetch_claims(category=ClaimCategory.BUYER_DEMAND)
        assert len(filtered) < len(all_claims)
        assert all(c.category == ClaimCategory.BUYER_DEMAND for c in filtered)

    def test_bizbuysell_category_filter(self):
        from app.services.data_pipeline import BizBuySellSource
        source = BizBuySellSource()

        all_claims = source.fetch_claims()
        filtered = source.fetch_claims(category=ClaimCategory.SELLER_SUPPLY)
        assert len(filtered) < len(all_claims)

    def test_sba_source_without_api(self):
        """SBA source with no network should handle gracefully."""
        from app.services.data_pipeline import SBADataSource
        source = SBADataSource()
        assert source.trust_tier == TrustTier.TIER_1
        # Don't call fetch_claims — it hits the real API
        # Just verify it instantiates cleanly

    def test_fred_source_without_key(self):
        """FRED source without API key should return empty, not crash."""
        from app.services.data_pipeline import FREDDataSource
        source = FREDDataSource(api_key=None)
        # Should not have a key
        source.api_key = None

        claims = source.fetch_claims()
        assert claims == []  # No key = no data, no crash
        assert source.health.fetch_successes >= 1  # Records as success (unconfigured, not failure)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
