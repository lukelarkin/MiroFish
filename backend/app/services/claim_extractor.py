"""
Claim Extractor

Takes raw data from any source and decomposes it into atomic,
verifiable Claims. This is the normalization layer — no matter
what format the source provides (JSON API, CSV, HTML, LLM text),
everything gets broken down into Claims with the same structure.

The extractor also handles:
- Deduplication (same claim from same source = skip)
- Staleness tagging (claims get TTLs based on data type)
- Provenance tracking (every claim traces back to its raw source)
"""

import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .data_source import (
    Claim, ClaimCategory, ClaimType, TrustTier
)
from ..utils.logger import get_logger

logger = get_logger('mirofish.claim_extractor')


# TTL defaults by claim type (hours)
# Metrics change fast, events are permanent, sentiments decay
DEFAULT_TTLS = {
    ClaimType.METRIC: 12.0,       # Refresh twice daily
    ClaimType.TREND: 48.0,        # Trends are more stable
    ClaimType.EVENT: 720.0,       # Events are facts (30 days)
    ClaimType.SENTIMENT: 6.0,     # Sentiment is volatile
    ClaimType.FORECAST: 168.0,    # Forecasts valid ~1 week
}


class ClaimExtractor:
    """
    Normalizes raw data into structured Claims.

    Works with three modes:
    1. Structured: JSON/dict data with known schema (API responses)
    2. Tabular: CSV/list data with column mappings
    3. Unstructured: Free text that needs LLM decomposition
    """

    def __init__(self):
        self._seen_fingerprints: Dict[str, datetime] = {}

    def extract_from_structured(
        self,
        source_id: str,
        data: Dict[str, Any],
        field_mapping: Dict[str, str],
        category: ClaimCategory,
        claim_type: ClaimType = ClaimType.METRIC,
        geography: Optional[str] = None,
        ttl_hours: Optional[float] = None,
    ) -> List[Claim]:
        """
        Extract claims from structured API/JSON data.

        Args:
            source_id: Which DataSource produced this
            data: Raw dict from API
            field_mapping: Maps data fields to Claim fields
                e.g. {"approval_count": "value", "period": "time_period"}
            category: What domain this belongs to
            claim_type: Nature of the claim
            geography: Geographic scope
            ttl_hours: Override default TTL

        Returns:
            List of deduplicated Claims
        """
        claims = []
        ttl = ttl_hours or DEFAULT_TTLS.get(claim_type, 24.0)

        # Handle single record
        records = data if isinstance(data, list) else [data]

        for record in records:
            try:
                claim = self._build_claim_from_mapping(
                    source_id=source_id,
                    record=record,
                    field_mapping=field_mapping,
                    category=category,
                    claim_type=claim_type,
                    geography=geography,
                    ttl_hours=ttl,
                )
                if claim and not self._is_duplicate(claim):
                    claims.append(claim)
                    self._seen_fingerprints[claim.fingerprint] = datetime.utcnow()
            except Exception as e:
                logger.warning(f"Failed to extract claim from record: {e}")
                continue

        logger.info(f"Extracted {len(claims)} claims from {source_id} (structured)")
        return claims

    def extract_from_tabular(
        self,
        source_id: str,
        rows: List[Dict[str, Any]],
        value_column: str,
        label_column: str,
        category: ClaimCategory,
        claim_type: ClaimType = ClaimType.METRIC,
        unit: Optional[str] = None,
        geography: Optional[str] = None,
        naics_column: Optional[str] = None,
        time_column: Optional[str] = None,
        ttl_hours: Optional[float] = None,
    ) -> List[Claim]:
        """
        Extract claims from tabular data (CSV rows, database results).

        Each row becomes one Claim.
        """
        claims = []
        ttl = ttl_hours or DEFAULT_TTLS.get(claim_type, 24.0)

        for row in rows:
            try:
                value = row.get(value_column)
                label = row.get(label_column, "")
                statement = f"{label}: {value}"
                if unit:
                    statement += f" {unit}"

                claim_id = self._make_id(source_id, statement)
                claim = Claim(
                    claim_id=claim_id,
                    source_id=source_id,
                    category=category,
                    claim_type=claim_type,
                    statement=statement,
                    value=self._to_numeric(value),
                    unit=unit,
                    naics_code=row.get(naics_column) if naics_column else None,
                    geography=geography,
                    time_period=row.get(time_column) if time_column else None,
                    raw_data=row,
                    ttl_hours=ttl,
                )

                if not self._is_duplicate(claim):
                    claims.append(claim)
                    self._seen_fingerprints[claim.fingerprint] = datetime.utcnow()
            except Exception as e:
                logger.warning(f"Failed to extract claim from row: {e}")
                continue

        logger.info(f"Extracted {len(claims)} claims from {source_id} (tabular)")
        return claims

    def extract_from_llm_decomposition(
        self,
        source_id: str,
        llm_claims: List[Dict[str, Any]],
        base_category: ClaimCategory,
        geography: Optional[str] = None,
        ttl_hours: float = 6.0,
    ) -> List[Claim]:
        """
        Extract claims from LLM-decomposed text.

        The LLM has already broken text into claim dicts with fields:
        - statement: str
        - claim_type: str (metric/trend/event/sentiment/forecast)
        - value: optional numeric
        - direction: optional up/down/flat
        - category: optional override

        These get SYNTHETIC trust unless cross-validated.
        """
        claims = []

        for item in llm_claims:
            try:
                statement = item.get("statement", "")
                if not statement:
                    continue

                ct_str = item.get("claim_type", "sentiment")
                try:
                    claim_type = ClaimType(ct_str)
                except ValueError:
                    claim_type = ClaimType.SENTIMENT

                cat_str = item.get("category")
                try:
                    category = ClaimCategory(cat_str) if cat_str else base_category
                except ValueError:
                    category = base_category

                claim_id = self._make_id(source_id, statement)
                claim = Claim(
                    claim_id=claim_id,
                    source_id=source_id,
                    category=category,
                    claim_type=claim_type,
                    statement=statement,
                    value=self._to_numeric(item.get("value")),
                    unit=item.get("unit"),
                    direction=item.get("direction"),
                    magnitude=self._to_numeric(item.get("magnitude")),
                    naics_code=item.get("naics_code"),
                    geography=geography,
                    time_period=item.get("time_period"),
                    raw_data=item,
                    ttl_hours=ttl_hours,
                )

                if not self._is_duplicate(claim):
                    claims.append(claim)
                    self._seen_fingerprints[claim.fingerprint] = datetime.utcnow()
            except Exception as e:
                logger.warning(f"Failed to extract LLM claim: {e}")
                continue

        logger.info(f"Extracted {len(claims)} claims from {source_id} (llm)")
        return claims

    def _build_claim_from_mapping(
        self,
        source_id: str,
        record: Dict[str, Any],
        field_mapping: Dict[str, str],
        category: ClaimCategory,
        claim_type: ClaimType,
        geography: Optional[str],
        ttl_hours: float,
    ) -> Optional[Claim]:
        """Build a Claim by mapping record fields to Claim fields."""
        # Build statement from available data
        mapped = {}
        for data_field, claim_field in field_mapping.items():
            if data_field in record:
                mapped[claim_field] = record[data_field]

        statement = mapped.get("statement", "")
        if not statement:
            # Auto-generate statement from value fields
            parts = []
            for k, v in mapped.items():
                if k != "raw_data":
                    parts.append(f"{k}={v}")
            statement = "; ".join(parts) if parts else str(record)

        claim_id = self._make_id(source_id, statement)
        return Claim(
            claim_id=claim_id,
            source_id=source_id,
            category=category,
            claim_type=claim_type,
            statement=statement,
            value=self._to_numeric(mapped.get("value")),
            unit=mapped.get("unit"),
            direction=mapped.get("direction"),
            magnitude=self._to_numeric(mapped.get("magnitude")),
            naics_code=mapped.get("naics_code"),
            geography=geography,
            time_period=mapped.get("time_period"),
            raw_data=record,
            source_url=mapped.get("source_url"),
            ttl_hours=ttl_hours,
        )

    def _is_duplicate(self, claim: Claim) -> bool:
        """Check if we've already seen this exact claim."""
        return claim.fingerprint in self._seen_fingerprints

    def _make_id(self, source_id: str, content: str) -> str:
        ts = datetime.utcnow().isoformat()
        raw = f"{source_id}:{content}:{ts}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _to_numeric(value: Any) -> Optional[float]:
        """Safely convert to float, return None if not possible."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def clear_seen(self):
        """Reset dedup cache. Call between full refresh cycles."""
        self._seen_fingerprints.clear()
