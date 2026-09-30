"""
normalizer.py - Clinical Drug Name Normalizer for VNR Hackathon
Maps Indian commercial brand names and colloquial inputs to canonical generic (INN) names.
Supports exact matching and fuzzy distance-based matching via rapidfuzz.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import rapidfuzz
from rapidfuzz import fuzz, process

# Default path for brand mapping data
DEFAULT_BRAND_MAP_PATH = Path(__file__).resolve().parent / "brand_map.csv"

# In-memory lookup cache: mapping lowercased normalized query token -> canonical generic name
_BRAND_TO_GENERIC: Dict[str, str] = {}
_CANDIDATE_KEYS: list[str] = []
_INITIALIZED = False


def _clean_input_token(text: str) -> str:
    """
    Cleans raw drug input strings by removing dosage artifacts (e.g., '650mg' -> '650',
    'Tab.', 'Cap.') and extraneous symbols while preserving key alphanumeric identifiers.
    """
    if not text:
        return ""
    cleaned = text.strip().lower()
    # Remove dosage units attached to numbers: e.g. "650mg" -> "650"
    cleaned = re.sub(r"(\d+)\s*(mg|mcg|ml|g)\b", r"\1", cleaned)
    # Remove common clinical formulation prefixes
    cleaned = re.sub(r"\b(tab|capsule|cap|tablet|syrup|inj|injection)\b\.?", "", cleaned)
    # Normalize multiple whitespace and dashes
    cleaned = re.sub(r"[\s\-_]+", " ", cleaned).strip()
    return cleaned


def load_brand_map(csv_path: Optional[str | Path] = None) -> None:
    """
    Loads brand-to-generic mappings from CSV into an in-memory index.
    Data provenance:
      - Curated Indian Brand Formulary: Indian trade names (e.g., Dolo, Pan-D, Ecosprin)
      - INN Canonical Generics: Standard International Nonproprietary Names
    """
    global _BRAND_TO_GENERIC, _CANDIDATE_KEYS, _INITIALIZED

    target_path = Path(csv_path) if csv_path else DEFAULT_BRAND_MAP_PATH
    if not target_path.exists():
        # Fallback to local working directory
        target_path = Path("brand_map.csv")

    _BRAND_TO_GENERIC.clear()

    if target_path.exists():
        with open(target_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(row for row in f if not row.startswith("#"))
            for row in reader:
                brand = row.get("brand_name", "").strip().lower()
                generic = row.get("generic_name", "").strip().lower()
                if not brand or not generic:
                    continue

                _BRAND_TO_GENERIC[brand] = generic
                # Also index generic directly so generic queries map to themselves
                _BRAND_TO_GENERIC[generic] = generic

                # Index cleaned representations without hyphens/spaces
                brand_clean = re.sub(r"[\s\-_]+", " ", brand).strip()
                _BRAND_TO_GENERIC[brand_clean] = generic

                # Index brand base without trailing numbers (e.g., "dolo 650" -> "dolo")
                base_brand = re.sub(r"\b\d+\b", "", brand_clean).strip()
                if base_brand and len(base_brand) >= 3 and base_brand not in _BRAND_TO_GENERIC:
                    _BRAND_TO_GENERIC[base_brand] = generic

    _CANDIDATE_KEYS = list(_BRAND_TO_GENERIC.keys())
    _INITIALIZED = True


def normalize_drug_name(
    raw: str,
    threshold: float = 78.0,
    brand_map_path: Optional[str | Path] = None,
) -> Optional[str]:
    """
    Normalizes raw user or prescription drug inputs to canonical generic names.

    Workflow:
      1. Empty/None check.
      2. Direct exact lookup (O(1)) against known brands and generics.
      3. Cleaned token lookup (removing formulation/dosage noise).
      4. Fuzzy matching via rapidfuzz.process.extractOne with WRatio.
      5. Rejection if score < threshold (returns None).

    Parameters:
      raw: The input string from clinician or prescription text.
      threshold: Minimum fuzzy similarity score (0-100) required for a match.
      brand_map_path: Optional custom path to brand_map.csv.

    Returns:
      Canonical generic name (lowercase string) or None if no acceptable match is found.
    """
    if not raw or not isinstance(raw, str):
        return None

    if not _INITIALIZED:
        load_brand_map(brand_map_path)

    raw_lower = raw.strip().lower()
    if not raw_lower:
        return None

    # Step 1: Direct exact match
    if raw_lower in _BRAND_TO_GENERIC:
        return _BRAND_TO_GENERIC[raw_lower]

    # Step 2: Cleaned token exact match
    cleaned = _clean_input_token(raw_lower)
    if cleaned in _BRAND_TO_GENERIC:
        return _BRAND_TO_GENERIC[cleaned]

    # Cleaned without trailing numbers
    base_cleaned = re.sub(r"\b\d+\b", "", cleaned).strip()
    if base_cleaned in _BRAND_TO_GENERIC:
        return _BRAND_TO_GENERIC[base_cleaned]

    # Step 3: Fuzzy matching using rapidfuzz WRatio
    if not _CANDIDATE_KEYS:
        return None

    match: Optional[Tuple[str, float, int]] = process.extractOne(
        cleaned if cleaned else raw_lower,
        _CANDIDATE_KEYS,
        scorer=fuzz.WRatio,
    )

    if match is not None:
        best_key, score, _ = match
        if score >= threshold:
            return _BRAND_TO_GENERIC[best_key]

    return None


# ============================================================================
# ISOLATED TEST SUITE
# ============================================================================
if __name__ == "__main__":
    load_brand_map()

    # Test cases covering: Indian brands, generics, brand typos, generic typos,
    # and unmatchable/garbage strings (should return None).
    test_cases = [
        # (Input, Expected Generic, Description)
        ("Dolo 650", "paracetamol", "Exact Indian brand with strength"),
        ("Crocin", "paracetamol", "Exact Indian brand"),
        ("Ecosprin 75", "aspirin", "Exact Indian cardiovascular brand"),
        ("Augmentin 625", "amoxicillin+clavulanate", "Exact combination antibiotic brand"),
        ("Pan-D", "pantoprazole+domperidone", "Exact hyphenated brand"),
        ("Glycomet 500", "metformin", "Exact antidiabetic brand"),
        ("Atorva 10", "atorvastatin", "Exact statin brand"),
        ("Silagra 50", "sildenafil", "Exact PDE5 inhibitor brand"),
        ("Sorbitrate", "nitroglycerin", "Exact nitrate brand"),
        ("paracetamol", "paracetamol", "Canonical generic directly"),
        ("metformin", "metformin", "Canonical generic directly"),
        ("dollo 650", "paracetamol", "Typo in Indian brand name"),
        ("ecospren", "aspirin", "Typo in Indian brand name"),
        ("pan d", "pantoprazole+domperidone", "Whitespace variation in brand"),
        ("augmntin", "amoxicillin+clavulanate", "Typo in antibiotic brand"),
        ("paracetmol", "paracetamol", "Common misspelling of generic"),
        ("asprin", "aspirin", "Common misspelling of generic"),
        ("metfomin", "metformin", "Common misspelling of generic"),
        ("Tab Dolo 650mg", "paracetamol", "Prescription prefix and unit formatting"),
        ("chocolate bar", None, "Non-drug negative control"),
        ("headache pain", None, "Symptom description negative control"),
        ("xyz123randomtext", None, "Random gibberish negative control"),
    ]

    print("=" * 80)
    print("RUNNING normalizer.py VERIFICATION SUITE (15+ TEST CASES)")
    print("=" * 80)

    passed_count = 0
    total_count = len(test_cases)

    for raw_input, expected, description in test_cases:
        actual = normalize_drug_name(raw_input)
        status = "PASS" if actual == expected else "FAIL"
        if status == "PASS":
            passed_count += 1
        print(f"[{status}] Input: '{raw_input:<20}' | Expected: {str(expected):<25} | Actual: {str(actual):<25} | ({description})")

    print("-" * 80)
    print(f"Results: {passed_count}/{total_count} passed ({(passed_count/total_count)*100:.1f}%)")
    print("=" * 80)

    if passed_count == total_count:
        print("ALL TESTS PASSED SUCCESSFULLY.")
    else:
        raise SystemExit(1)
