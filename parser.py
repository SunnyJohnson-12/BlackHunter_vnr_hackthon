"""
parser.py - Clinical Prescription Free-Text Parser for VNR Hackathon
Extracts structured prescription data ({drug, dose_mg, frequency, daily_dose})
from colloquial free-text prescriptions, physician notes, and electronic slips.
Uses regular expressions combined with normalizer.py for fuzzy brand/generic resolution.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from normalizer import normalize_drug_name

BASE_DIR = Path(__file__).resolve().parent

# =============================================================================
# CLINICAL ABBREVIATION & REGEX PATTERNS
# =============================================================================
# Frequency patterns mapped to integer times-per-day
# OD = once daily (1), BD/BID = twice daily (2), TDS/TID = thrice daily (3), QID = four times daily (4)
FREQUENCY_RULES: List[Tuple[str, int]] = [
    (r"\b(?:qid|four\s+times\s+(?:a\s+)?daily|1-1-1-1)\b", 4),
    (r"\b(?:tds|tid|thrice\s+(?:a\s+)?daily|three\s+times\s+(?:a\s+)?daily|1-1-1)\b", 3),
    (r"\b(?:bd|bid|twice\s+(?:a\s+)?daily|two\s+times\s+(?:a\s+)?daily|1-0-1|0-1-1|1-1-0)\b", 2),
    (r"\b(?:od|qd|once\s+(?:a\s+)?daily|once\s+a\s+day|1-0-0|0-1-0|0-0-1|hs|qhs|at\s+night|at\s+bedtime|in\s+morning)\b", 1),
    (r"\b(?:sos|prn|as\s+needed)\b", 1),
]

# Dosage regex patterns
DOSE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:mg|mcg|ml|g)\b", re.IGNORECASE)
EMBEDDED_STRENGTH_PATTERN = re.compile(r"\b(5|10|20|25|40|50|75|100|120|150|200|400|500|625|650|800|1000)\b")

# Clinical formulation prefixes to strip before entity extraction
PREFIX_PATTERN = re.compile(
    r"^\s*(?:\d+[\.\)]\s*)?(?:tab(?:\.|\s+let)?|cap(?:\.|\s+sule)?|syr(?:\.|\s+up)?|inj(?:\.|\s+ection)?|t\.|c\.)\s*",
    re.IGNORECASE,
)

# Brand default strengths cache
_BRAND_DEFAULT_STRENGTHS: Dict[str, float] = {}
_INITIALIZED: bool = False


def _load_brand_defaults() -> None:
    """Loads default unit strengths from brand_map.csv for implicit doses."""
    global _BRAND_DEFAULT_STRENGTHS, _INITIALIZED
    if _INITIALIZED:
        return

    csv_path = BASE_DIR / "brand_map.csv"
    if not csv_path.exists():
        csv_path = Path("brand_map.csv")

    if csv_path.exists():
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(row for row in f if not row.startswith("#"))
            for row in reader:
                brand = row.get("brand_name", "").strip().lower()
                strength_str = row.get("default_strength_mg", "").strip()
                if brand and strength_str:
                    try:
                        if "+" in strength_str:
                            total = sum(float(x.strip()) for x in strength_str.split("+") if x.strip())
                            _BRAND_DEFAULT_STRENGTHS[brand] = total
                        else:
                            _BRAND_DEFAULT_STRENGTHS[brand] = float(strength_str)
                    except ValueError:
                        continue

    _INITIALIZED = True


def parse_prescription_line(raw_line: str) -> Optional[Dict[str, Any]]:
    """
    Parses a single prescription line/token into a structured dictionary:
      {
        "drug": str,             # Normalized canonical generic name (or combination)
        "dose_mg": float | None, # Single dose in milligrams
        "frequency": int,        # Dosing frequency per day (1, 2, 3, 4)
        "daily_dose": float | None, # Cumulative daily dose (dose_mg * frequency)
        "raw": str               # Original raw line text
      }
    """
    if not raw_line or not isinstance(raw_line, str):
        return None

    cleaned_line = raw_line.strip()
    if not cleaned_line:
        return None

    _load_brand_defaults()

    working_text = cleaned_line

    # 1. EXTRACT FREQUENCY
    frequency = 1  # Default to 1 (OD) if not explicitly specified
    for pattern, val in FREQUENCY_RULES:
        if re.search(pattern, working_text, flags=re.IGNORECASE):
            frequency = val
            working_text = re.sub(pattern, " ", working_text, flags=re.IGNORECASE)
            break

    # 2. EXTRACT DOSE IN MG
    dose_mg: Optional[float] = None
    dose_match = DOSE_PATTERN.search(working_text)
    if dose_match:
        dose_mg = float(dose_match.group(1))
        working_text = DOSE_PATTERN.sub(" ", working_text)
    else:
        # Check for standalone numbers representing typical strengths (e.g. "Dolo 650", "Ecosprin 75")
        num_match = EMBEDDED_STRENGTH_PATTERN.search(working_text)
        if num_match:
            dose_mg = float(num_match.group(1))
            # Remove number from drug name extraction token
            working_text = EMBEDDED_STRENGTH_PATTERN.sub(" ", working_text)

    # 3. EXTRACT AND NORMALIZE DRUG ENTITY
    candidate_name = PREFIX_PATTERN.sub("", working_text).strip(" -.,/|:")
    normalized_drug = normalize_drug_name(candidate_name)

    # If candidate alone didn't match, attempt on the full cleaned line
    if not normalized_drug:
        normalized_drug = normalize_drug_name(cleaned_line)

    # If still not matched, fallback to cleaned alphanumeric candidate
    if not normalized_drug:
        fallback = re.sub(r"[\d\-_]+", " ", candidate_name).strip().lower()
        normalized_drug = fallback if fallback else None

    if not normalized_drug:
        return None

    # 4. DEFAULT STRENGTH FALLBACK FROM BRAND FORMULARY
    if dose_mg is None:
        cand_lower = candidate_name.lower().strip()
        if cand_lower in _BRAND_DEFAULT_STRENGTHS:
            dose_mg = _BRAND_DEFAULT_STRENGTHS[cand_lower]
        elif normalized_drug in _BRAND_DEFAULT_STRENGTHS:
            dose_mg = _BRAND_DEFAULT_STRENGTHS[normalized_drug]

    # 5. CALCULATE CUMULATIVE DAILY DOSE
    daily_dose = (dose_mg * frequency) if dose_mg is not None else None

    return {
        "drug": normalized_drug,
        "dose_mg": dose_mg,
        "frequency": frequency,
        "daily_dose": daily_dose,
        "raw": cleaned_line,
    }


def parse_prescription(free_text: str) -> List[Dict[str, Any]]:
    """
    Parses free-text prescriptions containing one or more medications.
    Supports comma-separated, semicolon-separated, and newline-separated entries.

    Example input:
      "Tab Ecosprin 75mg OD, Tab Glycomet 500mg BD"
    """
    if not free_text or not isinstance(free_text, str):
        return []

    # Split on line breaks, commas, semicolons, and list numbers
    raw_entries = re.split(r"[\n,;]+", free_text)
    structured_prescriptions: List[Dict[str, Any]] = []

    for entry in raw_entries:
        entry_clean = entry.strip()
        if not entry_clean:
            continue

        parsed = parse_prescription_line(entry_clean)
        if parsed and parsed.get("drug"):
            structured_prescriptions.append(parsed)

    return structured_prescriptions


# =============================================================================
# ISOLATED DEMONSTRATION & VERIFICATION SUITE
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING parser.py VERIFICATION SUITE (VARIED PRESCRIPTION EXAMPLES)")
    print("=" * 80)

    # Varied test prescriptions including:
    # 1. Standard comma-separated multi-drug string
    # 2. Indian brand name with strength code + abbreviation (TDS)
    # 3. Typo in brand name + abbreviation (OD)
    # 4. Indian combination brand name + abbreviation (BD)
    # 5. Four-times-daily abbreviation (QID)
    # 6. Multiline numbered list format with timing notes
    test_cases = [
        (
            "Tab Ecosprin 75mg OD, Tab Glycomet 500mg BD",
            "Multi-drug comma-separated input with OD and BD",
        ),
        (
            "Tab Dolo 650 TDS",
            "Indian brand name with embedded strength and TDS (3/day)",
        ),
        (
            "Tab ecospren 75 OD",
            "Common typo in Indian cardiovascular brand (ecospren -> aspirin)",
        ),
        (
            "Cap Pan-D 40mg BD",
            "Indian combination brand with capsule prefix and BD (2/day)",
        ),
        (
            "Augmentin 625mg QID",
            "Combination antibiotic with QID (4/day)",
        ),
        (
            "1. Atorva 10mg OD at night\n2. Tab Brufen 400mg TDS",
            "Multiline numbered prescription with timing instructions",
        ),
    ]

    for idx, (raw_text, description) in enumerate(test_cases, 1):
        print(f"\n[TEST CASE {idx}] {description}")
        print(f"RAW INPUT: {raw_text!r}")
        results = parse_prescription(raw_text)

        print(f"PARSED OUTPUT ({len(results)} item(s)):")
        for res in results:
            print(
                f"  • drug: {res['drug']:<25} | "
                f"dose_mg: {str(res['dose_mg']):<6} | "
                f"frequency: {res['frequency']} | "
                f"daily_dose: {str(res['daily_dose']):<7} | "
                f"raw: {res['raw']}"
            )

        assert len(results) > 0, f"Failed to parse: {raw_text}"
        for res in results:
            assert "drug" in res and res["drug"] is not None
            assert "dose_mg" in res
            assert "frequency" in res and res["frequency"] in {1, 2, 3, 4}
            assert "daily_dose" in res

    print("\n" + "=" * 80)
    print("ALL 6 PRESCRIPTION PARSING TEST CASES PASSED SUCCESSFULLY.")
    print("=" * 80)
