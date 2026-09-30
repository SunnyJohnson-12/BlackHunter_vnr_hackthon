"""
checkers.py - Clinical Safety Engine for VNR Hackathon
Implements explainable clinical safety checkers for:
  1. Drug-Drug Interactions (DDI)
  2. Drug-Disease Contraindications
  3. Duplicate Therapy (Active ingredient & Pharmacologic class)
  4. Dosage Limits (Single dose & Cumulative 24h adult limits)
  5. Allergy Conflicts & Cross-Reactivity
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx

from graph import (
    build_clinical_graph,
    get_allergy_cross_reactions,
    get_contraindications,
    get_drug_classes,
    get_drug_interaction,
)
from normalizer import normalize_drug_name
from parser import parse_prescription_line

BASE_DIR = Path(__file__).resolve().parent


# =============================================================================
# CORE ALERT DATACLASS
# =============================================================================
@dataclass
class Alert:
    type: str  # "drug-drug" | "drug-disease" | "duplicate" | "dosage" | "allergy"
    severity: str  # "major" | "moderate" | "minor"
    drugs: list[str]
    reason: str
    action: str


# =============================================================================
# GLOBAL / LAZY RESOURCE MANAGEMENT
# =============================================================================
_GRAPH: Optional[nx.DiGraph] = None
_DOSE_LIMITS: Dict[str, Dict[str, Any]] = {}
_INITIALIZED: bool = False

# Common clinical condition aliases mapping to canonical condition graph keys
CONDITION_ALIASES = {
    "ckd": "chronic kidney disease",
    "chronic kidney disease": "chronic kidney disease",
    "renal disease": "chronic kidney disease",
    "renal impairment": "chronic kidney disease",
    "renal failure": "chronic kidney disease",
    "kidney disease": "chronic kidney disease",
    "asthma": "asthma",
    "bronchial asthma": "asthma",
    "peptic ulcer disease": "peptic ulcer disease",
    "peptic ulcer": "peptic ulcer disease",
    "pud": "peptic ulcer disease",
    "gastric ulcer": "peptic ulcer disease",
    "heart failure": "heart failure",
    "chf": "heart failure",
    "congestive heart failure": "heart failure",
    "severe cardiovascular disease": "severe cardiovascular disease",
    "cad": "severe cardiovascular disease",
    "cardiovascular disease": "severe cardiovascular disease",
    "active liver disease": "active liver disease",
    "liver disease": "active liver disease",
    "cirrhosis": "active liver disease",
    "hepatic impairment": "active liver disease",
    "myasthenia gravis": "myasthenia gravis",
    "epilepsy": "epilepsy",
    "seizures": "epilepsy",
    "seizure disorder": "epilepsy",
    "severe hypertension": "severe hypertension",
    "hypertension": "severe hypertension",
    "htn": "severe hypertension",
    "pregnancy": "pregnancy",
    "pregnant": "pregnancy",
    "hyperkalemia": "hyperkalemia",
    "bilateral renal artery stenosis": "bilateral renal artery stenosis",
    "severe hypokalemia": "severe hypokalemia",
    "hypokalemia": "severe hypokalemia",
}


def _get_or_init_graph() -> nx.DiGraph:
    """Returns initialized clinical knowledge graph singleton."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_clinical_graph()
    return _GRAPH


def _load_dose_limits() -> None:
    """Loads maximum adult single and daily dose limits from dose_limits.csv."""
    global _DOSE_LIMITS, _INITIALIZED
    if _INITIALIZED:
        return

    csv_path = BASE_DIR / "dose_limits.csv"
    if not csv_path.exists():
        csv_path = Path("dose_limits.csv")

    if csv_path.exists():
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(row for row in f if not row.startswith("#"))
            for row in reader:
                drug = row["drug_name"].strip().lower()
                try:
                    _DOSE_LIMITS[drug] = {
                        "max_single_dose_mg": float(row["max_single_dose_mg"]),
                        "max_daily_dose_mg": float(row["max_daily_dose_mg"]),
                        "unit": row.get("unit", "mg").strip(),
                        "overdose_risk": row.get("overdose_risk", "").strip(),
                        "source": row.get("source", "FDA / WHO Formulary").strip(),
                    }
                except (ValueError, KeyError):
                    continue

    _INITIALIZED = True


# =============================================================================
# INPUT PARSING & NORMALIZATION HELPERS
# =============================================================================
def _parse_drug_info(
    drug_input: Union[str, Dict[str, Any]]
) -> Tuple[str, List[str], Optional[float], Optional[float]]:
    """
    Parses a drug representation (string or dict) into:
      1. raw_name: original display name
      2. ingredients: list of canonical generic ingredients (split on '+')
      3. single_dose_mg: float single dose in mg, if provided
      4. daily_dose_mg: float cumulative daily dose in mg, if provided
    """
    raw_name = ""
    single_dose: Optional[float] = None
    daily_dose: Optional[float] = None
    norm: Optional[str] = None

    if isinstance(drug_input, dict):
        raw_name = str(drug_input.get("raw") or drug_input.get("name") or drug_input.get("drug") or "")
        # Dose fields
        if "single_dose_mg" in drug_input and drug_input["single_dose_mg"] is not None:
            single_dose = float(drug_input["single_dose_mg"])
        elif "dose_mg" in drug_input and drug_input["dose_mg"] is not None:
            single_dose = float(drug_input["dose_mg"])

        freq = float(drug_input.get("frequency") or drug_input.get("frequency_per_day") or 1)
        if "daily_dose_mg" in drug_input and drug_input["daily_dose_mg"] is not None:
            daily_dose = float(drug_input["daily_dose_mg"])
        elif single_dose is not None:
            daily_dose = single_dose * freq

        drug_val = str(drug_input.get("drug") or drug_input.get("name") or "")
        norm = normalize_drug_name(drug_val) or normalize_drug_name(raw_name)
    else:
        raw_name = str(drug_input).strip()
        parsed = parse_prescription_line(raw_name)
        if parsed and parsed.get("drug"):
            single_dose = parsed.get("dose_mg")
            daily_dose = parsed.get("daily_dose")
            norm = parsed.get("drug")
        else:
            match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mg)?\b", raw_name, re.IGNORECASE)
            if match and "mg" in raw_name.lower():
                single_dose = float(match.group(1))
            norm = normalize_drug_name(raw_name)

    # Normalization
    if norm:
        ingredients = [ing.strip().lower() for ing in norm.split("+") if ing.strip()]
    else:
        # Fallback to cleaned raw name
        cleaned = re.sub(r"[\d\-_]+", " ", raw_name).strip().lower()
        ingredients = [cleaned] if cleaned else []

    return raw_name, ingredients, single_dose, daily_dose


# =============================================================================
# CHECKER 1: DRUG-DRUG INTERACTIONS
# =============================================================================
def check_drug_drug(
    patient: Dict[str, Any],
    new_drug: Union[str, Dict[str, Any]],
    graph: Optional[nx.DiGraph] = None,
) -> List[Alert]:
    """
    Checks for drug-drug interactions between new_drug and patient's current_meds.
    Returns explainable Alerts with severity, pharmacological mechanism, clinical risk,
    and prescriber action.
    """
    G = graph or _get_or_init_graph()
    alerts: List[Alert] = []

    _, new_ingredients, _, _ = _parse_drug_info(new_drug)
    current_meds = patient.get("current_meds", [])

    for cur_med in current_meds:
        _, cur_ingredients, _, _ = _parse_drug_info(cur_med)
        for d_new in new_ingredients:
            for d_cur in cur_ingredients:
                if d_new == d_cur:
                    continue  # Handled in duplicate therapy check

                interaction = get_drug_interaction(G, d_new, d_cur)
                if interaction:
                    severity = interaction.get("severity", "moderate").lower()
                    if severity not in {"major", "moderate", "minor"}:
                        severity = "moderate"

                    mechanism = interaction.get("mechanism", "Pharmacodynamic or pharmacokinetic synergy.")
                    risk = interaction.get("clinical_risk", "Heightened adverse event risk.")
                    action = interaction.get("action", "Monitor patient or consider alternative.")

                    reason = (
                        f"Dangerous Drug-Drug Interaction: {d_cur.capitalize()} + {d_new.capitalize()}.\n"
                        f"• Mechanism: {mechanism}\n"
                        f"• Clinical Risk: {risk}"
                    )

                    alerts.append(
                        Alert(
                            type="drug-drug",
                            severity=severity,
                            drugs=[d_cur, d_new],
                            reason=reason,
                            action=action,
                        )
                    )

    return alerts


# =============================================================================
# CHECKER 2: DRUG-DISEASE CONTRAINDICATIONS
# =============================================================================
def check_drug_disease(
    patient: Dict[str, Any],
    new_drug: Union[str, Dict[str, Any]],
    graph: Optional[nx.DiGraph] = None,
) -> List[Alert]:
    """
    Identifies absolute and relative contraindications between new_drug and
    the patient's diagnosed clinical conditions.
    """
    G = graph or _get_or_init_graph()
    alerts: List[Alert] = []

    _, new_ingredients, _, _ = _parse_drug_info(new_drug)
    conditions = patient.get("conditions", [])

    for cond in conditions:
        cond_raw = str(cond).strip().lower()
        canonical_cond = CONDITION_ALIASES.get(cond_raw, cond_raw)

        for d_new in new_ingredients:
            contra = get_contraindications(G, d_new, canonical_cond)
            if not contra:
                # Also try matching against condition nodes by substring
                for node, data in G.nodes(data=True):
                    if data.get("node_type") == "condition":
                        cond_name = data.get("name_lower", "")
                        if cond_raw in cond_name or cond_name in cond_raw:
                            contra = get_contraindications(G, d_new, cond_name)
                            if contra:
                                break

            if contra:
                raw_sev = contra.get("severity", "absolute").lower()
                # Map absolute -> major, relative -> moderate
                severity = "major" if raw_sev == "absolute" else "moderate"
                condition_display = contra.get("condition", cond)
                mechanism = contra.get("mechanism", "Pathophysiological conflict.")
                risk = contra.get("clinical_risk", "Exacerbation of pre-existing condition.")
                action = contra.get("action", f"Avoid or adjust dose in {condition_display}.")

                reason = (
                    f"Drug-Disease Contraindication ({raw_sev.upper()}): "
                    f"'{d_new.capitalize()}' in patient with '{condition_display}'.\n"
                    f"• Pathophysiology: {mechanism}\n"
                    f"• Clinical Consequence: {risk}"
                )

                alerts.append(
                    Alert(
                        type="drug-disease",
                        severity=severity,
                        drugs=[d_new],
                        reason=reason,
                        action=action,
                    )
                )

    return alerts


# =============================================================================
# CHECKER 3: DUPLICATE THERAPY DETECTION
# =============================================================================
def check_duplicate_therapy(
    patient: Dict[str, Any],
    new_drug: Union[str, Dict[str, Any]],
    graph: Optional[nx.DiGraph] = None,
) -> List[Alert]:
    """
    Detects two tiers of duplication:
      1. Direct active ingredient duplication (e.g., Dolo + Calpol, both paracetamol)
      2. Pharmacological class duplication (e.g., Ibuprofen + Naproxen, both NSAIDs)
    """
    G = graph or _get_or_init_graph()
    alerts: List[Alert] = []

    raw_new_name, new_ingredients, _, _ = _parse_drug_info(new_drug)
    current_meds = patient.get("current_meds", [])

    for cur_med in current_meds:
        raw_cur_name, cur_ingredients, _, _ = _parse_drug_info(cur_med)

        # Tier 1: Exact active ingredient overlap
        for d_new in new_ingredients:
            for d_cur in cur_ingredients:
                if d_new == d_cur:
                    reason = (
                        f"Direct Duplicate Therapy: Identical active ingredient '{d_new.capitalize()}' "
                        f"is already active in patient's current regimen ('{raw_cur_name}'). "
                        f"Concomitant prescription causes inadvertent cumulative overdose."
                    )
                    action = (
                        f"Discontinue redundant prescription of '{raw_new_name}' or verify "
                        f"whether intentional dose adjustment is intended."
                    )
                    alerts.append(
                        Alert(
                            type="duplicate",
                            severity="major",
                            drugs=[d_cur, d_new],
                            reason=reason,
                            action=action,
                        )
                    )
                    continue

                # Tier 2: Pharmacological class duplication (different drugs, same class)
                new_classes = {c["class_name"] for c in get_drug_classes(G, d_new)}
                cur_classes = {c["class_name"] for c in get_drug_classes(G, d_cur)}
                shared_classes = new_classes.intersection(cur_classes)

                for s_class in shared_classes:
                    reason = (
                        f"Therapeutic Class Duplication: Both '{d_cur.capitalize()}' and "
                        f"'{d_new.capitalize()}' belong to the '{s_class}' pharmacological class. "
                        f"Prescribing multiple agents from this class increases adverse toxicity "
                        f"without providing added therapeutic efficacy."
                    )
                    action = (
                        f"Avoid concurrent co-prescription of two {s_class} agents. "
                        f"Select a single agent optimized for patient tolerance."
                    )
                    alerts.append(
                        Alert(
                            type="duplicate",
                            severity="moderate",
                            drugs=[d_cur, d_new],
                            reason=reason,
                            action=action,
                        )
                    )

    return alerts


# =============================================================================
# CHECKER 4: DOSAGE SAFETY LIMITS
# =============================================================================
def check_dosage(
    patient: Dict[str, Any],
    new_drug: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]],
    graph: Optional[nx.DiGraph] = None,
) -> List[Alert]:
    """
    Evaluates new_drug against adult maximum single dose and maximum 24h daily dose.
    When multiple drugs are prescribed together (as a list), pre-aggregates the 24-hour
    cumulative daily dose of matching active ingredients across all prescribed items
    before comparing against monograph ceilings.
    Also factors in cumulative dosage if the same active ingredient is in patient's current_meds.
    """
    _load_dose_limits()
    alerts: List[Alert] = []

    drug_items = new_drug if isinstance(new_drug, list) else [new_drug]
    if not drug_items:
        return []

    current_meds = patient.get("current_meds", [])

    # Step 1: Pre-aggregate daily doses and check single-dose limits across new prescription
    new_ingredient_daily: Dict[str, float] = {}
    new_ingredient_items: Dict[str, List[Tuple[str, float]]] = {}
    flagged_single_dose: set[str] = set()

    for item in drug_items:
        raw_name, ingredients, single_dose, daily_dose = _parse_drug_info(item)
        for ing in ingredients:
            limit = _DOSE_LIMITS.get(ing)
            if not limit:
                continue

            max_single = limit["max_single_dose_mg"]
            unit = limit["unit"]
            risk = limit["overdose_risk"]

            # 1. Single Dose Check for this item
            if single_dose is not None and single_dose > max_single:
                item_key = f"{ing}_{single_dose}"
                if item_key not in flagged_single_dose:
                    flagged_single_dose.add(item_key)
                    display_name = raw_name if raw_name else ing.capitalize()
                    reason = (
                        f"Excessive Single Dose: Prescribed single dose of {ing.capitalize()} in '{display_name}' "
                        f"({single_dose} {unit}) exceeds the maximum adult single-dose ceiling "
                        f"({max_single} {unit}).\n"
                        f"• Clinical Danger: {risk}"
                    )
                    action = f"Reduce single dose of {ing.capitalize()} to <= {max_single} {unit}."
                    alerts.append(
                        Alert(
                            type="dosage",
                            severity="major",
                            drugs=[ing],
                            reason=reason,
                            action=action,
                        )
                    )

            # Accumulate daily dose for this ingredient
            if daily_dose is not None:
                new_ingredient_daily[ing] = new_ingredient_daily.get(ing, 0.0) + daily_dose
                if ing not in new_ingredient_items:
                    new_ingredient_items[ing] = []
                new_ingredient_items[ing].append((raw_name or ing, daily_dose))

    # Step 2: Sum current medications' daily doses by ingredient
    current_ingredient_daily: Dict[str, float] = {}
    current_ingredient_items: Dict[str, List[Tuple[str, float]]] = {}

    for cur_med in current_meds:
        raw_cur, cur_ings, cur_single, cur_daily = _parse_drug_info(cur_med)
        for c_ing in cur_ings:
            if cur_daily is not None:
                current_ingredient_daily[c_ing] = current_ingredient_daily.get(c_ing, 0.0) + cur_daily
                if c_ing not in current_ingredient_items:
                    current_ingredient_items[c_ing] = []
                current_ingredient_items[c_ing].append((raw_cur or c_ing, cur_daily))

    # Step 3: Check cumulative 24-hour daily exposure for each prescribed ingredient
    for ing, new_daily in new_ingredient_daily.items():
        limit = _DOSE_LIMITS.get(ing)
        if not limit:
            continue

        max_daily = limit["max_daily_dose_mg"]
        unit = limit["unit"]
        risk = limit["overdose_risk"]

        cur_daily = current_ingredient_daily.get(ing, 0.0)
        total_daily = new_daily + cur_daily

        if total_daily > max_daily:
            items_list = new_ingredient_items.get(ing, [])
            cur_items_list = current_ingredient_items.get(ing, [])

            if len(items_list) > 1 and cur_daily > 0:
                new_breakdown = " + ".join([f"'{itm}' ({d} {unit}/d)" for itm, d in items_list])
                cur_breakdown = " + ".join([f"'{itm}' ({d} {unit}/d)" for itm, d in cur_items_list])
                reason = (
                    f"Excessive 24-Hour Cumulative Dose: Co-prescribing multiple {ing.capitalize()} formulations "
                    f"({new_daily} {unit}/day from {new_breakdown}) combined with active regimen "
                    f"({cur_daily} {unit}/day from {cur_breakdown}) totals {total_daily} {unit}/day, "
                    f"exceeding the maximum safe adult daily ceiling ({max_daily} {unit}/day).\n"
                    f"• Clinical Danger: {risk}"
                )
            elif len(items_list) > 1:
                new_breakdown = " + ".join([f"'{itm}' ({d} {unit}/d)" for itm, d in items_list])
                reason = (
                    f"Excessive 24-Hour Cumulative Dose: Co-prescribing multiple {ing.capitalize()} formulations "
                    f"({new_breakdown}) totals {total_daily} {unit}/day in the new prescription, "
                    f"exceeding the maximum safe adult daily ceiling ({max_daily} {unit}/day).\n"
                    f"• Clinical Danger: {risk}"
                )
            elif cur_daily > 0:
                cur_breakdown = " + ".join([f"'{itm}' ({d} {unit}/d)" for itm, d in cur_items_list])
                reason = (
                    f"Excessive 24-Hour Cumulative Dose: New prescription of {ing.capitalize()} ({new_daily} {unit}/day) "
                    f"plus active regimen ({cur_daily} {unit}/day from {cur_breakdown}) totals {total_daily} {unit}/day, "
                    f"exceeding the maximum safe adult daily ceiling ({max_daily} {unit}/day).\n"
                    f"• Clinical Danger: {risk}"
                )
            else:
                reason = (
                    f"Excessive 24-Hour Cumulative Dose: Total daily exposure to {ing.capitalize()} "
                    f"({total_daily} {unit}/day) exceeds maximum safe adult limit ({max_daily} {unit}/day).\n"
                    f"• Clinical Danger: {risk}"
                )

            action = f"Cap total daily cumulative dosage of {ing.capitalize()} across all prescriptions to <= {max_daily} {unit}/day immediately."
            alerts.append(
                Alert(
                    type="dosage",
                    severity="major",
                    drugs=[ing],
                    reason=reason,
                    action=action,
                )
            )

    return alerts


# =============================================================================
# CHECKER 5: ALLERGY CONFLICTS & CROSS-REACTIVITY
# =============================================================================
def check_allergy(
    patient: Dict[str, Any],
    new_drug: Union[str, Dict[str, Any]],
    graph: Optional[nx.DiGraph] = None,
) -> List[Alert]:
    """
    Checks for direct allergy conflicts and immunological cross-reactivity
    against the patient's recorded allergies.
    """
    G = graph or _get_or_init_graph()
    alerts: List[Alert] = []

    _, new_ingredients, _, _ = _parse_drug_info(new_drug)
    allergies = patient.get("allergies", [])

    for allergy in allergies:
        alg_raw = str(allergy).strip().lower()
        if not alg_raw:
            continue

        # Check normalization of allergen string
        alg_norm = normalize_drug_name(alg_raw) or alg_raw

        for d_new in new_ingredients:
            # Case 1: Direct match (patient allergic to the prescribed substance)
            if alg_raw == d_new or alg_norm == d_new or alg_raw in d_new or d_new in alg_raw:
                reason = (
                    f"Direct Allergy Conflict: Patient has a documented hypersensitivity to '{allergy}'. "
                    f"Prescribed drug '{d_new.capitalize()}' is a direct immunological match."
                )
                action = (
                    f"Absolute contraindication. Withhold '{d_new.capitalize()}'. "
                    f"Prescribe an alternative from an unrelated chemical class."
                )
                alerts.append(
                    Alert(
                        type="allergy",
                        severity="major",
                        drugs=[d_new],
                        reason=reason,
                        action=action,
                    )
                )
                continue

            # Case 2: Cross-reactivity edge in knowledge graph
            cross_rxn = get_allergy_cross_reactions(G, alg_raw, d_new)
            if not cross_rxn and alg_norm != alg_raw:
                cross_rxn = get_allergy_cross_reactions(G, alg_norm, d_new)

            if cross_rxn:
                severity = cross_rxn.get("severity", "major").lower()
                if severity not in {"major", "moderate", "minor"}:
                    severity = "major"

                mechanism = cross_rxn.get("mechanism", "Immunological or epitope cross-reactivity.")
                risk = cross_rxn.get("clinical_risk", "Severe hypersensitivity / anaphylaxis risk.")
                action = cross_rxn.get("action", "Avoid cross-reactive drug.")

                reason = (
                    f"Allergy Cross-Reactivity Risk: Patient has documented allergy to '{allergy}'. "
                    f"Prescribed drug '{d_new.capitalize()}' exhibits known cross-sensitivity.\n"
                    f"• Mechanism: {mechanism}\n"
                    f"• Clinical Risk: {risk}"
                )

                alerts.append(
                    Alert(
                        type="allergy",
                        severity=severity,
                        drugs=[d_new],
                        reason=reason,
                        action=action,
                    )
                )

    return alerts


# =============================================================================
# UNIFIED SAFETY SCANNER
# =============================================================================
def check_all_safety_rules(
    patient: Dict[str, Any],
    new_drug: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]],
    graph: Optional[nx.DiGraph] = None,
) -> List[Alert]:
    """
    Runs all 5 safety checkers sequentially and aggregates identified alerts.
    Supports either an individual drug item or a list of prescribed items.
    """
    G = graph or _get_or_init_graph()
    all_alerts: List[Alert] = []

    if isinstance(new_drug, list):
        for item in new_drug:
            all_alerts.extend(check_drug_drug(patient, item, graph=G))
            all_alerts.extend(check_drug_disease(patient, item, graph=G))
            all_alerts.extend(check_duplicate_therapy(patient, item, graph=G))
            all_alerts.extend(check_allergy(patient, item, graph=G))
        all_alerts.extend(check_dosage(patient, new_drug, graph=G))
    else:
        all_alerts.extend(check_drug_drug(patient, new_drug, graph=G))
        all_alerts.extend(check_drug_disease(patient, new_drug, graph=G))
        all_alerts.extend(check_duplicate_therapy(patient, new_drug, graph=G))
        all_alerts.extend(check_dosage(patient, new_drug, graph=G))
        all_alerts.extend(check_allergy(patient, new_drug, graph=G))

    return all_alerts


# =============================================================================
# ISOLATED VERIFICATION TEST SUITE
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING checkers.py SAFETY RULE VERIFICATION SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TEST 1: Drug-Drug Interaction Checker (warfarin + aspirin)
    # -------------------------------------------------------------------------
    p1 = {
        "age": 65,
        "conditions": ["Atrial Fibrillation"],
        "allergies": [],
        "current_meds": ["warfarin"],
    }
    alerts_ddi = check_drug_drug(p1, "aspirin")
    assert len(alerts_ddi) > 0 and alerts_ddi[0].type == "drug-drug"
    print(f"[PASS] 1. Drug-Drug Checker fired ({len(alerts_ddi)} alert):")
    print(f"       Type: {alerts_ddi[0].type} | Severity: {alerts_ddi[0].severity} | Drugs: {alerts_ddi[0].drugs}")
    print(f"       Action: {alerts_ddi[0].action[:70]}...")

    # -------------------------------------------------------------------------
    # TEST 2: Drug-Disease Contraindication Checker (propranolol in Asthma)
    # -------------------------------------------------------------------------
    p2 = {
        "age": 42,
        "conditions": ["Asthma"],
        "allergies": [],
        "current_meds": [],
    }
    alerts_contra = check_drug_disease(p2, "propranolol")
    assert len(alerts_contra) > 0 and alerts_contra[0].type == "drug-disease"
    print(f"\n[PASS] 2. Drug-Disease Checker fired ({len(alerts_contra)} alert):")
    print(f"       Type: {alerts_contra[0].type} | Severity: {alerts_contra[0].severity} | Drugs: {alerts_contra[0].drugs}")
    print(f"       Action: {alerts_contra[0].action[:70]}...")

    # -------------------------------------------------------------------------
    # TEST 3: Duplicate Therapy Checker (ibuprofen + naproxen, both NSAIDs)
    # -------------------------------------------------------------------------
    p3 = {
        "age": 50,
        "conditions": ["Osteoarthritis"],
        "allergies": [],
        "current_meds": ["ibuprofen"],
    }
    alerts_dup = check_duplicate_therapy(p3, "naproxen")
    assert len(alerts_dup) > 0 and alerts_dup[0].type == "duplicate"
    print(f"\n[PASS] 3. Duplicate Therapy Checker fired ({len(alerts_dup)} alert):")
    print(f"       Type: {alerts_dup[0].type} | Severity: {alerts_dup[0].severity} | Drugs: {alerts_dup[0].drugs}")
    print(f"       Action: {alerts_dup[0].action[:70]}...")

    # -------------------------------------------------------------------------
    # TEST 4: Dosage Safety Checker (paracetamol 1500mg single, 6000mg/day)
    # -------------------------------------------------------------------------
    p4 = {
        "age": 30,
        "conditions": ["Fever"],
        "allergies": [],
        "current_meds": [],
    }
    alerts_dose = check_dosage(
        p4,
        {"name": "paracetamol", "single_dose_mg": 1500, "daily_dose_mg": 6000},
    )
    assert len(alerts_dose) > 0 and alerts_dose[0].type == "dosage"
    print(f"\n[PASS] 4. Dosage Safety Checker fired ({len(alerts_dose)} alerts):")
    for a in alerts_dose:
        print(f"       Type: {a.type} | Severity: {a.severity} | Drugs: {a.drugs}")
        print(f"       Action: {a.action[:70]}...")

    # -------------------------------------------------------------------------
    # TEST 5: Allergy Conflict & Cross-Reactivity Checker (penicillin -> amoxicillin)
    # -------------------------------------------------------------------------
    p5 = {
        "age": 28,
        "conditions": ["Bacterial Pharyngitis"],
        "allergies": ["penicillin"],
        "current_meds": [],
    }
    alerts_alg = check_allergy(p5, "amoxicillin")
    assert len(alerts_alg) > 0 and alerts_alg[0].type == "allergy"
    print(f"\n[PASS] 5. Allergy Checker fired ({len(alerts_alg)} alert):")
    print(f"       Type: {alerts_alg[0].type} | Severity: {alerts_alg[0].severity} | Drugs: {alerts_alg[0].drugs}")
    print(f"       Action: {alerts_alg[0].action[:70]}...")

    # -------------------------------------------------------------------------
    # TEST 6: Clean Patient Case (ZERO Alerts across ALL checkers)
    # -------------------------------------------------------------------------
    p6_clean = {
        "age": 35,
        "conditions": ["Mild Dyspepsia"],
        "allergies": ["dust mites"],
        "current_meds": ["pantoprazole"],
    }
    clean_new_drug = {"name": "paracetamol", "single_dose_mg": 650, "daily_dose_mg": 1300}
    clean_alerts = check_all_safety_rules(p6_clean, clean_new_drug)

    assert len(clean_alerts) == 0, f"Expected 0 alerts for clean patient, got {len(clean_alerts)}"
    print(f"\n[PASS] 6. Clean Patient Case: Exactly {len(clean_alerts)} alerts produced across all 5 checkers.")

    # -------------------------------------------------------------------------
    # TEST 7: Intra-prescription Cumulative Dosage Check (Dolo 650 TDS + Calpol 500 BD)
    # -------------------------------------------------------------------------
    p7 = {
        "age": 52,
        "conditions": ["Fever"],
        "allergies": [],
        "current_meds": ["Crocin 650mg BD"],  # 1300 mg/day
    }
    multi_items = [
        {"raw": "Tab Dolo 650 TDS", "drug": "paracetamol", "dose_mg": 650.0, "frequency": 3, "daily_dose": 1950.0},
        {"raw": "Tab Calpol 500 BD", "drug": "paracetamol", "dose_mg": 500.0, "frequency": 2, "daily_dose": 1000.0},
    ]
    alerts_multi = check_dosage(p7, multi_items)
    assert len(alerts_multi) > 0 and alerts_multi[0].type == "dosage"
    assert "4250.0" in alerts_multi[0].reason
    print(f"\n[PASS] 7. Intra-prescription Cumulative Dosage Check fired ({len(alerts_multi)} alert):")
    print(f"       Type: {alerts_multi[0].type} | Severity: {alerts_multi[0].severity} | Drugs: {alerts_multi[0].drugs}")
    print(f"       Reason: {alerts_multi[0].reason[:110]}...")

    print("=" * 80)
    print("ALL 7 CLINICAL SAFETY CHECKERS & MULTI-DOSE TESTS VERIFIED SUCCESSFULLY.")
    print("=" * 80)
