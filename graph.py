"""
graph.py - Clinical Knowledge Graph Builder for VNR Hackathon
Constructs an explainable NetworkX clinical knowledge graph from local CSV datasets:
  - Nodes: drugs, conditions, allergens, drug_classes
  - Typed edges: interacts_with, belongs_to_class, contraindicated_in, cross_reacts_with
  - Attributes: severity, mechanism, clinical_risk, action, source
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

BASE_DIR = Path(__file__).resolve().parent


def _resolve_path(filename: str, base_dir: Optional[Path | str] = None) -> Path:
    """Resolves path to dataset file, falling back to local directory."""
    root = Path(base_dir) if base_dir else BASE_DIR
    target = root / filename
    if not target.exists():
        target = Path(filename)
    return target


def build_clinical_graph(data_dir: Optional[Path | str] = None) -> nx.DiGraph:
    """
    Constructs and returns the directed clinical knowledge graph.

    Data Provenance:
      - interactions.csv: Curated clinical pharmacology guidelines & FDA advisories
      - drug_classes.csv: WHO ATC classification & pharmacological classes
      - contraindications.csv: FDA Boxed Warnings & clinical disease monographs
      - allergies.csv: Clinical immunology practice guidelines & FDA alerts
    """
    G = nx.DiGraph()

    # -------------------------------------------------------------------------
    # 1. DRUG-DRUG INTERACTIONS (interacts_with)
    # -------------------------------------------------------------------------
    interactions_file = _resolve_path("interactions.csv", data_dir)
    if interactions_file.exists():
        with open(interactions_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(row for row in f if not row.startswith("#"))
            for row in reader:
                da = row["drug_a"].strip().lower()
                db = row["drug_b"].strip().lower()
                if not da or not db:
                    continue

                G.add_node(da, node_type="drug", name=da)
                G.add_node(db, node_type="drug", name=db)

                edge_attrs = {
                    "edge_type": "interacts_with",
                    "severity": row.get("severity", "moderate").strip().lower(),
                    "mechanism": row.get("mechanism", "").strip(),
                    "clinical_risk": row.get("clinical_risk", "").strip(),
                    "action": row.get("recommended_action", "").strip(),
                    "source": row.get("source", "Curated clinical pharmacology").strip(),
                }
                # Interactions are bidirectional: add both directions for O(1) lookup
                G.add_edge(da, db, **edge_attrs)
                G.add_edge(db, da, **edge_attrs)

    # -------------------------------------------------------------------------
    # 2. DRUG CLASSES & THERAPEUTIC CATEGORIES (belongs_to_class)
    # -------------------------------------------------------------------------
    classes_file = _resolve_path("drug_classes.csv", data_dir)
    if classes_file.exists():
        with open(classes_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(row for row in f if not row.startswith("#"))
            for row in reader:
                drug = row["drug_name"].strip().lower()
                p_class = row["pharmacologic_class"].strip()
                category = row["therapeutic_category"].strip()
                if not drug or not p_class:
                    continue

                class_node = f"class:{p_class}"
                if not G.has_node(drug):
                    G.add_node(drug, node_type="drug", name=drug)

                G.add_node(
                    class_node,
                    node_type="drug_class",
                    name=p_class,
                    therapeutic_category=category,
                )

                G.add_edge(
                    drug,
                    class_node,
                    edge_type="belongs_to_class",
                    severity="info",
                    mechanism=f"Pharmacological class: {p_class} ({category})",
                    action="Check co-prescriptions for therapeutic or pharmacologic duplication",
                    class_name=p_class,
                    therapeutic_category=category,
                    source=row.get("source", "WHO ATC / Curated").strip(),
                )

    # -------------------------------------------------------------------------
    # 3. DRUG-DISEASE CONTRAINDICATIONS (contraindicated_in)
    # -------------------------------------------------------------------------
    contra_file = _resolve_path("contraindications.csv", data_dir)
    if contra_file.exists():
        with open(contra_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(row for row in f if not row.startswith("#"))
            for row in reader:
                drug = row["drug_name"].strip().lower()
                condition = row["condition"].strip()
                if not drug or not condition:
                    continue

                cond_node = f"condition:{condition.lower()}"
                if not G.has_node(drug):
                    G.add_node(drug, node_type="drug", name=drug)

                G.add_node(
                    cond_node,
                    node_type="condition",
                    name=condition,
                    name_lower=condition.lower(),
                )

                G.add_edge(
                    drug,
                    cond_node,
                    edge_type="contraindicated_in",
                    severity=row.get("severity", "absolute").strip().lower(),
                    mechanism=row.get("rationale", "").strip(),
                    clinical_risk=row.get("clinical_consequence", "").strip(),
                    action=f"Contraindicated in {condition}: {row.get('clinical_consequence', '').strip()}",
                    condition=condition,
                    source=row.get("source", "FDA Boxed Warning / Curated").strip(),
                )

    # -------------------------------------------------------------------------
    # 4. ALLERGEN CROSS-REACTIVITY (cross_reacts_with)
    # -------------------------------------------------------------------------
    allergy_file = _resolve_path("allergies.csv", data_dir)
    if allergy_file.exists():
        with open(allergy_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(row for row in f if not row.startswith("#"))
            for row in reader:
                allergen = row["allergen"].strip().lower()
                drug = row["drug_name"].strip().lower()
                if not allergen or not drug:
                    continue

                alg_node = f"allergen:{allergen}"
                if not G.has_node(drug):
                    G.add_node(drug, node_type="drug", name=drug)

                G.add_node(alg_node, node_type="allergen", name=allergen)

                G.add_edge(
                    alg_node,
                    drug,
                    edge_type="cross_reacts_with",
                    severity=row.get("severity", "major").strip().lower(),
                    mechanism=row.get("mechanism", "").strip(),
                    clinical_risk=row.get("clinical_risk", "").strip(),
                    action=row.get("recommended_action", "").strip(),
                    allergen=allergen,
                    source=row.get("source", "Curated allergy immunology").strip(),
                )

    return G


# =============================================================================
# CLINICAL GRAPH QUERY HELPERS
# =============================================================================
def get_drug_interaction(
    G: nx.DiGraph, drug_a: str, drug_b: str
) -> Optional[Dict[str, Any]]:
    """O(1) lookup for drug-drug interaction between drug_a and drug_b."""
    da = drug_a.strip().lower()
    db = drug_b.strip().lower()
    if G.has_edge(da, db):
        edge_data = G.get_edge_data(da, db)
        if edge_data.get("edge_type") == "interacts_with":
            return edge_data
    return None


def get_drug_classes(G: nx.DiGraph, drug: str) -> List[Dict[str, str]]:
    """Returns list of pharmacological classes the drug belongs to."""
    d = drug.strip().lower()
    classes = []
    if G.has_node(d):
        for _, target, data in G.out_edges(d, data=True):
            if data.get("edge_type") == "belongs_to_class":
                classes.append(
                    {
                        "class_name": data.get("class_name", ""),
                        "therapeutic_category": data.get("therapeutic_category", ""),
                    }
                )
    return classes


def get_contraindications(
    G: nx.DiGraph, drug: str, condition: str
) -> Optional[Dict[str, Any]]:
    """Checks if a drug is contraindicated in a given medical condition."""
    d = drug.strip().lower()
    cond_node = f"condition:{condition.strip().lower()}"
    if G.has_edge(d, cond_node):
        edge_data = G.get_edge_data(d, cond_node)
        if edge_data.get("edge_type") == "contraindicated_in":
            return edge_data
    return None


def get_allergy_cross_reactions(
    G: nx.DiGraph, allergen: str, drug: str
) -> Optional[Dict[str, Any]]:
    """Checks if an allergen cross-reacts with a prescribed drug."""
    alg_node = f"allergen:{allergen.strip().lower()}"
    d = drug.strip().lower()
    if G.has_edge(alg_node, d):
        edge_data = G.get_edge_data(alg_node, d)
        if edge_data.get("edge_type") == "cross_reacts_with":
            return edge_data
    return None


def get_graph_summary(G: nx.DiGraph) -> Dict[str, Any]:
    """Generates structured counts of nodes and edges by type."""
    node_counts: Dict[str, int] = {}
    for _, data in G.nodes(data=True):
        ntype = data.get("node_type", "unknown")
        node_counts[ntype] = node_counts.get(ntype, 0) + 1

    edge_counts: Dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        etype = data.get("edge_type", "unknown")
        edge_counts[etype] = edge_counts.get(etype, 0) + 1

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_breakdown": node_counts,
        "edge_breakdown": edge_counts,
    }


# =============================================================================
# SANITY CHECK TEST SUITE
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("BUILDING CLINICAL KNOWLEDGE GRAPH (graph.py)")
    print("=" * 80)

    graph = build_clinical_graph()
    summary = get_graph_summary(graph)

    print(f"Total Nodes: {summary['total_nodes']}")
    print(f"Total Edges: {summary['total_edges']}")
    print("\nNode Counts by Type:")
    for ntype, count in summary["node_breakdown"].items():
        print(f"  - {ntype:<15}: {count}")

    print("\nEdge Counts by Type:")
    for etype, count in summary["edge_breakdown"].items():
        print(f"  - {etype:<20}: {count}")

    print("-" * 80)
    print("SANITY-CHECKING QUERY OPERATIONS ON ALL 4 TYPED EDGES:")
    print("-" * 80)

    # 1. interacts_with verification
    ddi = get_drug_interaction(graph, "warfarin", "aspirin")
    assert ddi is not None, "Failed to retrieve warfarin-aspirin interaction"
    print(f"[PASS] DDI Check (warfarin + aspirin):")
    print(f"       Severity : {ddi['severity']}")
    print(f"       Mechanism: {ddi['mechanism']}")
    print(f"       Action   : {ddi['action'][:75]}...")

    # 2. belongs_to_class verification
    classes = get_drug_classes(graph, "ibuprofen")
    assert any(c["class_name"] == "NSAID" for c in classes), "Failed to retrieve NSAID class"
    print(f"[PASS] Class Check (ibuprofen): {classes[0]['class_name']} ({classes[0]['therapeutic_category']})")

    # 3. contraindicated_in verification
    contra = get_contraindications(graph, "metformin", "chronic kidney disease")
    assert contra is not None, "Failed to retrieve metformin contraindication"
    print(f"[PASS] Contraindication Check (metformin in CKD):")
    print(f"       Severity : {contra['severity']}")
    print(f"       Mechanism: {contra['mechanism']}")
    print(f"       Action   : {contra['action'][:75]}...")

    # 4. cross_reacts_with verification
    allergy = get_allergy_cross_reactions(graph, "penicillin", "amoxicillin")
    assert allergy is not None, "Failed to retrieve penicillin-amoxicillin cross-reaction"
    print(f"[PASS] Allergy Check (penicillin -> amoxicillin):")
    print(f"       Severity : {allergy['severity']}")
    print(f"       Mechanism: {allergy['mechanism']}")
    print(f"       Action   : {allergy['action'][:75]}...")

    print("=" * 80)
    print("ALL GRAPH SANITY CHECKS PASSED.")
    print("=" * 80)
