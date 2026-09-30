"""
app.py - AI-Based Medication Error and Dangerous Drug Interaction Detection System
VNR Hackathon - Healthcare Track (Offline, 100% Local Execution)

Tech Stack:
  - Python 3.10+, Streamlit, NetworkX, pandas, rapidfuzz, dataclasses
  - Modules: normalizer.py, graph.py, checkers.py, explain.py, parser.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from checkers import Alert, check_all_safety_rules
from explain import explain_alert
from graph import build_clinical_graph, get_drug_interaction, get_graph_summary
from parser import parse_prescription

# =============================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(
    page_title="SafeRx | AI Medication Error & Interaction Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom modern clinical theme styling
st.markdown(
    """
    <style>
    /* Global Styles */
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-offline {
        display: inline-block;
        background-color: #ECFDF5;
        color: #065F46;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #A7F3D0;
    }

    /* Patient Profile Card */
    .patient-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.1rem;
        margin-bottom: 1.25rem;
    }
    .patient-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
    }
    .pill-condition {
        display: inline-block;
        background: #EFF6FF;
        color: #1E40AF;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
        border: 1px solid #BFDBFE;
    }
    .pill-allergy {
        display: inline-block;
        background: #FEF2F2;
        color: #991B1B;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
        border: 1px solid #FECACA;
    }
    .pill-med {
        display: inline-block;
        background: #F1F5F9;
        color: #334155;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 2px;
        border: 1px solid #CBD5E1;
    }

    /* Summary Banners */
    .summary-danger {
        background: #FEF2F2;
        border-left: 6px solid #DC2626;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        color: #991B1B;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 1.25rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .summary-warning {
        background: #FFFBEB;
        border-left: 6px solid #D97706;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        color: #92400E;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 1.25rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .summary-success {
        background: #ECFDF5;
        border-left: 6px solid #059669;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        color: #065F46;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 1.25rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Alert Severity Cards */
    .card-major {
        background: #FFFFFF;
        border-left: 5px solid #DC2626;
        border-top: 1px solid #FEE2E2;
        border-right: 1px solid #FEE2E2;
        border-bottom: 1px solid #FEE2E2;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.85rem;
    }
    .card-moderate {
        background: #FFFFFF;
        border-left: 5px solid #D97706;
        border-top: 1px solid #FEF3C7;
        border-right: 1px solid #FEF3C7;
        border-bottom: 1px solid #FEF3C7;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.85rem;
    }
    .card-minor {
        background: #FFFFFF;
        border-left: 5px solid #2563EB;
        border-top: 1px solid #DBEAFE;
        border-right: 1px solid #DBEAFE;
        border-bottom: 1px solid #DBEAFE;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.85rem;
    }
    .action-box {
        background: #F8FAFC;
        border: 1px dashed #94A3B8;
        border-radius: 6px;
        padding: 0.65rem 0.85rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #0F172A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent


# =============================================================================
# RESOURCE CACHING
# =============================================================================
@st.cache_resource(show_spinner="Initializing Clinical Knowledge Graph...")
def load_graph():
    """Builds and caches the NetworkX clinical interaction graph."""
    return build_clinical_graph()


@st.cache_data
def load_patients() -> List[Dict[str, Any]]:
    """Loads demo patient profiles from demo_patients.json."""
    patients_file = BASE_DIR / "demo_patients.json"
    if patients_file.exists():
        with open(patients_file, mode="r", encoding="utf-8") as f:
            return json.load(f)
    return []


G = load_graph()
patients = load_patients()
graph_summary = get_graph_summary(G)


# =============================================================================
# SIDEBAR: PATIENT SELECTION & PROFILE
# =============================================================================
with st.sidebar:
    st.markdown("### 👤 Patient Clinical Record")
    st.caption("Select a preloaded patient profile to simulate bedside validation:")

    if "patient_idx" not in st.session_state:
        st.session_state["patient_idx"] = 0

    patient_options = [f"{p['id']}: {p['name']}" for p in patients]
    selected_patient_idx = st.selectbox(
        "Select Patient Profile",
        options=range(len(patient_options)),
        format_func=lambda i: patient_options[i] if i < len(patient_options) else "",
        index=st.session_state["patient_idx"],
        key="patient_selector",
    )
    st.session_state["patient_idx"] = selected_patient_idx

    current_patient = patients[selected_patient_idx] if patients else {}

    if current_patient:
        st.markdown(
            f"""
            <div class="patient-card">
                <div class="patient-name">{current_patient.get('name', 'Unknown')}</div>
                <div style="font-size: 0.85rem; color: #475569; margin-top: 4px;">
                    <strong>Age:</strong> {current_patient.get('age')} years | 
                    <strong>Gender:</strong> {current_patient.get('gender', 'N/A')}
                </div>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #E2E8F0;" />
                <div style="font-size: 0.8rem; font-weight: 600; color: #334155; margin-bottom: 2px;">
                    DIAGNOSED CONDITIONS:
                </div>
                <div>
                    {''.join([f'<span class="pill-condition">{c}</span>' for c in current_patient.get('conditions', [])])}
                </div>
                <div style="font-size: 0.8rem; font-weight: 600; color: #334155; margin-top: 8px; margin-bottom: 2px;">
                    DOCUMENTED ALLERGIES:
                </div>
                <div>
                    {''.join([f'<span class="pill-allergy">⚠️ {a}</span>' for a in current_patient.get('allergies', [])]) or '<span style="color:#64748B; font-size:0.75rem;">None recorded</span>'}
                </div>
                <div style="font-size: 0.8rem; font-weight: 600; color: #334155; margin-top: 8px; margin-bottom: 2px;">
                    ACTIVE CURRENT REGIMEN:
                </div>
                <div>
                    {''.join([f'<span class="pill-med">💊 {m}</span>' for m in current_patient.get('current_meds', [])]) or '<span style="color:#64748B; font-size:0.75rem;">No active medications</span>'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📋 Physician Clinical Notes", expanded=False):
            st.info(current_patient.get("clinical_notes", "No additional clinical notes."))

    st.markdown("---")
    st.markdown("### 🕸️ Knowledge Graph Telemetry")
    col_t1, col_t2 = st.columns(2)
    col_t1.metric("Graph Nodes", graph_summary["total_nodes"])
    col_t2.metric("Typed Edges", graph_summary["total_edges"])
    st.caption("🔒 100% Offline Local Inference. Zero External API Calls.")


# =============================================================================
# MAIN PANEL: PRESCRIPTION INPUT & QUICK PRESETS
# =============================================================================
col_head, col_badge = st.columns([4, 1])
with col_head:
    st.markdown('<div class="main-title">AI Medication Error & Drug Interaction Shield</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Automated clinical detection of dangerous DDIs, contraindications, duplicates, dosage errors, and allergies.</div>',
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown('<div style="text-align: right; margin-top: 10px;"><span class="badge-offline">⚡ Local & Offline Engine</span></div>', unsafe_allow_html=True)

# Preset Demo Prescriptions for Rapid Hackathon Testing
st.markdown("##### ⚡ One-Click Demo Presets for Hackathon Judges:")
c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)

if "rx_input" not in st.session_state:
    st.session_state["rx_input"] = "Tab Ecosprin 75mg OD, Tab Augmentin 625mg BD"

if c1.button("🩸 1. Drug-Drug (DDI)", help="Warfarin + Ecosprin Bleed Risk", use_container_width=True):
    st.session_state["patient_idx"] = 0
    st.session_state["rx_input"] = "Tab Ecosprin 75mg OD"
    st.rerun()

if c2.button("🫁 2. Drug-Disease", help="Asthma + Inderal Bronchospasm", use_container_width=True):
    st.session_state["patient_idx"] = 1
    st.session_state["rx_input"] = "Tab Inderal 40mg OD"
    st.rerun()

if c3.button("🔄 3. Duplicate Therapy", help="Brufen + Naproxen Dual NSAID", use_container_width=True):
    st.session_state["patient_idx"] = 2
    st.session_state["rx_input"] = "Tab Naproxen 500mg BD"
    st.rerun()

if c4.button("⚖️ 4. Dosage Overdose", help="Paracetamol 6000mg/day Overdose", use_container_width=True):
    st.session_state["patient_idx"] = 3
    st.session_state["rx_input"] = "Tab Paracetamol 1500mg QID"
    st.rerun()

if c5.button("🛡️ 5. Allergy Conflict", help="Penicillin Allergy + Augmentin", use_container_width=True):
    st.session_state["patient_idx"] = 4
    st.session_state["rx_input"] = "Tab Augmentin 625mg BD"
    st.rerun()

if c6.button("✅ 6. Clean True Negative", help="Zero-alert safe prescription", use_container_width=True):
    st.session_state["patient_idx"] = 5
    st.session_state["rx_input"] = "Tab Pantoprazole 40mg OD, Tab Paracetamol 650mg TDS"
    st.rerun()

# Free-Text Prescription Input Box
rx_text = st.text_area(
    "Enter Free-Text Prescription (Brand names, generics, dosages, & abbreviations like OD, BD, TDS, QID):",
    value=st.session_state["rx_input"],
    height=90,
    help="Supports Indian brand names (e.g. Dolo, Pan-D, Ecosprin), generics, dosages, and typos.",
)

scan_clicked = st.button("🔍 Scan Prescription for Clinical Safety", type="primary", use_container_width=True)


# =============================================================================
# EVALUATION & RESULTS PIPELINE
# =============================================================================
if rx_text.strip():
    # 1. Parse free-text into structured entities
    parsed_items = parse_prescription(rx_text)

    # 2. Run all checkers across each prescribed item
    all_raw_alerts: List[Alert] = []

    # A) Check each new drug against patient's current profile
    for item in parsed_items:
        found_alerts = check_all_safety_rules(current_patient, item, graph=G)
        all_raw_alerts.extend(found_alerts)

    # B) Check intra-prescription interactions (pairwise between newly prescribed drugs)
    if len(parsed_items) >= 2:
        for i in range(len(parsed_items)):
            for j in range(i + 1, len(parsed_items)):
                item_a = parsed_items[i]
                item_b = parsed_items[j]
                da, db = item_a["drug"], item_b["drug"]

                # Intra-order DDI
                ddi_edge = get_drug_interaction(G, da, db)
                if ddi_edge:
                    all_raw_alerts.append(
                        Alert(
                            type="drug-drug",
                            severity=ddi_edge.get("severity", "major"),
                            drugs=[da, db],
                            reason="",
                            action="",
                        )
                    )

                # Intra-order Duplicate Therapy
                dup_alerts = check_all_safety_rules({"current_meds": [item_a]}, item_b, graph=G)
                for d_alert in dup_alerts:
                    if d_alert.type == "duplicate":
                        all_raw_alerts.append(d_alert)

    # 3. Enrich all alerts with non-generic mechanistic explanations
    enriched_alerts: List[Alert] = []
    seen_fingerprints = set()

    for raw_alert in all_raw_alerts:
        explained = explain_alert(raw_alert, graph=G)
        fingerprint = (explained.type, tuple(sorted(explained.drugs)), explained.severity)
        if fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            enriched_alerts.append(explained)

    # 4. Sort alerts strictly: major -> moderate -> minor
    severity_order = {"major": 0, "moderate": 1, "minor": 2}
    enriched_alerts.sort(key=lambda a: severity_order.get(a.severity.lower(), 99))

    # Calculate summary metrics
    major_count = sum(1 for a in enriched_alerts if a.severity.lower() == "major")
    moderate_count = sum(1 for a in enriched_alerts if a.severity.lower() == "moderate")
    minor_count = sum(1 for a in enriched_alerts if a.severity.lower() == "minor")
    total_alerts = len(enriched_alerts)

    st.markdown("---")

    # Display Parsed Structured Data Table in Expander
    with st.expander(f"📦 Parsed Prescription Entities ({len(parsed_items)} medications identified)", expanded=False):
        if parsed_items:
            table_data = []
            for itm in parsed_items:
                table_data.append(
                    {
                        "Original Free-Text": itm.get("raw"),
                        "Normalized Generic Drug": itm.get("drug"),
                        "Single Dose (mg)": itm.get("dose_mg") if itm.get("dose_mg") is not None else "N/A",
                        "Frequency": f"{itm.get('frequency')}x/day",
                        "Cumulative 24h Dose": f"{itm.get('daily_dose')} mg" if itm.get("daily_dose") is not None else "N/A",
                    }
                )
            st.dataframe(table_data, use_container_width=True)
        else:
            st.info("No recognizable drug entities found in input text.")

    # 5. Summary Banner
    if total_alerts > 0:
        summary_parts = []
        if major_count > 0:
            summary_parts.append(f"{major_count} Major Risk{'s' if major_count > 1 else ''}")
        if moderate_count > 0:
            summary_parts.append(f"{moderate_count} Moderate Risk{'s' if moderate_count > 1 else ''}")
        if minor_count > 0:
            summary_parts.append(f"{minor_count} Minor Risk{'s' if minor_count > 1 else ''}")

        summary_text = ", ".join(summary_parts) + " Found!"
        banner_class = "summary-danger" if major_count > 0 else "summary-warning"

        st.markdown(
            f'<div class="{banner_class}">⚠️ Clinical Alert: {summary_text} Review before dispensing.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="summary-success">✅ Safe to Dispense: No Contraindications, Dangerous DDIs, Duplicates, Dosage Overdoses, or Allergy Conflicts Found.</div>',
            unsafe_allow_html=True,
        )

    # 6. Detailed Alert Cards (Expandable, Color-Coded)
    if enriched_alerts:
        st.markdown("### 🚨 Identified Clinical Safety Hazards")

        for idx, alert in enumerate(enriched_alerts, 1):
            sev = alert.severity.lower()
            icon_map = {
                "drug-drug": "⚡",
                "drug-disease": "🩺",
                "duplicate": "🔄",
                "dosage": "⚖️",
                "allergy": "🛡️",
            }
            icon = icon_map.get(alert.type, "⚠️")

            card_class = f"card-{sev}"
            badge_color = "#DC2626" if sev == "major" else ("#D97706" if sev == "moderate" else "#2563EB")

            drugs_display = " + ".join([d.capitalize() for d in alert.drugs])
            expander_title = f"{icon} [{alert.severity.upper()}] {alert.type.upper()}: {drugs_display}"

            with st.expander(expander_title, expanded=(sev == "major")):
                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div style="font-size: 0.85rem; font-weight: 700; color: {badge_color}; text-transform: uppercase; margin-bottom: 4px;">
                            Severity: {alert.severity.upper()} | Type: {alert.type}
                        </div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #1E293B; margin-bottom: 8px;">
                            Target Medication(s): <span style="color: #4338CA;">{drugs_display}</span>
                        </div>
                        <div style="font-size: 0.95rem; color: #334155; line-height: 1.5; margin-bottom: 8px;">
                            <strong>Mechanistic Rationale:</strong><br />
                            {alert.reason}
                        </div>
                        <div class="action-box">
                            <strong>👨‍⚕️ Prescriber Action:</strong><br />
                            {alert.action}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
