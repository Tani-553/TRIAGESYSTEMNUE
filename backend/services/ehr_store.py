from io import BytesIO
import re

import pandas as pd


_user_ehr_store = {}

_REQUIRED_COLUMNS = [
    "Patient_ID",
    "Past_Visits",
    "Last_Risk_Level",
    "Avg_BP",
    "Avg_Heart_Rate",
    "Chronic_Conditions",
]

_ALIASES = {
    "Patient_ID": ["patientid", "patient_id", "id", "patient"],
    "Past_Visits": ["pastvisits", "past_visits", "visits", "visitcount", "visit_count"],
    "Last_Risk_Level": ["lastrisklevel", "last_risk_level", "lastrisk", "last_risk"],
    "Avg_BP": ["avgbp", "avg_bp", "averagebp", "average_bp", "bpavg", "bp_avg"],
    "Avg_Heart_Rate": [
        "avgheartrate",
        "avg_heart_rate",
        "averageheartrate",
        "average_heart_rate",
        "hravg",
        "hr_avg",
    ],
    "Chronic_Conditions": [
        "chronicconditions",
        "chronic_conditions",
        "conditions",
        "preexistingconditions",
        "pre_existing_conditions",
    ],
}


def _norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).strip().lower())


def _map_columns(df):
    normalized_existing = {_norm(col): col for col in df.columns}
    rename_map = {}

    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            existing_col = normalized_existing.get(alias)
            if existing_col:
                rename_map[existing_col] = canonical
                break

    mapped = df.rename(columns=rename_map).copy()

    for col in _REQUIRED_COLUMNS:
        if col not in mapped.columns:
            if col in ("Past_Visits", "Avg_BP", "Avg_Heart_Rate"):
                mapped[col] = 0
            elif col == "Last_Risk_Level":
                mapped[col] = "Unknown"
            elif col == "Chronic_Conditions":
                mapped[col] = "None"

    return mapped


def _coerce_ehr_types(df):
    working = df.copy()

    if "Patient_ID" not in working.columns:
        raise ValueError("Uploaded CSV must include a patient id column.")

    working["Patient_ID"] = pd.to_numeric(working["Patient_ID"], errors="coerce")
    working = working.dropna(subset=["Patient_ID"])
    working["Patient_ID"] = working["Patient_ID"].astype(int)

    for numeric_col in ("Past_Visits", "Avg_BP", "Avg_Heart_Rate"):
        working[numeric_col] = pd.to_numeric(working[numeric_col], errors="coerce").fillna(0)

    working["Last_Risk_Level"] = working["Last_Risk_Level"].fillna("Unknown").astype(str)
    working["Chronic_Conditions"] = working["Chronic_Conditions"].fillna("None").astype(str)

    return working[_REQUIRED_COLUMNS]


def upload_ehr_dataset(user_id, file_bytes):
    try:
        uploaded = pd.read_csv(BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError("Unable to parse CSV file.") from exc

    if uploaded.empty:
        raise ValueError("Uploaded CSV is empty.")

    mapped = _map_columns(uploaded)
    prepared = _coerce_ehr_types(mapped)
    _user_ehr_store[user_id] = prepared

    return len(prepared.index)


def get_user_ehr_dataset(user_id):
    return _user_ehr_store.get(user_id)
