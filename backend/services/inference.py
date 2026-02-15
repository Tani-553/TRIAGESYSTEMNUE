import joblib
import pandas as pd
from services.routing import recommend_department
from services.explain import Explainer


# Load once when service starts
model = joblib.load("model/triage_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
symptom_encoder = joblib.load("model/symptom_encoder.pkl")
dept_encoder = joblib.load("model/department_encoder.pkl")
dept_model = joblib.load("model/department_model.pkl")

explainer = Explainer(model)

ehr_df = pd.read_csv("data/ehr_data.csv")


def _normalize_symptoms(raw_symptoms):
    if isinstance(raw_symptoms, list):
        return [str(item).strip() for item in raw_symptoms if str(item).strip()]
    if isinstance(raw_symptoms, str):
        return [item.strip() for item in raw_symptoms.split(",") if item.strip()]
    return []


def _normalize_conditions(raw_conditions):
    if isinstance(raw_conditions, list):
        return [str(item).strip() for item in raw_conditions if str(item).strip()]
    if isinstance(raw_conditions, str):
        return [item.strip() for item in raw_conditions.split(",") if item.strip()]
    return []


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def preprocess(patient_input):
    patient = patient_input.copy()

    symptoms_list = _normalize_symptoms(patient.pop("Symptoms", []))
    conditions = _normalize_conditions(patient.get("Pre_Existing_Conditions", []))
    patient["Pre_Existing_Conditions"] = conditions[0] if conditions else "None"

    patient["Age"] = int(_safe_float(patient.get("Age"), 0))
    patient["Blood_Pressure"] = _safe_float(patient.get("Blood_Pressure"), 0)
    patient["Heart_Rate"] = _safe_float(patient.get("Heart_Rate"), 0)
    patient["Temperature"] = _safe_float(patient.get("Temperature"), 0)

    df = pd.DataFrame([patient])

    symptoms_encoded = pd.DataFrame(
        symptom_encoder.transform([symptoms_list]), columns=symptom_encoder.classes_
    )

    df = pd.concat([df, symptoms_encoded], axis=1)

    df = pd.get_dummies(
        df,
        columns=["Gender", "Pre_Existing_Conditions"],
        drop_first=True,
    )

    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    return df[model.feature_names_in_]


def _calculate_priority_score(risk_label, confidence):
    base = {"Low": 25, "Medium": 55, "High": 80}.get(risk_label, 40)
    score = base + int(round(confidence * 20))
    return max(0, min(100, score))


def _build_default_ehr_summary():
    return {
        "past_visits": 0,
        "last_risk": "Unknown",
        "avg_bp": 0,
        "avg_heart_rate": 0,
        "chronic_conditions": "None",
    }


def _extract_ehr_summary(patient_id, source_df):
    if source_df is None or "Patient_ID" not in source_df.columns:
        return _build_default_ehr_summary()

    ehr_record = source_df[source_df["Patient_ID"] == patient_id]
    if ehr_record.empty:
        return _build_default_ehr_summary()

    row = ehr_record.iloc[0]
    return {
        "past_visits": int(row.get("Past_Visits", 0) or 0),
        "last_risk": str(row.get("Last_Risk_Level", "Unknown") or "Unknown"),
        "avg_bp": float(row.get("Avg_BP", 0) or 0),
        "avg_heart_rate": float(row.get("Avg_Heart_Rate", 0) or 0),
        "chronic_conditions": str(row.get("Chronic_Conditions", "None") or "None"),
    }


def predict_patient(patient_input, ehr_source_df=None):
    processed_df = preprocess(patient_input)
    patient_id = int(_safe_float(patient_input.get("Patient_ID"), 0))
    source_df = ehr_source_df if ehr_source_df is not None else ehr_df
    ehr_summary = _extract_ehr_summary(patient_id, source_df)

    prediction = model.predict(processed_df)
    probabilities = model.predict_proba(processed_df)

    risk_label = label_encoder.inverse_transform(prediction)[0]
    confidence = float(max(probabilities[0]))

    dept_pred = dept_model.predict(processed_df)
    dept_label = dept_encoder.inverse_transform(dept_pred)[0]
    routed_department = recommend_department(patient_input, risk_label)

    if risk_label == "High" and (
        _safe_float(patient_input.get("Heart_Rate"), 0) > 130
        or _safe_float(patient_input.get("Temperature"), 0) > 103
    ):
        dept_label = "Emergency"
    elif not dept_label:
        dept_label = routed_department

    explanation = explainer.explain(processed_df, model.feature_names_in_)
    priority_score = _calculate_priority_score(risk_label, confidence)

    return {
        "risk_level": risk_label,
        "confidence": round(confidence, 3),
        "priority_score": priority_score,
        "recommended_department": dept_label,
        "top_contributors": explanation,
        "ehr_summary": ehr_summary,
    }
