def recommend_department(patient_data, predicted_risk):

    if predicted_risk == "High":
        if patient_data["Heart_Rate"] > 130 or patient_data["Temperature"] > 103:
            return "Emergency"

    symptoms = patient_data.get("Symptoms", [])
    conditions = patient_data.get("Pre_Existing_Conditions", [])

    if "Chest Pain" in symptoms:
        return "Cardiology"

    if "Shortness of Breath" in symptoms:
        return "Pulmonology"

    if "Seizure" in symptoms:
        return "Neurology"

    if "Diabetes" in conditions:
        return "Endocrinology"

    return "General Medicine"
