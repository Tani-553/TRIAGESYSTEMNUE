import os
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def generate_summary(patient_input, risk, department):

    if not OPENAI_API_KEY:
        return "AI summary unavailable."

    prompt = f"""
    You are a clinical triage assistant.
    Based on the following patient information:

    Age: {patient_input['Age']}
    Symptoms: {patient_input['Symptoms']}
    Blood Pressure: {patient_input['Blood_Pressure']}
    Heart Rate: {patient_input['Heart_Rate']}
    Temperature: {patient_input['Temperature']}
    Risk Level: {risk}
    Recommended Department: {department}

    Generate a short professional medical triage summary (2-3 sentences).
    """

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=5
        )

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except:
        return "AI summary generation failed."
