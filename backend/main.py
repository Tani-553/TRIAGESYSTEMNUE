from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.inference import predict_patient
from services.ehr_store import get_user_ehr_dataset, upload_ehr_dataset
from services.database import (
    authenticate_user,
    get_next_patient_id,
    get_user_by_token,
    init_db,
    list_recent_records,
    save_patient_record,
)


class LoginRequest(BaseModel):
    username: str
    password: str


app = FastAPI(title="AI Smart Triage System")


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/")
def health():
    return {"status": "Triage API running"}


def _authorized_user(authorization):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = authorization.split(" ", 1)[1]
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return user


@app.post("/auth/login")
def login(payload: LoginRequest):
    auth_data = authenticate_user(payload.username.strip(), payload.password)
    if not auth_data:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {
        "access_token": auth_data["token"],
        "token_type": "bearer",
        "username": auth_data["username"],
        "expires_at": auth_data["expires_at"].strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


@app.get("/auth/me")
def current_user(authorization: str | None = Header(default=None)):
    user = _authorized_user(authorization)
    return {"username": user["username"]}


@app.get("/patients/next-id")
def next_patient_id(authorization: str | None = Header(default=None)):
    _authorized_user(authorization)
    return {"patient_id": get_next_patient_id()}


@app.post("/predict")
def predict(data: dict, authorization: str | None = Header(default=None)):
    user = _authorized_user(authorization)
    user_ehr_df = get_user_ehr_dataset(user["id"])
    result = predict_patient(data, user_ehr_df)
    recorded_at = save_patient_record(user["id"], data, result)
    result["recorded_at"] = recorded_at
    return result


@app.post("/ehr/upload")
async def upload_ehr(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    user = _authorized_user(authorization)
    filename = (file.filename or "").lower()
    if not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        raw_bytes = await file.read()
        rows = upload_ehr_dataset(user["id"], raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"message": "EHR dataset uploaded successfully.", "rows_loaded": rows}


@app.get("/records")
def records(
    limit: int = 10,
    q: str | None = None,
    authorization: str | None = Header(default=None),
):
    user = _authorized_user(authorization)
    safe_limit = max(1, min(limit, 500))
    return {"records": list_recent_records(user["id"], safe_limit, q)}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
