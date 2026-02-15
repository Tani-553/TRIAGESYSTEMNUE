import hashlib
import sqlite3
from datetime import datetime, timedelta, UTC
from pathlib import Path
import secrets

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "triage.db"


def _utc_timestamp():
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _hash_password(password, salt_hex):
    salt = bytes.fromhex(salt_hex)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return digest.hex()


def _create_password_hash(password):
    salt_hex = secrets.token_hex(16)
    return salt_hex, _hash_password(password, salt_hex)


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                patient_id INTEGER,
                patient_name TEXT,
                age INTEGER,
                gender TEXT,
                symptoms TEXT,
                pre_existing_conditions TEXT,
                blood_pressure REAL,
                heart_rate REAL,
                temperature REAL,
                risk_level TEXT,
                confidence REAL,
                priority_score INTEGER,
                recommended_department TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )

        # Lightweight migration for existing databases created before patient_name was added.
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(patient_records)").fetchall()
        }
        if "patient_name" not in columns:
            conn.execute("ALTER TABLE patient_records ADD COLUMN patient_name TEXT")

        user_count = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()["count"]
        if user_count == 0:
            salt, password_hash = _create_password_hash("admin123")
            conn.execute(
                """
                INSERT INTO users (username, password_hash, salt, created_at)
                VALUES (?, ?, ?, ?)
                """,
                ("admin", password_hash, salt, _utc_timestamp()),
            )


def authenticate_user(username, password):
    with get_connection() as conn:
        user = conn.execute(
            "SELECT id, username, password_hash, salt FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if not user:
            return None

        if _hash_password(password, user["salt"]) != user["password_hash"]:
            return None

        created_at = datetime.now(UTC)
        expires_at = created_at + timedelta(hours=12)
        token = secrets.token_urlsafe(32)

        conn.execute(
            """
            INSERT INTO sessions (token, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                token,
                user["id"],
                created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                expires_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            ),
        )

        return {"token": token, "username": user["username"], "expires_at": expires_at}


def get_user_by_token(token):
    with get_connection() as conn:
        session = conn.execute(
            """
            SELECT s.user_id, s.expires_at, u.username
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()

        if not session:
            return None

        expires_at = datetime.strptime(session["expires_at"], "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=UTC)
        if datetime.now(UTC) > expires_at:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            return None

        return {"id": session["user_id"], "username": session["username"]}


def save_patient_record(user_id, patient_data, prediction_result):
    created_at = _utc_timestamp()

    symptoms = patient_data.get("Symptoms", [])
    if isinstance(symptoms, list):
        symptoms_text = ", ".join(symptoms)
    else:
        symptoms_text = str(symptoms)

    conditions = patient_data.get("Pre_Existing_Conditions", [])
    if isinstance(conditions, list):
        conditions_text = ", ".join(conditions)
    else:
        conditions_text = str(conditions)

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO patient_records (
                user_id,
                patient_id,
                patient_name,
                age,
                gender,
                symptoms,
                pre_existing_conditions,
                blood_pressure,
                heart_rate,
                temperature,
                risk_level,
                confidence,
                priority_score,
                recommended_department,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                patient_data.get("Patient_ID"),
                patient_data.get("Patient_Name"),
                patient_data.get("Age"),
                patient_data.get("Gender"),
                symptoms_text,
                conditions_text,
                patient_data.get("Blood_Pressure"),
                patient_data.get("Heart_Rate"),
                patient_data.get("Temperature"),
                prediction_result.get("risk_level"),
                prediction_result.get("confidence"),
                prediction_result.get("priority_score"),
                prediction_result.get("recommended_department"),
                created_at,
            ),
        )

    return created_at


def list_recent_records(user_id, limit=10, search=None):
    params = [user_id]
    where_clause = "WHERE user_id = ?"

    if search:
        query = search.strip()
        if query:
            where_clause += " AND (CAST(patient_id AS TEXT) LIKE ? OR LOWER(COALESCE(patient_name, '')) LIKE ?)"
            params.extend([f"%{query}%", f"%{query.lower()}%"])

    params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT
                id,
                patient_id,
                patient_name,
                age,
                gender,
                symptoms,
                pre_existing_conditions,
                blood_pressure,
                heart_rate,
                temperature,
                risk_level,
                confidence,
                priority_score,
                recommended_department,
                created_at
            FROM patient_records
            {where_clause}
            ORDER BY id DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()

    return [dict(row) for row in rows]


def get_next_patient_id():
    with get_connection() as conn:
        row = conn.execute(
            "SELECT MAX(patient_id) AS max_patient_id FROM patient_records"
        ).fetchone()

    current_max = row["max_patient_id"] if row and row["max_patient_id"] is not None else None
    if current_max is None:
        return 1001

    try:
        next_id = int(current_max) + 1
    except (TypeError, ValueError):
        next_id = 1001

    return max(1001, next_id)
