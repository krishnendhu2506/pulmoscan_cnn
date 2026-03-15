import os
import sqlite3
import sys
import uuid
from datetime import datetime
from tempfile import gettempdir

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename, safe_join

from utils.preprocess import preprocess_image
from model.predict import load_model_once, predict_image
from utils.report_generator import generate_report
from models.patient_model import parse_patient_form

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
IS_VERCEL = bool(os.environ.get("VERCEL"))
RUNTIME_DIR = os.path.join(gettempdir(), "pulmoscan") if IS_VERCEL else BASE_DIR
UPLOAD_DIR = os.path.join(RUNTIME_DIR, "uploads") if IS_VERCEL else os.path.join(STATIC_DIR, "uploads")
REPORT_DIR = os.path.join(RUNTIME_DIR, "reports") if IS_VERCEL else os.path.join(STATIC_DIR, "reports")
DB_PATH = os.path.join(RUNTIME_DIR, "patients.db") if IS_VERCEL else os.path.join(BASE_DIR, "database", "patients.db")
MODEL_PATH = os.path.join(BASE_DIR, "model", "lung_cancer_model.h5")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
CLASS_NAMES = ["Adenocarcinoma", "Normal", "Squamous Cell Carcinoma"]
CLASS_LABEL_MAP = {
    "lung_n": "Normal",
    "lung_aca": "Adenocarcinoma",
    "lung_scc": "Squamous Cell Carcinoma",
}

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.secret_key = os.environ.get("SECRET_KEY", "dev_secret_key_change_me")
DEBUG_PREDICTIONS = os.environ.get("DEBUG_PREDICTIONS", "").strip().lower() in {"1", "true", "yes"}

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                contact TEXT,
                notes TEXT,
                smoking_history TEXT,
                years_of_smoking INTEGER,
                family_history_lung_cancer TEXT,
                air_pollution_exposure TEXT,
                occupational_exposure TEXT,
                persistent_cough TEXT,
                unexplained_weight_loss TEXT,
                created_at TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                prob_normal REAL NOT NULL,
                prob_aca REAL NOT NULL,
                prob_scc REAL NOT NULL,
                report_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(patient_id) REFERENCES patients(id)
            )
            """
        )
        ensure_patient_columns(conn)


def ensure_patient_columns(conn):
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(patients)").fetchall()}
    columns = {
        "smoking_history": "smoking_history TEXT",
        "years_of_smoking": "years_of_smoking INTEGER",
        "family_history_lung_cancer": "family_history_lung_cancer TEXT",
        "air_pollution_exposure": "air_pollution_exposure TEXT",
        "occupational_exposure": "occupational_exposure TEXT",
        "persistent_cough": "persistent_cough TEXT",
        "unexplained_weight_loss": "unexplained_weight_loss TEXT",
    }
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE patients ADD COLUMN {ddl}")


class User(UserMixin):
    def __init__(self, row):
        self.id = row["id"]
        self.username = row["username"]
        self.password_hash = row["password_hash"]


@login_manager.user_loader
def load_user(user_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row:
        return User(row)
    return None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_patient_code():
    stamp = datetime.now().strftime("%Y%m%d")
    token = uuid.uuid4().hex[:6].upper()
    return f"P-{stamp}-{token}"


def load_class_names():
    class_index_path = os.path.join(os.path.dirname(MODEL_PATH), "class_indices.json")
    if os.path.exists(class_index_path):
        try:
            import json

            with open(class_index_path, "r", encoding="utf-8") as f:
                indices = json.load(f)
            ordered = sorted(indices.items(), key=lambda kv: kv[1])
            names = [CLASS_LABEL_MAP.get(k, k) for k, _ in ordered]
            return names
        except Exception:
            return CLASS_NAMES
    return CLASS_NAMES


@app.route("/")
def index():
    return render_template(
        "entry.html",
        title="PulmoScan AI",
        hide_nav=True,
        hide_decor=True,
        main_class="p-0",
        body_class="splash-body",
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if not username or not password:
            flash("Username and password are required.", "danger")
            return redirect(url_for("register"))
        password_hash = generate_password_hash(password)
        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, password_hash, datetime.now().isoformat()),
                )
            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "warning")
            return redirect(url_for("register"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        with get_db() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if not row or not check_password_hash(row["password_hash"], password):
            flash("Invalid username or password.", "danger")
            return redirect(url_for("login"))
        login_user(User(row))
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    with get_db() as conn:
        total_patients = conn.execute("SELECT COUNT(*) FROM patients WHERE user_id = ?", (current_user.id,)).fetchone()[0]
        total_scans = conn.execute(
            "SELECT COUNT(*) FROM predictions p JOIN patients pa ON p.patient_id = pa.id WHERE pa.user_id = ?",
            (current_user.id,),
        ).fetchone()[0]
        stats = conn.execute(
            """
            SELECT p.predicted_class, COUNT(*) as count
            FROM predictions p
            JOIN patients pa ON p.patient_id = pa.id
            WHERE pa.user_id = ?
            GROUP BY p.predicted_class
            """,
            (current_user.id,),
        ).fetchall()
        patients = conn.execute(
            "SELECT * FROM patients WHERE user_id = ? ORDER BY created_at DESC",
            (current_user.id,),
        ).fetchall()
        recent = conn.execute(
            """
            SELECT p.*, pa.patient_id as code, pa.name as patient_name
            FROM predictions p
            JOIN patients pa ON p.patient_id = pa.id
            WHERE pa.user_id = ?
            ORDER BY p.created_at DESC
            LIMIT 5
            """,
            (current_user.id,),
        ).fetchall()

    stats_map = {row["predicted_class"]: row["count"] for row in stats}
    chart_labels = ["Normal", "Adenocarcinoma", "Squamous Cell Carcinoma"]
    chart_counts = [stats_map.get(label, 0) for label in chart_labels]

    return render_template(
        "dashboard.html",
        total_patients=total_patients,
        total_scans=total_scans,
        patients=patients,
        recent=recent,
        chart_labels=chart_labels,
        chart_counts=chart_counts,
    )


@app.route("/patients/add", methods=["POST"])
@login_required
def add_patient():
    patient_input, errors = parse_patient_form(request.form)
    if errors:
        for message in errors:
            flash(message, "warning")
        return redirect(url_for("dashboard"))

    patient_code = generate_patient_code()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO patients (
                patient_id,
                name,
                age,
                gender,
                contact,
                notes,
                smoking_history,
                years_of_smoking,
                family_history_lung_cancer,
                air_pollution_exposure,
                occupational_exposure,
                persistent_cough,
                unexplained_weight_loss,
                created_at,
                user_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_code,
                patient_input.name,
                patient_input.age,
                patient_input.gender,
                patient_input.contact,
                patient_input.notes,
                patient_input.smoking_history,
                patient_input.years_of_smoking,
                patient_input.family_history_lung_cancer,
                patient_input.air_pollution_exposure,
                patient_input.occupational_exposure,
                patient_input.persistent_cough,
                patient_input.unexplained_weight_loss,
                datetime.now().isoformat(),
                current_user.id,
            ),
        )

    flash("Patient added successfully.", "success")
    return redirect(url_for("dashboard"))


@app.route("/upload", methods=["GET"])
@login_required
def upload():
    with get_db() as conn:
        patients = conn.execute(
            "SELECT * FROM patients WHERE user_id = ? ORDER BY created_at DESC",
            (current_user.id,),
        ).fetchall()
    return render_template("upload.html", patients=patients)


@app.route("/uploads/<path:filename>")
@login_required
def uploaded_file(filename):
    safe_path = safe_join(UPLOAD_DIR, filename)
    if not safe_path or not os.path.exists(safe_path):
        flash("Uploaded file not found.", "warning")
        return redirect(url_for("history"))
    return send_file(safe_path)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    patient_id = request.form.get("patient_id")
    file = request.files.get("image")

    if not patient_id:
        flash("Select a patient before uploading.", "warning")
        return redirect(url_for("upload"))

    if not file or file.filename == "":
        flash("Please upload a CT scan image.", "warning")
        return redirect(url_for("upload"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Upload an image file.", "danger")
        return redirect(url_for("upload"))

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    image_path = os.path.join(UPLOAD_DIR, unique_name)
    file.save(image_path)

    if not os.path.exists(MODEL_PATH):
        flash("Model file not found. Train the model first.", "danger")
        return redirect(url_for("upload"))

    model = load_model_once(MODEL_PATH)
    processed = preprocess_image(image_path)
    raw_probs = predict_image(model, processed)

    class_names = load_class_names()
    probs = np.asarray(raw_probs, dtype="float64").reshape(-1)

    if probs.size != len(class_names):
        flash("Model output does not match class mapping. Re-train/export the model with class_indices.json.", "danger")
        return redirect(url_for("upload"))

    probs_sum = float(probs.sum())
    if probs.min() < 0.0 or probs.max() > 1.0 or abs(probs_sum - 1.0) > 0.05:
        exps = np.exp(probs - probs.max())
        probs = exps / exps.sum()

    predicted_index = int(probs.argmax())
    predicted_class = class_names[predicted_index]
    confidence = float(probs[predicted_index]) * 100.0

    color_by_label = {
        "Normal": "#4CD4B0",
        "Adenocarcinoma": "#F5B461",
        "Squamous Cell Carcinoma": "#F76C6C",
    }

    bar_class_by_label = {
        "Normal": "",
        "Adenocarcinoma": "bg-warning",
        "Squamous Cell Carcinoma": "bg-danger",
    }

    prob_rows = []
    for idx, label in enumerate(class_names):
        prob_rows.append(
            {
                "label": label,
                "prob": float(probs[idx]) * 100.0,
                "bar_class": bar_class_by_label.get(label, ""),
                "color": color_by_label.get(label, "#2fb0c5"),
            }
        )

    prob_by_label = {row["label"]: row["prob"] for row in prob_rows}
    prob_normal = float(prob_by_label.get("Normal", 0.0))
    prob_aca = float(prob_by_label.get("Adenocarcinoma", 0.0))
    prob_scc = float(prob_by_label.get("Squamous Cell Carcinoma", 0.0))

    if DEBUG_PREDICTIONS:
        app.logger.info(
            "prediction user=%s image=%s class=%s conf=%.2f probs=%s",
            getattr(current_user, "id", None),
            unique_name,
            predicted_class,
            confidence,
            {row["label"]: round(row["prob"], 2) for row in prob_rows},
        )

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO predictions (patient_id, image_path, predicted_class, confidence, prob_normal, prob_aca, prob_scc, report_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(patient_id),
                image_path,
                predicted_class,
                confidence,
                prob_normal,
                prob_aca,
                prob_scc,
                "",
                datetime.now().isoformat(),
            ),
        )
        prediction_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_filename = f"report_{prediction_id}.pdf"
    report_path = os.path.join(REPORT_DIR, report_filename)

    with get_db() as conn:
        patient = conn.execute("SELECT * FROM patients WHERE id = ?", (int(patient_id),)).fetchone()

    generate_report(
        report_path=report_path,
        patient=patient,
        image_path=image_path,
        predicted_class=predicted_class,
        confidence=confidence,
        probs={row["label"]: row["prob"] for row in prob_rows},
        prediction_id=prediction_id,
        model_name="PulmoScan CNN v1",
        scan_date=datetime.now().strftime("%Y-%m-%d"),
        logo_path=os.path.join(STATIC_DIR, "images", "logoorg.png"),
    )

    with get_db() as conn:
        conn.execute(
            "UPDATE predictions SET report_path = ? WHERE id = ?",
            (report_path, prediction_id),
        )

    result = {
        "prediction_id": prediction_id,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "prob_normal": prob_normal,
        "prob_aca": prob_aca,
        "prob_scc": prob_scc,
        "class_names": class_names,
        "prob_rows": prob_rows,
        "chart_labels": [row["label"] for row in prob_rows],
        "chart_data": [row["prob"] for row in prob_rows],
        "chart_colors": [row["color"] for row in prob_rows],
    }

    return render_template("result.html", result=result, patient=patient, image_filename=unique_name)


@app.route("/history")
@login_required
def history():
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT p.*, pa.patient_id as code, pa.name as patient_name
            FROM predictions p
            JOIN patients pa ON p.patient_id = pa.id
            WHERE pa.user_id = ?
            ORDER BY p.created_at DESC
            """,
            (current_user.id,),
        ).fetchall()
    return render_template("history.html", rows=rows)


@app.route("/report/<int:prediction_id>")
@login_required
def download_report(prediction_id):
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT p.report_path
            FROM predictions p
            JOIN patients pa ON p.patient_id = pa.id
            WHERE p.id = ? AND pa.user_id = ?
            """,
            (prediction_id, current_user.id),
        ).fetchone()
    if not row:
        flash("Report not found.", "warning")
        return redirect(url_for("history"))
    return send_file(row["report_path"], as_attachment=True)


init_db()

if __name__ == "__main__":
    app.run(debug=True)
