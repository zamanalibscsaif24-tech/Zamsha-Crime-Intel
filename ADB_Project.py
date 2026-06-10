import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import hashlib
import io
import random
from pathlib import Path
from datetime import datetime
from decimal import Decimal

# Optional heavy imports
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except Exception:
    MYSQL_AVAILABLE = False

try:
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except Exception:
    PYMONGO_AVAILABLE = False

try:
    import pydeck as pdk
except Exception:
    pdk = None

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import altair as alt

# ==========================================================
# CONFIG & ZAMSHA BRANDING
# ==========================================================
APP_TITLE = "Crime Intelligence - Zamsha Enterprise"

# Read from Streamlit secrets OR environment variables OR hardcoded defaults
def _cfg(key, default):
    try:
        return st.secrets["mysql"][key]
    except Exception:
        return os.getenv(key.upper(), default)

MYSQL_CONFIG = {
    "host":     _cfg("host",     "localhost"),
    "user":     _cfg("user",     "root"),
    "password": _cfg("password", "4510145424465"),
    "database": _cfg("database", "crime_db"),
}

MONGO_URI               = os.getenv("MONGO_URI",  "mongodb://localhost:27017/")
MONGO_DB                = "crime_project"
MONGO_COLLECTION_REPORTS  = "crime_reports"
MONGO_COLLECTION_EVIDENCE = "case_evidence"

# Dataset URL for auto-seeding (e.g. Google Drive sharing link)
DATASET_URL = os.getenv("DATASET_URL", "https://drive.google.com/file/d/1N5EmanXdFph183pSwkUEmPD6EhmlArsk/view?usp=sharing")

def get_gdrive_direct_url(url):
    """Converts a standard Google Drive sharing link into a direct download URL."""
    if not url:
        return ""
    if "drive.google.com" in url:
        if "file/d/" in url:
            parts = url.split("file/d/")
            if len(parts) > 1:
                file_id = parts[1].split("/")[0]
                return f"https://drive.google.com/uc?export=download&id={file_id}"
        elif "id=" in url:
            parts = url.split("id=")
            if len(parts) > 1:
                file_id = parts[1].split("&")[0]
                return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

# SQLite path – /tmp is always writable on Streamlit Cloud and any device
import tempfile as _tempfile
SQLITE_PATH = str(Path(_tempfile.gettempdir()) / "crime_db.sqlite")

CITY_COORDS = {
    "Karachi":    [24.8607, 67.0011], "Lahore":      [31.5204, 74.3587],
    "Islamabad":  [33.6844, 73.0479], "Rawalpindi":  [33.5909, 73.0436],
    "Faisalabad": [31.4187, 73.0791], "Peshawar":    [34.0151, 71.5249],
    "Multan":     [30.1575, 71.5249], "Quetta":      [30.1798, 66.9750],
    "Gujranwala": [32.1617, 74.1883], "Hyderabad":   [25.3960, 68.3578],
    "Sialkot":    [32.4925, 74.5310],
}

st.set_page_config(page_title=APP_TITLE, layout="centered", page_icon="🛡️")

# Zamsha Premium Dark Mode CSS
st.markdown("""
<style>
    /* ── Base ── */
    body { background-color: #0f172a; color: #f8fafc; }
    .stApp { background-color: #0f172a; color: #f8fafc; }

    /* ── Form Labels and Widgets ── */
    label, [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p {
        color: #f8fafc !important;
    }

    /* ── Tabs: ensure visibility in all themes ── */
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }

    /* ── Metric cards: stack on mobile ── */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
        border-top: 4px solid #38bdf8;
        margin-bottom: 1rem;
        width: 100%;
        box-sizing: border-box;
    }
    .metric-val   { font-size: 2rem; font-weight: bold; color: #f8fafc; word-break: break-word; }
    .metric-label { font-size: 0.8rem; color: #94a3b8; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 1px; }

    /* ── Buttons (both regular and form submit) ── */
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #0ea5e9, #2563eb) !important;
        color: white !important; border: none !important; border-radius: 8px !important; font-weight: bold !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important; transition: 0.3s !important;
        width: 100% !important; min-height: 44px !important;          /* touch-friendly */
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(90deg, #38bdf8, #3b82f6) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.5) !important;
        color: white !important;
    }

    /* ── Inputs: bigger touch targets & cohesive dark mode ── */
    input, textarea, select, [data-baseweb="select"] > div {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        min-height: 44px !important;
        font-size: 16px !important;             /* prevents iOS zoom */
    }
    input::placeholder {
        color: #64748b !important;
    }

    /* Dropdown popover listbox styling */
    div[data-baseweb="popover"] ul {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
    }
    div[data-baseweb="popover"] li {
        color: #f8fafc !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: #334155 !important;
    }

    /* ── Typography ── */
    h1 { color: #f8fafc; font-family: 'Inter', sans-serif;
         border-bottom: 1px solid #334155; padding-bottom: 0.5rem;
         margin-bottom: 1.5rem; font-size: clamp(1.4rem, 5vw, 2rem); }
    h2, h3 { color: #e2e8f0; font-family: 'Inter', sans-serif; }

    /* ── Footer: smaller on mobile so it doesn't eat screen ── */
    .zamsha-footer {
        position: fixed; left: 0; bottom: 0; width: 100%; text-align: center;
        background: rgba(15, 23, 42, 0.95); padding: 6px 4px;
        border-top: 1px solid #334155;
        color: #94a3b8; font-family: monospace;
        font-size: clamp(0.6rem, 2.5vw, 0.82rem);
        z-index: 1000; backdrop-filter: blur(5px);
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }

    /* ── Sidebar branding ── */
    .zamsha-sidebar {
        margin-top: 20px; margin-bottom: 20px; padding: 12px; border-radius: 10px;
        background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155;
        text-align: center; color: #38bdf8; font-weight: bold;
        font-family: 'Inter', sans-serif; box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }

    /* ── DB badge ── */
    .db-badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: bold; margin-left: 8px;
    }
    .db-mysql  { background: #166534; color: #bbf7d0; }
    .db-sqlite { background: #1e3a5f; color: #bfdbfe; }

    /* ── Columns: switch to single-column on very small screens ── */
    @media (max-width: 480px) {
        [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
        .metric-val { font-size: 1.6rem; }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='zamsha-footer'>🚀 Developed by Zamsha developers (Zaman Shar) | Enterprise Advanced Database Project</div>", unsafe_allow_html=True)

# ==========================================================
# DB BACKEND DETECTION
# ==========================================================
def _try_mysql_connect():
    """Try to connect to MySQL. Returns conn or None."""
    if not MYSQL_AVAILABLE:
        return None
    try:
        conn = mysql.connector.connect(
            host=MYSQL_CONFIG["host"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"],
            connection_timeout=3,
        )
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_CONFIG['database']}`")
        cur.close()
        conn.database = MYSQL_CONFIG["database"]
        return conn
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def _detect_backend():
    conn = _try_mysql_connect()
    if conn:
        conn.close()
        return "mysql"
    return "sqlite"

# ==========================================================
# ENTERPRISE ENGINE  (MySQL ↔ SQLite transparent dual-backend)
# ==========================================================
class EnterpriseEngine:
    def __init__(self):
        self.backend = _detect_backend()
        self.mongo_client = None
        self.mongo_collection_reports = None
        self.mongo_collection_evidence = None
        self.outcome_model = None
        self.outcome_model_features = []
        self.outcome_model_accuracy = None
        if self.backend == "sqlite":
            self._sqlite_init_schema()

    # ── connection helpers ─────────────────────────────────
    def get_fresh_mysql_conn(self):
        conn = mysql.connector.connect(
            host=MYSQL_CONFIG["host"], user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"],
        )
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_CONFIG['database']}`")
        cur.close()
        conn.database = MYSQL_CONFIG["database"]
        return conn

    def _get_sqlite_conn(self):
        conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ── unified query layer ────────────────────────────────
    def _adapt_sql(self, query):
        """Convert MySQL-flavoured SQL to SQLite-compatible SQL."""
        import re
        q = query
        # placeholders: %s → ?
        q = q.replace("%s", "?")
        # IFNULL already works in SQLite; GROUP_CONCAT, etc. mostly OK
        # Remove MySQL-specific index hints
        q = re.sub(r"\bFORCE INDEX\s*\([^)]*\)", "", q, flags=re.IGNORECASE)
        # WITH ROLLUP not supported; strip it (results lose subtotals)
        q = re.sub(r"\bWITH ROLLUP\b", "", q, flags=re.IGNORECASE)
        # AUTO_INCREMENT → AUTOINCREMENT for DDL
        q = re.sub(r"\bAUTO_INCREMENT\b", "AUTOINCREMENT", q, flags=re.IGNORECASE)
        # ENGINE= … DEFAULT CHARSET= … — remove
        q = re.sub(r"ENGINE\s*=\s*\w+", "", q, flags=re.IGNORECASE)
        q = re.sub(r"DEFAULT CHARSET\s*=\s*\w+", "", q, flags=re.IGNORECASE)
        # UNIQUE KEY → nothing extra needed in SQLite CREATE TABLE
        # DATETIME DEFAULT CURRENT_TIMESTAMP — OK in SQLite
        # ON DUPLICATE KEY UPDATE → INSERT OR REPLACE (handled separately)
        return q

    def db_query(self, query, params=None):
        """Execute a SELECT and return list of dicts."""
        if self.backend == "mysql":
            conn = self.get_fresh_mysql_conn()
            cur = conn.cursor(dictionary=True)
            cur.execute(query, params or ())
            rows = cur.fetchall()
            cur.close(); conn.close()
            return [{k: (float(v) if isinstance(v, Decimal) else v) for k, v in r.items()} for r in rows]
        else:
            q = self._adapt_sql(query)
            p = [str(x) if isinstance(x, (bytes,)) else x for x in (params or ())]
            conn = self._get_sqlite_conn()
            cur = conn.execute(q, p)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
            conn.close()
            return rows

    # Keep old name as alias for backward compat
    def mysql_query(self, query, params=None):
        return self.db_query(query, params)

    def db_execute(self, query, params=None):
        if self.backend == "mysql":
            conn = self.get_fresh_mysql_conn()
            cur = conn.cursor()
            cur.execute(query, params or ())
            conn.commit(); cur.close(); conn.close()
        else:
            q = self._adapt_sql(query)
            p = list(params or ())
            conn = self._get_sqlite_conn()
            try:
                conn.execute(q, p)
                conn.commit()
            except Exception:
                pass  # DDL differences – best effort
            conn.close()

    def mysql_execute(self, query, params=None):
        self.db_execute(query, params)

    def db_executemany(self, query, values):
        if self.backend == "mysql":
            conn = self.get_fresh_mysql_conn()
            cur = conn.cursor()
            cur.executemany(query, values)
            conn.commit(); cur.close(); conn.close()
        else:
            q = self._adapt_sql(query)
            # Convert ON DUPLICATE KEY UPDATE → INSERT OR REPLACE
            import re
            q = re.sub(r"ON DUPLICATE KEY UPDATE.*", "", q, flags=re.IGNORECASE|re.DOTALL).strip()
            q = q.replace("INSERT INTO", "INSERT OR REPLACE INTO")
            conn = self._get_sqlite_conn()
            conn.executemany(q, values)
            conn.commit(); conn.close()

    def mysql_executemany(self, query, values):
        self.db_executemany(query, values)

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    # ── SQLite schema bootstrap ────────────────────────────
    def _sqlite_init_schema(self):
        conn = self._get_sqlite_conn()
        ddl = [
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""",
            """CREATE TABLE IF NOT EXISTS user_activity_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT,
                details TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""",
            """CREATE TABLE IF NOT EXISTS city_dim (
                city_id INTEGER PRIMARY KEY AUTOINCREMENT,
                city_name TEXT NOT NULL UNIQUE)""",
            """CREATE TABLE IF NOT EXISTS crime_type_dim (
                crime_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
                crime_type_name TEXT NOT NULL UNIQUE)""",
            """CREATE TABLE IF NOT EXISTS date_dim (
                date_id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_date TEXT NOT NULL UNIQUE,
                year INTEGER, month INTEGER, month_name TEXT,
                day INTEGER, quarter_name TEXT, season_name TEXT, day_of_week TEXT)""",
            """CREATE TABLE IF NOT EXISTS crime_fact (
                case_id TEXT PRIMARY KEY,
                city_id INTEGER, crime_type_id INTEGER, date_id INTEGER,
                weapon_used TEXT, victim_age INTEGER, victim_gender TEXT,
                suspect_age INTEGER, suspect_gender TEXT, location_type TEXT,
                time_of_day TEXT, reported_to_police INTEGER, outcome TEXT,
                crime_severity REAL, crime_count INTEGER)""",
            """CREATE TABLE IF NOT EXISTS import_log (
                import_id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT,
                rows_affected INTEGER,
                imported_at DATETIME)""",
        ]
        for q in ddl:
            try: conn.execute(q)
            except Exception: pass
        conn.commit()
        # Views (SQLite supports CREATE VIEW)
        views = [
            """CREATE VIEW IF NOT EXISTS city_crime_summary AS
               SELECT c.city_name, SUM(f.crime_count) AS total_crimes
               FROM crime_fact f JOIN city_dim c ON f.city_id = c.city_id
               GROUP BY c.city_name ORDER BY total_crimes DESC""",
            """CREATE VIEW IF NOT EXISTS yearly_crime_summary AS
               SELECT d.year, SUM(f.crime_count) AS total_crimes
               FROM crime_fact f JOIN date_dim d ON f.date_id = d.date_id
               GROUP BY d.year ORDER BY d.year""",
        ]
        for v in views:
            try: conn.execute(v)
            except Exception: pass
        conn.commit()
        conn.close()

    # ── MongoDB connect ────────────────────────────────────
    def connect_mongo(self):
        if not PYMONGO_AVAILABLE:
            raise RuntimeError("pymongo not installed")
        if self.mongo_client is None:
            self.mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            self.mongo_client.admin.command("ping")          # raises if offline
            self.mongo_collection_reports  = self.mongo_client[MONGO_DB][MONGO_COLLECTION_REPORTS]
            self.mongo_collection_evidence = self.mongo_client[MONGO_DB][MONGO_COLLECTION_EVIDENCE]

    # ── 1. SECURITY & AUTH ─────────────────────────────────
    def setup_security_schema(self):
        if self.backend == "mysql":
            self.db_execute("CREATE TABLE IF NOT EXISTS users (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(100) NOT NULL, email VARCHAR(100) NOT NULL UNIQUE, password_hash VARCHAR(255) NOT NULL, role VARCHAR(20) NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
            self.db_execute("CREATE TABLE IF NOT EXISTS user_activity_log (log_id INT AUTO_INCREMENT PRIMARY KEY, user_id INT, action VARCHAR(255), details TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL)")
        # SQLite tables already created in _sqlite_init_schema
        if not self.db_query("SELECT * FROM users LIMIT 1"):
            self.register_user("System Admin",  "admin@gmail.com", "admin123", "Admin")
            self.register_user("Analyst User",  "user@gmail.com",  "user123",  "User")
        # ── Auto-seed demo crime data or load from local CSV or Google Drive URL ──
        if self.db_query("SELECT COUNT(*) as c FROM crime_fact")[0]['c'] == 0:
            dataset_loaded = False
            
            # 1. Try importing from local CSV file (if it exists in the repository/deploy folder)
            local_csv = Path(__file__).parent / "pakistan_crime_2012_2026_prediction.csv"
            if local_csv.exists():
                try:
                    self.import_dataset(str(local_csv), 1)  # Admin user ID = 1
                    dataset_loaded = True
                except Exception:
                    pass
            
            # 2. Try importing from Google Drive URL if local CSV is missing
            if not dataset_loaded and DATASET_URL:
                try:
                    direct_url = get_gdrive_direct_url(DATASET_URL)
                    self.import_dataset(direct_url, 1)  # Admin user ID = 1
                    dataset_loaded = True
                except Exception:
                    # Fallback if link fails
                    pass
                    
            if not dataset_loaded:
                self._seed_demo_data()

    def _seed_demo_data(self):
        """Insert realistic demo data so dashboard works without a CSV upload."""
        import random as _rnd
        cities      = list(CITY_COORDS.keys())
        crime_types = ["Theft", "Robbery", "Assault", "Fraud", "Kidnapping",
                       "Cybercrime", "Drug Trafficking", "Murder", "Burglary", "Vandalism"]
        weapons     = ["None", "Knife", "Gun", "Blunt Object", "Unknown"]
        outcomes    = ["Arrested", "Escaped", "Under Investigation", "Acquitted"]
        locations   = ["Street", "Home", "Market", "Office", "School"]
        times       = ["Morning", "Afternoon", "Evening", "Night"]

        ph = "?" if self.backend == "sqlite" else "%s"
        for city in cities:
            self.db_execute(
                f"INSERT OR IGNORE INTO city_dim(city_name) VALUES ({ph})" if self.backend == "sqlite"
                else f"INSERT IGNORE INTO city_dim(city_name) VALUES ({ph})", (city,))
        for ct in crime_types:
            self.db_execute(
                f"INSERT OR IGNORE INTO crime_type_dim(crime_type_name) VALUES ({ph})" if self.backend == "sqlite"
                else f"INSERT IGNORE INTO crime_type_dim(crime_type_name) VALUES ({ph})", (ct,))

        for yr in range(2019, 2025):
            for mo in range(1, 13):
                dt_str = f"{yr}-{mo:02d}-15"
                mname = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][mo-1]
                self.db_execute(
                    "INSERT OR IGNORE INTO date_dim(full_date,year,month,month_name,day) VALUES (?,?,?,?,?)"
                    if self.backend == "sqlite" else
                    "INSERT IGNORE INTO date_dim(full_date,year,month,month_name,day) VALUES (%s,%s,%s,%s,%s)",
                    (dt_str, yr, mo, mname, 15))

        city_map  = {r["city_name"]: r["city_id"]       for r in self.db_query("SELECT * FROM city_dim")}
        crime_map = {r["crime_type_name"]: r["crime_type_id"] for r in self.db_query("SELECT * FROM crime_type_dim")}
        date_map  = {r["full_date"]: r["date_id"]        for r in self.db_query("SELECT * FROM date_dim")}

        ins = ("INSERT OR REPLACE INTO crime_fact VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
               if self.backend == "sqlite" else
               "INSERT INTO crime_fact VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
               "ON DUPLICATE KEY UPDATE city_id=VALUES(city_id)")

        batch, idx = [], 0
        _rnd.seed(42)
        for yr in range(2019, 2025):
            for mo in range(1, 13):
                dt_key = f"{yr}-{mo:02d}-15"
                for _ in range(_rnd.randint(4, 12)):
                    idx += 1
                    batch.append((
                        f"SEED-{idx:06d}",
                        city_map[_rnd.choice(cities)],
                        crime_map[_rnd.choice(crime_types)],
                        date_map.get(dt_key),
                        _rnd.choice(weapons),
                        _rnd.randint(15, 65),
                        _rnd.choice(["Male", "Female"]),
                        _rnd.randint(16, 55),
                        _rnd.choice(["Male", "Female"]),
                        _rnd.choice(locations),
                        _rnd.choice(times),
                        _rnd.randint(0, 1),
                        _rnd.choice(outcomes),
                        round(_rnd.uniform(1.0, 10.0), 1),
                        _rnd.randint(1, 5),
                    ))
        self.db_executemany(ins, batch)

    def register_user(self, name, email, password, role):
        try:
            self.db_execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)" if self.backend == "sqlite"
                else "INSERT INTO users (name, email, password_hash, role) VALUES (%s, %s, %s, %s)",
                (name, email, self.hash_password(password), role)
            )
            return True, "Registration successful!"
        except Exception as e:
            return False, str(e)

    def login(self, email, password):
        q = ("SELECT id, name, email, role FROM users WHERE email=? AND password_hash=?"
             if self.backend == "sqlite"
             else "SELECT id, name, email, role FROM users WHERE email=%s AND password_hash=%s")
        users = self.db_query(q, (email, self.hash_password(password)))
        if users:
            self.log_audit(users[0]['id'], "LOGIN", "System login")
            return users[0]
        return None

    def log_audit(self, user_id, action, details):
        q = ("INSERT INTO user_activity_log (user_id, action, details) VALUES (?, ?, ?)"
             if self.backend == "sqlite"
             else "INSERT INTO user_activity_log (user_id, action, details) VALUES (%s, %s, %s)")
        self.db_execute(q, (user_id, action, details))

    def get_audit_logs(self):
        return self.db_query(
            "SELECT l.timestamp, u.name, u.role, l.action, l.details "
            "FROM user_activity_log l LEFT JOIN users u ON l.user_id = u.id "
            "ORDER BY l.timestamp DESC LIMIT 150"
        )

    def get_all_users(self):
        return self.db_query("SELECT id, name, email, role, created_at FROM users ORDER BY created_at DESC")

    # ── 2. SCHEMA & ETL ────────────────────────────────────
    def create_schema(self, user_id):
        if self.backend == "mysql":
            self.db_execute("DROP TABLE IF EXISTS import_log")
            ddl = [
                "CREATE TABLE IF NOT EXISTS city_dim (city_id INT AUTO_INCREMENT PRIMARY KEY, city_name VARCHAR(100) NOT NULL UNIQUE)",
                "CREATE TABLE IF NOT EXISTS crime_type_dim (crime_type_id INT AUTO_INCREMENT PRIMARY KEY, crime_type_name VARCHAR(100) NOT NULL UNIQUE)",
                "CREATE TABLE IF NOT EXISTS date_dim (date_id INT AUTO_INCREMENT PRIMARY KEY, full_date DATE NOT NULL UNIQUE, year INT, month INT, month_name VARCHAR(20), day INT, quarter_name VARCHAR(10), season_name VARCHAR(20), day_of_week VARCHAR(20))",
                "CREATE TABLE IF NOT EXISTS crime_fact (case_id VARCHAR(60) PRIMARY KEY, city_id INT, crime_type_id INT, date_id INT, weapon_used VARCHAR(100), victim_age INT, victim_gender VARCHAR(20), suspect_age INT, suspect_gender VARCHAR(20), location_type VARCHAR(100), time_of_day VARCHAR(50), reported_to_police INT, outcome VARCHAR(100), crime_severity DECIMAL(10,2), crime_count INT, FOREIGN KEY (city_id) REFERENCES city_dim(city_id), FOREIGN KEY (crime_type_id) REFERENCES crime_type_dim(crime_type_id), FOREIGN KEY (date_id) REFERENCES date_dim(date_id))",
                "CREATE TABLE IF NOT EXISTS import_log (import_id INT AUTO_INCREMENT PRIMARY KEY, action_type VARCHAR(255), rows_affected INT, imported_at DATETIME)",
            ]
            for q in ddl: self.db_execute(q)
            self.db_execute("CREATE OR REPLACE VIEW city_crime_summary AS SELECT c.city_name, SUM(f.crime_count) AS total_crimes FROM crime_fact f JOIN city_dim c ON f.city_id = c.city_id GROUP BY c.city_name ORDER BY total_crimes DESC")
            self.db_execute("CREATE OR REPLACE VIEW yearly_crime_summary AS SELECT d.year, SUM(f.crime_count) AS total_crimes FROM crime_fact f JOIN date_dim d ON f.date_id = d.date_id GROUP BY d.year ORDER BY d.year")
            self.db_execute("DROP PROCEDURE IF EXISTS GetCrimeCountByCity")
            self.db_execute("CREATE PROCEDURE GetCrimeCountByCity(IN p_city VARCHAR(100)) BEGIN SELECT c.city_name, ct.crime_type_name, SUM(f.crime_count) AS total_cases FROM crime_fact f JOIN city_dim c ON f.city_id = c.city_id JOIN crime_type_dim ct ON f.crime_type_id = ct.crime_type_id WHERE c.city_name = p_city GROUP BY c.city_name, ct.crime_type_name ORDER BY total_cases DESC; END")
            self.db_execute("DROP TRIGGER IF EXISTS trg_after_import_insert")
            self.db_execute("CREATE TRIGGER trg_after_import_insert AFTER INSERT ON crime_fact FOR EACH ROW BEGIN INSERT INTO import_log(action_type, rows_affected, imported_at) VALUES ('ROW_INSERT', 1, NOW()); END")
        else:
            # SQLite – re-init (tables are IF NOT EXISTS safe)
            self._sqlite_init_schema()
        self.log_audit(user_id, "SCHEMA_INIT", "Re-initialized system schema")

    def _clean_dataframe(self, df):
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        rename_map = {
            "Crime_Typ": "Crime_Type", 
            "Weapon_U": "Weapon_Used", 
            "Victim_Ge": "Victim_Gender", 
            "Suspect_A": "Suspect_Age", 
            "Suspect_G": "Suspect_Gender", 
            "Location_T": "Location_Type", 
            "Time_of_D": "Time_of_Day", 
            "Reported_t": "Reported_to_Police", 
            "Reported_to_Police_Hours": "Reported_to_Police",
            "Crime_Count_This_Year": "Crime_Count",
            "Crime_Sev": "Crime_Severity"
        }
        df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month
            df["Month_Name"] = df["Date"].dt.month_name()
            df["Day"] = df["Date"].dt.day
        if "Crime_Count" not in df.columns: df["Crime_Count"] = 1
        if "Case_ID" not in df.columns: df["Case_ID"] = [f"CASE-{i+1:06d}" for i in range(len(df))]
        return df.dropna(subset=["Date", "Year"]).reset_index(drop=True)

    def import_dataset(self, file_path, user_id):
        is_csv = file_path.lower().endswith(".csv") or "drive.google.com" in file_path or file_path.startswith("http")
        df = self._clean_dataframe(pd.read_csv(file_path) if is_csv else pd.read_excel(file_path))
        self.create_schema(user_id)

        cities = sorted(df["City"].dropna().astype(str).unique().tolist()) if "City" in df.columns else []
        crime_types = sorted(df["Crime_Type"].dropna().astype(str).unique().tolist()) if "Crime_Type" in df.columns else []

        ph = "?" if self.backend == "sqlite" else "%s"
        for city in cities:
            self.db_execute(f"INSERT OR IGNORE INTO city_dim(city_name) VALUES ({ph})" if self.backend == "sqlite"
                           else f"INSERT IGNORE INTO city_dim(city_name) VALUES ({ph})", (city,))
        for crime in crime_types:
            self.db_execute(f"INSERT OR IGNORE INTO crime_type_dim(crime_type_name) VALUES ({ph})" if self.backend == "sqlite"
                           else f"INSERT IGNORE INTO crime_type_dim(crime_type_name) VALUES ({ph})", (crime,))

        date_df = df[["Date", "Year", "Month", "Month_Name", "Day"]].drop_duplicates()
        for _, row in date_df.iterrows():
            dt = pd.to_datetime(row["Date"]).date()
            if self.backend == "sqlite":
                self.db_execute("INSERT OR IGNORE INTO date_dim(full_date, year, month, month_name, day) VALUES (?,?,?,?,?)",
                                (str(dt), int(row["Year"]), int(row["Month"]), str(row["Month_Name"]), int(row["Day"])))
            else:
                self.db_execute("INSERT IGNORE INTO date_dim(full_date, year, month, month_name, day) VALUES (%s,%s,%s,%s,%s)",
                                (dt, int(row["Year"]), int(row["Month"]), str(row["Month_Name"]), int(row["Day"])))

        city_lookup  = {r["city_name"]: r["city_id"]       for r in self.db_query("SELECT * FROM city_dim")}
        crime_lookup = {r["crime_type_name"]: r["crime_type_id"] for r in self.db_query("SELECT * FROM crime_type_dim")}
        date_lookup  = {r["full_date"]: r["date_id"]        for r in self.db_query("SELECT * FROM date_dim")}

        if self.backend == "sqlite":
            insert_sql = ("INSERT OR REPLACE INTO crime_fact(case_id, city_id, crime_type_id, date_id, weapon_used, "
                          "victim_age, victim_gender, suspect_age, suspect_gender, location_type, time_of_day, "
                          "reported_to_police, outcome, crime_severity, crime_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
        else:
            insert_sql = ("INSERT INTO crime_fact(case_id, city_id, crime_type_id, date_id, weapon_used, "
                          "victim_age, victim_gender, suspect_age, suspect_gender, location_type, time_of_day, "
                          "reported_to_police, outcome, crime_severity, crime_count) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                          "ON DUPLICATE KEY UPDATE city_id = VALUES(city_id)")

        batch = []
        for _, r in df.iterrows():
            dt = pd.to_datetime(r["Date"], errors="coerce")
            dt_key = str(dt.date()) if pd.notna(dt) else None
            batch.append((
                str(r.get("Case_ID")),
                city_lookup.get(str(r.get("City"))),
                crime_lookup.get(str(r.get("Crime_Type"))),
                date_lookup.get(dt_key),
                str(r.get("Weapon_Used", "Unknown")),
                int(r.get("Victim_Age")) if pd.notna(r.get("Victim_Age")) else None,
                str(r.get("Victim_Gender", "Unknown")),
                int(r.get("Suspect_Age")) if pd.notna(r.get("Suspect_Age")) else None,
                str(r.get("Suspect_Gender", "Unknown")),
                str(r.get("Location_Type", "Unknown")),
                str(r.get("Time_of_Day", "Unknown")),
                int(r.get("Reported_to_Police")) if pd.notna(r.get("Reported_to_Police")) else None,
                str(r.get("Outcome", "Unknown")),
                float(r.get("Crime_Severity", 5.0)),
                int(r.get("Crime_Count", 1)),
            ))

        self.db_executemany(insert_sql, batch)
        log_ph = "?,?,?" if self.backend == "sqlite" else "%s,%s,%s"
        self.db_execute(f"INSERT INTO import_log(action_type, rows_affected, imported_at) VALUES ({log_ph})",
                        ("FILE_IMPORT", len(df), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.log_audit(user_id, "DATA_IMPORT", f"Imported {len(df)} rows")
        return len(df)

    # ── 3. DATABASE QUERIES ────────────────────────────────
    def get_dashboard_kpis(self):
        return {
            "Total Incidents": self.db_query("SELECT SUM(crime_count) as v FROM crime_fact")[0]['v'] or 0,
            "Total Cities":    self.db_query("SELECT COUNT(*) as v FROM city_dim")[0]['v'],
            "Crime Types":     self.db_query("SELECT COUNT(*) as v FROM crime_type_dim")[0]['v'],
        }

    def run_dynamic_query(self, cols, group_by, filter_city, user_id):
        select = ", ".join(cols) if cols else "*"
        where  = f"WHERE c.city_name = '{filter_city}'" if filter_city != "All" else ""
        group  = f"GROUP BY {group_by}" if group_by != "None" else ""
        q = f"SELECT {select} FROM crime_fact f LEFT JOIN city_dim c ON f.city_id = c.city_id LEFT JOIN crime_type_dim ct ON f.crime_type_id = ct.crime_type_id LEFT JOIN date_dim d ON f.date_id = d.date_id {where} {group} LIMIT 1000"
        self.log_audit(user_id, "DYNAMIC_QUERY", f"Ran dynamic query grouped by {group_by}")
        return self.db_query(q)

    def get_explain_plan(self, query, user_id):
        self.log_audit(user_id, "EXPLAIN_PLAN", "Executed Explain Plan")
        if self.backend == "mysql":
            return self.db_query(f"EXPLAIN {query}")
        else:
            return self.db_query(f"EXPLAIN QUERY PLAN {query}")

    def call_procedure_city(self, city, user_id):
        self.log_audit(user_id, "PROCEDURE_CALL", f"Called GetCrimeCountByCity for {city}")
        if self.backend == "mysql":
            conn = self.get_fresh_mysql_conn()
            cursor = conn.cursor(dictionary=True)
            cursor.callproc('GetCrimeCountByCity', [city])
            res = []
            for r in cursor.stored_results(): res.extend(r.fetchall())
            cursor.close()
            conn.close()
            return res
        else:
            # SQLite: simulate the stored procedure with an equivalent SELECT
            return self.db_query(
                "SELECT c.city_name, ct.crime_type_name, SUM(f.crime_count) AS total_cases "
                "FROM crime_fact f "
                "JOIN city_dim c ON f.city_id = c.city_id "
                "JOIN crime_type_dim ct ON f.crime_type_id = ct.crime_type_id "
                "WHERE c.city_name = ? "
                "GROUP BY c.city_name, ct.crime_type_name ORDER BY total_cases DESC",
                (city,)
            )

    def get_trigger_logs(self):
        return self.db_query("SELECT * FROM import_log ORDER BY imported_at DESC LIMIT 50")

    # ── 4. MONGO POLYGLOT ──────────────────────────────────
    def add_case_evidence(self, case_id, text, user_name, user_id):
        self.connect_mongo()
        self.mongo_collection_evidence.insert_one({"case_id": case_id, "evidence_text": text, "added_by": user_name, "added_at": datetime.now()})
        self.log_audit(user_id, "EVIDENCE_ADD", f"Added notes to case {case_id}")

    def get_case_evidence(self, case_id):
        self.connect_mongo()
        docs = list(self.mongo_collection_evidence.find({"case_id": case_id}).sort("added_at", -1))
        for d in docs: d.pop("_id", None)
        return docs

    def save_report_to_mongo(self, report_name, rows, user_name):
        self.connect_mongo()
        self.mongo_collection_reports.insert_one({"report_type": report_name, "generated_on": datetime.now(), "generated_by": user_name, "row_count": len(rows), "results": rows})

    def get_saved_reports(self):
        self.connect_mongo()
        docs = list(self.mongo_collection_reports.find({}, {"results": 0}).sort("generated_on", -1).limit(50))
        for d in docs: d.pop("_id", None)
        return docs

    # ── 5. ML FORECASTING & PREDICTION ────────────────────
    def yearly_forecast(self, target_year, user_id):
        self.log_audit(user_id, "FORECAST", f"Ran forecast for {target_year}")
        df = pd.DataFrame(self.db_query("SELECT * FROM yearly_crime_summary"))
        if len(df) < 3: raise ValueError("Need 3+ years of data.")
        model = LinearRegression().fit(df[["year"]], df["total_crimes"])
        return df, int(max(0, round(float(model.predict(np.array([[target_year]]))[0]))))

    def train_outcome_model(self, user_id):
        df = pd.DataFrame(self.db_query(
            "SELECT c.city_name as city, ct.crime_type_name as crime_type, f.weapon_used, "
            "f.location_type, f.time_of_day, f.victim_age, f.suspect_age, f.reported_to_police, "
            "f.crime_severity, f.outcome FROM crime_fact f "
            "JOIN city_dim c ON f.city_id = c.city_id "
            "JOIN crime_type_dim ct ON f.crime_type_id = ct.crime_type_id "
            "WHERE f.outcome IS NOT NULL AND f.outcome != 'Unknown'"
        ))
        if df.empty: raise ValueError("No valid training records.")
        features = ["city", "crime_type", "weapon_used", "location_type", "time_of_day",
                    "victim_age", "suspect_age", "reported_to_police", "crime_severity"]
        X, y = df[features], df["outcome"].astype(str)
        prep = ColumnTransformer([
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             ["city", "crime_type", "weapon_used", "location_type", "time_of_day"]),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]),
             ["victim_age", "suspect_age", "reported_to_police", "crime_severity"]),
        ])
        model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        self.outcome_model, self.outcome_model_features = model, features
        self.outcome_model_accuracy = round(float(accuracy_score(y_test, model.predict(X_test))), 4)
        self.log_audit(user_id, "MODEL_TRAIN", f"Trained outcome model (Acc: {self.outcome_model_accuracy})")
        return self.outcome_model_accuracy

    def predict_outcome(self, payload, user_id):
        if self.outcome_model is None: raise ValueError("Model not trained.")
        self.log_audit(user_id, "PREDICT", "Ran outcome prediction")
        return self.outcome_model.predict(pd.DataFrame([{f: payload.get(f) for f in self.outcome_model_features}]))[0]

# ==========================================================
# UI COMPONENTS
# ==========================================================
def login_screen():
    st.markdown("<h1 style='text-align: center; margin-top: 30px;'>🏢 Crime Intelligence Enterprise Suite</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>Advanced Access Gateway - Powered by Zamsha Developers</p>", unsafe_allow_html=True)

    engine = st.session_state.engine
    badge_class = "db-mysql" if engine.backend == "mysql" else "db-sqlite"
    badge_label = "MySQL"   if engine.backend == "mysql" else "SQLite (Cloud Mode)"
    st.markdown(f"<p style='text-align:center;color:#64748b;font-size:0.85rem;'>Database backend: <span class='db-badge {badge_class}'>{badge_label}</span></p>", unsafe_allow_html=True)

    # No column wrapper — works on all screen sizes
    tabs = st.tabs(["🔒 Secure Login", "📝 User Registration"])
    with tabs[0]:
        with st.form("login_form"):
            email = st.text_input("Email / Gmail", placeholder="admin@gmail.com")
            pwd   = st.text_input("Password", type="password", placeholder="admin123")
            submitted = st.form_submit_button("🔐 Authenticate", use_container_width=True)
        # Handle outside the form to avoid mobile "submit on Enter" triggering mid-field
        if submitted:
            user = st.session_state.engine.login(email, pwd)
            if user:
                st.session_state.user = user
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Try admin@gmail.com / admin123")
    with tabs[1]:
        with st.form("register_form"):
            n_name  = st.text_input("Full Name",      placeholder="Your full name")
            n_email = st.text_input("Gmail Address",  placeholder="yourname@gmail.com")
            n_pwd   = st.text_input("Password",       type="password", placeholder="Choose a password")
            n_role  = st.selectbox("Role", ["User", "Admin"])
            reg_submitted = st.form_submit_button("✅ Register", use_container_width=True)
        if reg_submitted:
            if not n_name or not n_email or not n_pwd:
                st.error("Please fill in all fields before registering.")
            else:
                s, m = st.session_state.engine.register_user(n_name, n_email, n_pwd, n_role)
                if s: st.success(m + " You can now log in.")
                else: st.error(m)

# ─── ADMIN UI ──────────────────────────────────────────────
def admin_health(engine):
    st.title("1. System Health & Metrics")
    badge = ("🟢 MySQL" if engine.backend == "mysql" else "🔵 SQLite (Cloud)")
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Database Backend</div><div class='metric-val' style='font-size:1.4rem'>{badge}</div></div>", unsafe_allow_html=True)
    try:
        engine.connect_mongo()
        st.markdown("<div class='metric-card'><div class='metric-label'>Mongo Evidence DB</div><div class='metric-val' style='color:#059669'>ONLINE</div></div>", unsafe_allow_html=True)
    except:
        st.markdown("<div class='metric-card'><div class='metric-label'>Mongo Evidence DB</div><div class='metric-val' style='color:#dc2626'>OFFLINE</div></div>", unsafe_allow_html=True)
    try:
        kpis = engine.get_dashboard_kpis()
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Data Rows</div><div class='metric-val'>{kpis['Total Incidents']}</div></div>", unsafe_allow_html=True)
    except:
        st.markdown("<div class='metric-card'><div class='metric-label'>Total Data Rows</div><div class='metric-val'>N/A</div></div>", unsafe_allow_html=True)

def admin_audit(engine):
    st.title("2. Security Audit Logs")
    st.write("Immutable enterprise tracking of all user actions.")
    try: st.dataframe(pd.DataFrame(engine.get_audit_logs()), use_container_width=True, height=500)
    except: st.warning("Schema not initialized.")

def admin_users(engine):
    st.title("3. User Access Management")
    try: st.dataframe(pd.DataFrame(engine.get_all_users()), use_container_width=True)
    except: st.warning("Schema not initialized.")

def admin_schema(engine, user_id):
    st.title("4. Database Schema Deployment")
    st.write("Force initialize or reset the entire schema.")
    if engine.backend == "sqlite":
        st.info("ℹ️ Running in **SQLite (Cloud Mode)**. Stored procedures & MySQL triggers are simulated via equivalent SQL queries.")
    if st.button("Deploy Enterprise Schema", type="primary"):
        with st.spinner("Executing DDL..."):
            engine.create_schema(user_id)
            st.success("Schema, Tables, Views, and Triggers successfully deployed!")

def admin_etl(engine, user_id):
    st.title("5. ETL Data Intake Pipeline")
    uploaded = st.file_uploader("Upload Raw Dataset", type=["csv", "xlsx"])
    if st.button("Run ETL Pipeline", type="primary") and uploaded:
        temp = f"temp_{uploaded.name}"
        with open(temp, "wb") as f: f.write(uploaded.getbuffer())
        with st.spinner("Extracting, Transforming, and Loading..."):
            try:
                rows = engine.import_dataset(temp, user_id)
                st.success(f"ETL Complete! {rows} normalized rows inserted.")
            except Exception as e:
                st.error(f"ETL Error: {e}")

def admin_explain(engine, user_id):
    st.title("6. DB Query Optimizer (EXPLAIN)")
    q = st.text_area("SQL Query", "SELECT * FROM crime_fact LIMIT 10")
    if st.button("Analyze Execution Plan"):
        try: st.dataframe(pd.DataFrame(engine.get_explain_plan(q, user_id)), use_container_width=True)
        except Exception as e: st.error(str(e))

def admin_triggers(engine, user_id):
    st.title("7. Triggers & Stored Procedures")
    if engine.backend == "sqlite":
        st.info("ℹ️ **SQLite Cloud Mode** — Stored procedure is simulated via an equivalent SELECT query. Trigger logs are populated on ETL import.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Test Stored Procedure")
        st.write("Executes `CALL GetCrimeCountByCity` directly.")
        city = st.text_input("Enter City", "Karachi")
        if st.button("Execute Procedure"):
            try: st.dataframe(pd.DataFrame(engine.call_procedure_city(city, user_id)), use_container_width=True)
            except Exception as e: st.error(str(e))
    with col2:
        st.subheader("View Trigger Logs")
        st.write("Reads from `import_log` populated by Triggers.")
        if st.button("Fetch Trigger Logs"):
            try: st.dataframe(pd.DataFrame(engine.get_trigger_logs()), use_container_width=True)
            except Exception as e: st.error(str(e))

def admin_model(engine, user_id):
    st.title("8. ML Model Deployment")
    st.info(f"Status: {'🟢 Deployed' if engine.outcome_model else '🔴 Offline'}")
    if st.button("Train / Retrain Model on Live Data", type="primary"):
        with st.spinner("Training Logistic Regression Model..."):
            try:
                acc = engine.train_outcome_model(user_id)
                st.success(f"Model trained! Accuracy: {acc}")
            except Exception as e:
                st.error(str(e))

# ─── USER UI ───────────────────────────────────────────────
def user_dashboard(engine):
    st.title("1. Zamsha Professional Dashboard")
    try:
        kpis = engine.get_dashboard_kpis()
        # Stack metrics vertically — works on mobile and desktop
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Incidents</div><div class='metric-val'>{kpis['Total Incidents']}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Registered Cities</div><div class='metric-val'>{kpis['Total Cities']}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Crime Categories</div><div class='metric-val'>{kpis['Crime Types']}</div></div>", unsafe_allow_html=True)

        df_city  = pd.DataFrame(engine.db_query("SELECT * FROM city_crime_summary LIMIT 10"))
        df_trend = pd.DataFrame(engine.db_query("SELECT d.year, SUM(f.crime_count) as total FROM crime_fact f JOIN date_dim d ON f.date_id = d.date_id GROUP BY d.year ORDER BY d.year"))

        if not df_city.empty:
            donut = alt.Chart(df_city).mark_arc(innerRadius=60).encode(
                theta=alt.Theta(field="total_crimes", type="quantitative"),
                color=alt.Color(field="city_name", type="nominal", legend=alt.Legend(title="City")),
                tooltip=['city_name', 'total_crimes']
            ).properties(title="Distribution by City", height=280)
            st.altair_chart(donut, use_container_width=True)

        if not df_trend.empty:
            area = alt.Chart(df_trend).mark_area(opacity=0.6, color="#0ea5e9").encode(
                x=alt.X('year:O', title="Year"),
                y=alt.Y('total:Q', title="Total Crimes"),
                tooltip=['year', 'total']
            ).properties(title="Time-Series Trend", height=280)
            st.altair_chart(area, use_container_width=True)
    except Exception as e:
        st.warning(f"Data not available. Please run ETL setup. ({e})")

def user_hotspots(engine):
    st.title("2. 3D Geospatial Hotspots (PyDeck)")
    try:
        df = pd.DataFrame(engine.db_query("SELECT * FROM city_crime_summary"))
        if not df.empty:
            min_crimes = st.slider("Minimum Crime Count", 0, int(df['total_crimes'].max()), 0)
            filtered_df = df[df['total_crimes'] >= min_crimes]
            st.metric("Cities Displayed", len(filtered_df))
            if not filtered_df.empty:
                map_data = []
                for _, r in filtered_df.iterrows():
                    lat, lon = CITY_COORDS.get(r['city_name'], [30.0, 70.0])
                    map_data.append({"lat": lat, "lon": lon, "size": float(r['total_crimes']) * 15, "color": "#ef4444"})
                st.map(pd.DataFrame(map_data), size="size", color="color", zoom=5, use_container_width=True)
            else:
                st.info("No cities meet the filter criteria.")
    except:
        st.warning("Data not available.")

def user_builder(engine, user_id):
    st.title("3. Dynamic Query Builder")
    c1, c2, c3 = st.columns(3)
    cols = c1.multiselect("Columns", ["c.city_name", "ct.crime_type_name", "f.weapon_used", "SUM(f.crime_count) as total"])
    grp  = c2.selectbox("Group By", ["None", "c.city_name", "ct.crime_type_name", "c.city_name, ct.crime_type_name"])
    flt  = c3.selectbox("Filter City", ["All", "Karachi", "Lahore", "Islamabad"])
    if st.button("Generate Report", type="primary"):
        try:
            res_df = pd.DataFrame(engine.run_dynamic_query(cols, grp, flt, user_id))
            st.dataframe(res_df, use_container_width=True)
            st.download_button("📥 Export CSV", res_df.to_csv(index=False), "report.csv", "text/csv")
        except Exception as e:
            st.error(str(e))

def user_advanced(engine, user_id):
    st.title("4. Advanced SQL Analytics")
    st.write("Demonstrates Window Functions (`LAG`, `OVER`) and Hierarchical `ROLLUP`.")
    q_yoy  = "SELECT d.year, SUM(f.crime_count) AS total, LAG(SUM(f.crime_count)) OVER (ORDER BY d.year) AS prev_year, ROUND((SUM(f.crime_count) - LAG(SUM(f.crime_count)) OVER (ORDER BY d.year)) / LAG(SUM(f.crime_count)) OVER (ORDER BY d.year) * 100, 2) AS yoy_growth FROM crime_fact f JOIN date_dim d ON f.date_id = d.date_id GROUP BY d.year ORDER BY d.year"
    q_roll = ("SELECT IFNULL(c.city_name, 'GRAND TOTAL') as city, IFNULL(ct.crime_type_name, 'CITY TOTAL') as crime_type, SUM(f.crime_count) as cases FROM crime_fact f JOIN city_dim c ON f.city_id = c.city_id JOIN crime_type_dim ct ON f.crime_type_id = ct.crime_type_id GROUP BY c.city_name, ct.crime_type_name WITH ROLLUP"
              if engine.backend == "mysql"
              else "SELECT c.city_name as city, ct.crime_type_name as crime_type, SUM(f.crime_count) as cases FROM crime_fact f JOIN city_dim c ON f.city_id = c.city_id JOIN crime_type_dim ct ON f.crime_type_id = ct.crime_type_id GROUP BY c.city_name, ct.crime_type_name ORDER BY city, crime_type")
    tabs = st.tabs(["Year-Over-Year Growth", "Hierarchical Rollup"])
    with tabs[0]:
        if st.button("Run YoY Growth"):
            engine.log_audit(user_id, "ADVANCED_SQL", "Ran YoY")
            try: st.dataframe(pd.DataFrame(engine.db_query(q_yoy)), use_container_width=True)
            except Exception as e: st.error(str(e))
    with tabs[1]:
        if st.button("Run Rollup"):
            engine.log_audit(user_id, "ADVANCED_SQL", "Ran Rollup")
            try: st.dataframe(pd.DataFrame(engine.db_query(q_roll)), use_container_width=True)
            except Exception as e: st.error(str(e))

def user_polyglot(engine, user):
    st.title("5. Polyglot Case Explorer")
    st.write("Relational structured data from **SQLite/MySQL**, combined with unstructured evidence from **MongoDB**.")
    try:
        cases = pd.DataFrame(engine.db_query("SELECT case_id, city_name, crime_type_name, outcome FROM crime_fact f JOIN city_dim c ON f.city_id = c.city_id JOIN crime_type_dim ct ON f.crime_type_id = ct.crime_type_id LIMIT 100"))
        if not cases.empty:
            c_id = st.selectbox("Select Case ID", cases['case_id'])
            st.write(cases[cases['case_id'] == c_id].iloc[0].to_dict())
            ev = st.text_area("Investigator Evidence Notes (MongoDB)")
            if st.button("Save Evidence"):
                try:
                    engine.add_case_evidence(c_id, ev, user['name'], user['id'])
                    st.success("Saved to MongoDB!")
                except Exception as e:
                    st.warning(f"MongoDB offline — note not saved: {e}")
            st.write("---")
            try:
                for d in engine.get_case_evidence(c_id):
                    st.info(f"**{d['added_by']}**: {d['evidence_text']}")
            except:
                st.caption("MongoDB evidence store offline.")
    except:
        st.warning("Data not available.")

def user_predict(engine, user_id):
    st.title("6. Outcome Prediction Lab")
    if not engine.outcome_model:
        st.warning("Admin must train the model first.")
    else:
        try:
            p = {
                'city':              st.selectbox("City", [r['city_name'] for r in engine.db_query("SELECT * FROM city_dim")]),
                'crime_type':        st.selectbox("Type", [r['crime_type_name'] for r in engine.db_query("SELECT * FROM crime_type_dim")]),
                'weapon_used':       'None', 'location_type': 'Street', 'time_of_day': 'Night',
                'victim_age':        st.number_input("Victim Age", 30),
                'suspect_age':       st.number_input("Suspect Age", 25),
                'reported_to_police': 1, 'crime_severity': 7
            }
            if st.button("Predict Outcome", type="primary"):
                if p['suspect_age'] < 18:
                    st.error("Cannot predict arrest: Suspect is underage (under 18 in Pakistan).")
                else:
                    res = engine.predict_outcome(p, user_id)
                    st.success(f"Predicted Outcome: **{res}**")
        except:
            pass

def user_forecast(engine, user_id):
    st.title("7. Time-Series Forecasting")
    y = st.number_input("Target Year", 2024, 2050, 2027)
    if st.button("Run Forecast", type="primary"):
        try:
            df, pred = engine.yearly_forecast(y, user_id)
            st.metric(f"Forecasted Crimes ({y})", pred)
            st.altair_chart(alt.Chart(df).mark_line(point=True).encode(x='year:O', y='total_crimes'), use_container_width=True)
        except Exception as e:
            st.error(str(e))

def user_reports(engine, user):
    st.title("8. Automated Report Center")
    st.write("Save and retrieve reports via MongoDB.")
    if st.button("Save Current City Summary to Mongo"):
        try:
            rows = engine.db_query("SELECT * FROM city_crime_summary")
            engine.save_report_to_mongo("city_summary", rows, user['name'])
            engine.log_audit(user['id'], "REPORT_SAVE", "Saved city summary to Mongo")
            st.success("Saved successfully!")
        except Exception as e:
            st.warning(f"MongoDB offline — {e}")
    st.subheader("Historical Reports")
    try:
        st.dataframe(pd.DataFrame(engine.get_saved_reports()), use_container_width=True)
    except:
        st.warning("No reports found or MongoDB offline.")

# ==========================================================
# MAIN ROUTING
# ==========================================================
def main():
    if "engine" not in st.session_state:
        st.session_state.engine = EnterpriseEngine()
        try:
            st.session_state.engine.setup_security_schema()
        except Exception:
            pass

    if not st.session_state.get("logged_in", False):
        login_screen()
    else:
        user   = st.session_state.user
        engine = st.session_state.engine
        st.sidebar.markdown("<div class='zamsha-sidebar'>⚙️ Engineered by<br>Zamsha Developers</div>", unsafe_allow_html=True)
        st.sidebar.markdown(f"### 👤 {user['name']}")
        st.sidebar.caption(f"{user['email']} | **{user['role']}**")
        badge_class = "db-mysql" if engine.backend == "mysql" else "db-sqlite"
        badge_label = "MySQL"   if engine.backend == "mysql" else "SQLite"
        st.sidebar.markdown(f"<span class='db-badge {badge_class}'>DB: {badge_label}</span>", unsafe_allow_html=True)
        if st.sidebar.button("Logout"):
            engine.log_audit(user['id'], "LOGOUT", "User logged out")
            st.session_state.logged_in = False
            st.rerun()
        st.sidebar.divider()

        if user['role'] == "Admin":
            nav = st.sidebar.radio("Administrator Modules", [
                "1. System Health & Metrics", "2. Security Audit Logs", "3. User Access Management",
                "4. Database Schema Deployment", "5. ETL Data Intake Pipeline", "6. DB Query Optimizer (EXPLAIN)",
                "7. Triggers & Stored Procedures", "8. ML Model Deployment"
            ])
            if nav.startswith("1"):   admin_health(engine)
            elif nav.startswith("2"): admin_audit(engine)
            elif nav.startswith("3"): admin_users(engine)
            elif nav.startswith("4"): admin_schema(engine, user['id'])
            elif nav.startswith("5"): admin_etl(engine, user['id'])
            elif nav.startswith("6"): admin_explain(engine, user['id'])
            elif nav.startswith("7"): admin_triggers(engine, user['id'])
            elif nav.startswith("8"): admin_model(engine, user['id'])
        else:
            nav = st.sidebar.radio("Analyst Modules", [
                "1. Zamsha Professional Dashboard", "2. 3D Geospatial Hotspots (PyDeck)", "3. Dynamic Query Builder",
                "4. Advanced SQL Analytics", "5. Polyglot Case Explorer", "6. Outcome Prediction Lab",
                "7. Time-Series Forecasting", "8. Automated Report Center"
            ])
            if nav.startswith("1"):   user_dashboard(engine)
            elif nav.startswith("2"): user_hotspots(engine)
            elif nav.startswith("3"): user_builder(engine, user['id'])
            elif nav.startswith("4"): user_advanced(engine, user['id'])
            elif nav.startswith("5"): user_polyglot(engine, user)
            elif nav.startswith("6"): user_predict(engine, user['id'])
            elif nav.startswith("7"): user_forecast(engine, user['id'])
            elif nav.startswith("8"): user_reports(engine, user)

if __name__ == "__main__":
    main()
